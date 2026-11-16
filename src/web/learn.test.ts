import { afterEach, describe, expect, test } from "bun:test";
import type { LocatedArchitecture } from "../shared/graph";
import type { AskTurn } from "../shared/learn";
import { InFlight, LearnError, askHistoryPayload, inspectTarget, requestArtifact } from "./learn";

describe("askHistoryPayload", () => {
  test("sends at most the last 10 turns, each clipped to 4000 chars (the server's limits)", () => {
    const turns: AskTurn[] = Array.from({ length: 14 }, (_, i) => ({
      role: i % 2 ? "assistant" : "user",
      text: i === 13 ? "x".repeat(9000) : `turn ${i}`,
    }));
    const sent = askHistoryPayload(turns);
    expect(sent).toHaveLength(10);
    expect(sent[0]!.text).toBe("turn 4"); // a user turn: the window starts on a question
    expect(sent.at(-1)!.text).toHaveLength(4000);
  });
});

describe("inspectTarget", () => {
  test("hover previews other blocks while a block is pinned", () => {
    expect(inspectTarget("ffn", "attention", false)).toBe("ffn");
    expect(inspectTarget(null, "attention", false)).toBe("attention");
  });

  test("the pinned block stays while a question is being typed or answered", () => {
    expect(inspectTarget("ffn", "attention", true)).toBe("attention");
  });
});

describe("InFlight", () => {
  test("a repeat request for the same kind reuses the one in flight", async () => {
    const f = new InFlight<string>();
    let calls = 0;
    const work = () => new Promise<string>((r) => setTimeout(() => r(`done ${++calls}`), 5));
    const [a, b] = await Promise.all([f.run("journey", work), f.run("journey", work)]);
    expect(calls).toBe(1);
    expect(a.value).toBe(b.value);
  });

  test("only the most recent request is current; busy until all finish", async () => {
    const f = new InFlight<string>();
    const slow = f.run("journey", () => new Promise((r) => setTimeout(() => r("w"), 20)));
    const fast = f.run("pytorch", () => new Promise((r) => setTimeout(() => r("p"), 5)));
    expect((await fast).current).toBe(true);
    expect(f.busy).toBe(true);
    expect((await slow).current).toBe(false);
    expect(f.busy).toBe(false);
  });
});

describe("requestArtifact, prefetch", () => {
  const real = globalThis.fetch;
  afterEach(() => {
    globalThis.fetch = real;
  });

  test("existing_only goes in the body only for a prefetch, and not_ready comes back as its code", async () => {
    const bodies: Record<string, unknown>[] = [];
    globalThis.fetch = (async (_url: string, init: RequestInit) => {
      bodies.push(JSON.parse(String(init.body)));
      return new Response(`${JSON.stringify({ type: "error", message: "Not built yet.", code: "not_ready" })}\n`);
    }) as unknown as typeof fetch;
    const g = { arxiv_id: "1706.03762" } as unknown as LocatedArchitecture;
    const err = await requestArtifact("1706.03762", "training", g, "", () => {}, undefined, true).catch((e) => e);
    expect(err).toBeInstanceOf(LearnError);
    expect((err as LearnError).code).toBe("not_ready");
    await requestArtifact("1706.03762", "training", g, "", () => {}).catch(() => {});
    expect(bodies[0]!.existing_only).toBe(true);
    expect("existing_only" in bodies[1]!).toBe(false);
  });
});
