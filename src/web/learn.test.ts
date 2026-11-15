import { describe, expect, test } from "bun:test";
import type { AskTurn } from "../shared/learn";
import { InFlight, askHistoryPayload, inspectTarget } from "./learn";

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
    const [a, b] = await Promise.all([f.run("walkthrough", work), f.run("walkthrough", work)]);
    expect(calls).toBe(1);
    expect(a.value).toBe(b.value);
  });

  test("only the most recent request is current; busy until all finish", async () => {
    const f = new InFlight<string>();
    const slow = f.run("walkthrough", () => new Promise((r) => setTimeout(() => r("w"), 20)));
    const fast = f.run("pytorch", () => new Promise((r) => setTimeout(() => r("p"), 5)));
    expect((await fast).current).toBe(true);
    expect(f.busy).toBe(true);
    expect((await slow).current).toBe(false);
    expect(f.busy).toBe(false);
  });
});
