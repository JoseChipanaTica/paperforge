import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import type { Architecture, LocatedArchitecture } from "../shared/graph";
import { sharedDecision, type StoredPaper } from "./localStore";
import { openPlan, replaceWithShared, sharedStillApplies, storedEvents, withTimeout } from "./paperOpen";

const SAMPLES = join(import.meta.dir, "../../samples");
const graph = (await Bun.file(join(SAMPLES, "1706.03762.json")).json()) as Architecture;

const located: LocatedArchitecture = { ...graph, arxiv_id: "1706.03762", blocks: graph.blocks.map((b) => ({ ...b, source: null })) };
// Pages that contain the first block's quote verbatim, after some text.
const pages = ["Abstract. Nothing here.", `Intro text. ${graph.blocks[0]!.source_quote} More text.`];

const stored = (over: Partial<StoredPaper> = {}): StoredPaper => ({
  arxiv_id: "1706.03762",
  pages,
  graph: located,
  graph_hash: "old",
  save_token: "tok",
  source: "claude",
  opened_at: 1,
  bytes: 10,
  ...over,
});

describe("openPlan", () => {
  test("a stored paper replays unless the visitor forges again", () => {
    expect(openPlan(stored(), false)).toBe("replay");
    expect(openPlan(stored(), true)).toBe("network");
    expect(openPlan(null, false)).toBe("network");
    expect(openPlan(null, true)).toBe("network");
  });

  test("a replay needs no key: it is decided from the store alone", () => {
    // openPlan takes no key, and the replayed events carry everything handle() needs.
    expect(openPlan(stored({ save_token: null }), false)).toBe("replay");
  });
});

describe("storedEvents", () => {
  test("paper first, then the graph with its source and token", () => {
    const events = storedEvents(stored());
    expect(events).toEqual([
      { type: "paper", arxiv_id: "1706.03762", pages },
      { type: "graph", graph: located, source: "claude", save_token: "tok" },
    ]);
  });

  test("no save_token key without a token", () => {
    const [, g] = storedEvents(stored({ save_token: null, source: "gallery" }));
    expect(g).toEqual({ type: "graph", graph: located, source: "gallery" });
    expect("save_token" in g!).toBe(false);
  });
});

describe("replaceWithShared", () => {
  test("locates the shared quotes against the stored pages, clears the token, takes the new hash and source", () => {
    const next = replaceWithShared(stored(), { graph, source: "gallery" }, "new");
    expect(next.arxiv_id).toBe("1706.03762");
    expect(next.pages).toBe(pages);
    expect(next.graph_hash).toBe("new");
    expect(next.source).toBe("gallery");
    expect(next.save_token).toBeNull();
    expect(next.graph.arxiv_id).toBe("1706.03762");
    expect(next.graph.blocks).toHaveLength(graph.blocks.length);
    expect(next.graph.blocks[0]!.source).toMatchObject({ page: 1 });
    expect(next.graph.blocks[1]!.source).toBeNull();
  });
});

describe("the background check", () => {
  test("offline (no shared graph) keeps the stored paper", () => {
    expect(sharedDecision(stored(), null, null)).toBe("keep");
  });

  test("applies only to the same opening of the same paper, never during a refresh", () => {
    const asked = { id: "1706.03762", seq: 3 };
    expect(sharedStillApplies(asked, { id: "1706.03762", seq: 3, refreshing: false })).toBe(true);
    expect(sharedStillApplies(asked, { id: "1512.03385", seq: 3, refreshing: false })).toBe(false);
    expect(sharedStillApplies(asked, { id: "1706.03762", seq: 4, refreshing: false })).toBe(false); // opened again since
    expect(sharedStillApplies(asked, { id: null, seq: 3, refreshing: false })).toBe(false);
    expect(sharedStillApplies(asked, { id: "1706.03762", seq: 3, refreshing: true })).toBe(false);
  });
});

describe("withTimeout", () => {
  test("a promise that settles in time wins", async () => {
    expect(await withTimeout(Promise.resolve("store"), 50, "fallback")).toBe("store");
  });

  test("a hung promise gives the fallback after the timeout", async () => {
    const t0 = performance.now();
    expect(await withTimeout(new Promise<string>(() => {}), 30, "fallback")).toBe("fallback");
    expect(performance.now() - t0).toBeGreaterThanOrEqual(25);
  });

  test("a rejection gives the fallback", async () => {
    expect(await withTimeout(Promise.reject(new Error("x")), 50, "fallback")).toBe("fallback");
  });
});
