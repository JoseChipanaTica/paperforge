import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import type { Architecture, LocatedArchitecture } from "../shared/graph";
import type { JourneyArtifact, PytorchArtifact } from "../shared/learn";
import { pickText } from "../shared/translate";
import { LocalStore, graphHash, textHash } from "./localStore";
import { type Backend, memoryBackend, nullBackend } from "./storeBackend";

const SAMPLES = join(import.meta.dir, "../../samples");
const graph = (await Bun.file(join(SAMPLES, "1706.03762.json")).json()) as Architecture;
const journey = (await Bun.file(join(SAMPLES, "1706.03762.journey.json")).json()) as JourneyArtifact;
const pytorch = (await Bun.file(join(SAMPLES, "1706.03762.pytorch.json")).json()) as PytorchArtifact;
const inside = (await Bun.file(join(SAMPLES, "1706.03762.inside.encoder_ffn.json")).json()) as Architecture;

const located: LocatedArchitecture = {
  ...graph,
  arxiv_id: "1706.03762",
  blocks: graph.blocks.map((b, i) => ({ ...b, source: i % 2 ? { page: 2, start: 10 * i, end: 10 * i + 5 } : null })),
};

/** The same object with every key inserted in reverse order, recursively. */
function reversed(v: unknown): unknown {
  if (Array.isArray(v)) return v.map(reversed);
  if (v && typeof v === "object")
    return Object.fromEntries(
      Object.entries(v)
        .reverse()
        .map(([k, x]) => [k, reversed(x)]),
    );
  return v;
}

const paper = (h: string) => ({
  arxiv_id: "1706.03762",
  pages: ["page one", "page two"],
  graph: located,
  graph_hash: h,
  save_token: "tok",
  source: "claude" as const,
});

describe("hashes", () => {
  test("graphHash ignores located sources and key order", async () => {
    const plain = await graphHash(graph);
    expect(plain).toMatch(/^[0-9a-f]{64}$/);
    const { arxiv_id: _, ...withoutId } = located;
    expect(await graphHash(withoutId as LocatedArchitecture)).toBe(plain);
    expect(await graphHash(reversed(graph) as Architecture)).toBe(plain);
    expect(await graphHash({ ...graph, summary: "changed" })).not.toBe(plain);
  });

  test("textHash depends on kind and text, not key order", async () => {
    const text = pickText("graph", graph);
    const h = await textHash("graph", text);
    expect(h).toMatch(/^[0-9a-f]{64}$/);
    expect(await textHash("graph", reversed(text) as typeof text)).toBe(h);
    expect(await textHash("expansion", text)).not.toBe(h);
  });
});

describe("LocalStore", () => {
  test("a paper round-trips and opening bumps opened_at", async () => {
    let t = 1000;
    const backend = memoryBackend();
    const store = new LocalStore(backend, () => t);
    expect(store.available).toBe(true);
    expect(await store.getPaper("1706.03762")).toBeNull();
    const h = await graphHash(located);
    await store.putPaper(paper(h));
    expect(((await backend.get("papers", "1706.03762")) as { opened_at: number }).opened_at).toBe(1000);
    t = 5000;
    const got = await store.getPaper("1706.03762");
    expect(got).not.toBeNull();
    expect(got!.graph).toEqual(located); // located sources survive
    expect(got!.graph.blocks[1]!.source).toEqual({ page: 2, start: 10, end: 15 });
    expect(got!.pages).toEqual(["page one", "page two"]);
    expect(got!.save_token).toBe("tok");
    expect(got!.graph_hash).toBe(h);
    expect(got!.bytes).toBeGreaterThan(0);
    expect(got!.opened_at).toBe(5000);
    expect(((await backend.get("papers", "1706.03762")) as { opened_at: number }).opened_at).toBe(5000);
  });

  test("artifacts are keyed by graph hash, scope and kind", async () => {
    const store = new LocalStore(memoryBackend());
    await store.putArtifact("1706.03762", "hashA", "main", "journey", journey);
    expect(await store.getArtifact("hashA", "main", "journey")).toEqual(journey);
    expect(await store.getArtifact("hashB", "main", "journey")).toBeNull();
    expect(await store.getArtifact("hashA", "encoder_ffn", "journey")).toBeNull();
    expect(await store.getArtifact("hashA", "main", "pytorch")).toBeNull();
    await store.putArtifact("1706.03762", "hashA", "main", "pytorch", pytorch);
    expect(await store.getArtifact("hashA", "main", "pytorch")).toEqual(pytorch);
  });

  test("inside graphs round-trip under kind inside", async () => {
    const store = new LocalStore(memoryBackend());
    await store.putArtifact("1706.03762", "hashA", "encoder_ffn", "inside", inside);
    expect(await store.getArtifact("hashA", "encoder_ffn", "inside")).toEqual(inside);
    expect(await store.getArtifact("hashA", "encoder_self_attention", "inside")).toBeNull();
  });

  test("translations are keyed by text, language and level", async () => {
    const backend = memoryBackend();
    const store = new LocalStore(backend);
    const text = pickText("graph", graph);
    const h = await textHash("graph", text);
    const fr = { ...text, summary: "Un modèle de traduction" };
    await store.putTranslation("1706.03762", h, "fr", "engineer", "graph", fr);
    expect(await store.getTranslation(h, "fr", "engineer", "graph")).toEqual(fr);
    expect(await store.getTranslation(h, "fr", "beginner", "graph")).toBeNull();
    expect(await store.getTranslation(h, "es", "engineer", "graph")).toBeNull();
    expect(await store.getTranslation("other", "fr", "engineer", "graph")).toBeNull();
    await store.putTranslation("1512.03385", h, "fr", "engineer", "graph", fr);
    await store.putTranslation("1512.03385", h, "fr", "engineer", "graph", fr);
    const raw = (await backend.get("translations", [h, "fr", "engineer"])) as { arxiv_ids: string[] };
    expect(raw.arxiv_ids).toEqual(["1706.03762", "1512.03385"]);
  });

  test("an entry that no longer parses is deleted and missed", async () => {
    const backend = memoryBackend();
    const store = new LocalStore(backend);
    const { steps: _, ...broken } = journey;
    await store.putArtifact("1706.03762", "hashA", "main", "journey", broken as unknown as JourneyArtifact);
    expect(await store.getArtifact("hashA", "main", "journey")).toBeNull();
    expect(await backend.get("artifacts", ["hashA", "main", "journey"])).toBeUndefined();

    await backend.put("papers", "x", { arxiv_id: "x", pages: "nope" });
    expect(await store.getPaper("x")).toBeNull();
    expect(await backend.get("papers", "x")).toBeUndefined();

    await backend.put("translations", ["t", "fr", "engineer"], { arxiv_ids: ["x"], text: { summary: 1 } });
    expect(await store.getTranslation("t", "fr", "engineer", "graph")).toBeNull();
    expect(await backend.get("translations", ["t", "fr", "engineer"])).toBeUndefined();
  });

  test("a throwing backend reads as misses and writes as no-ops", async () => {
    const boom = () => {
      throw new Error("boom");
    };
    const throwing: Backend = { get: boom, put: boom, delete: boom, entries: boom, clear: boom };
    const rejecting: Backend = {
      get: () => Promise.reject(new Error("no")),
      put: () => Promise.reject(new Error("no")),
      delete: () => Promise.reject(new Error("no")),
      entries: () => Promise.reject(new Error("no")),
      clear: () => Promise.reject(new Error("no")),
    };
    for (const backend of [throwing, rejecting]) {
      const store = new LocalStore(backend);
      await store.putPaper(paper("h"));
      expect(await store.getPaper("1706.03762")).toBeNull();
      await store.putArtifact("1706.03762", "h", "main", "journey", journey);
      expect(await store.getArtifact("h", "main", "journey")).toBeNull();
      await store.putTranslation("1706.03762", "t", "fr", "engineer", "graph", pickText("graph", graph));
      expect(await store.getTranslation("t", "fr", "engineer", "graph")).toBeNull();
      expect(await store.usage()).toEqual({ papers: 0, bytes: 0 });
      await store.clear();
    }
  });

  test("a quota failure is caught and its name reported by the write helper", async () => {
    const quota = { name: "QuotaExceededError", message: "full" };
    const backend: Backend = { ...memoryBackend(), put: () => Promise.reject(quota) };
    const store = new LocalStore(backend);
    await store.putPaper(paper("h")); // no throw
    expect(await store.write("papers", "x", {})).toEqual({ ok: false, name: "QuotaExceededError", error: quota });
    expect(await new LocalStore(memoryBackend()).write("papers", "x", {})).toEqual({ ok: true });
  });

  test("a paper without a save token still reads back", async () => {
    const backend = memoryBackend();
    const store = new LocalStore(backend);
    const { save_token: _, ...noToken } = paper("h");
    await store.putPaper(noToken);
    expect((await store.getPaper("1706.03762"))?.save_token).toBeUndefined();
    expect(await backend.get("papers", "1706.03762")).toBeDefined();
  });

  test("the null backend makes the store unavailable", async () => {
    const store = new LocalStore(nullBackend);
    expect(store.available).toBe(false);
    await store.putPaper(paper("h"));
    expect(await store.getPaper("1706.03762")).toBeNull();
  });

  test("usage and clear", async () => {
    const store = new LocalStore(memoryBackend());
    expect(await store.usage()).toEqual({ papers: 0, bytes: 0 });
    await store.putPaper(paper("h"));
    const afterPaper = await store.usage();
    expect(afterPaper.papers).toBe(1);
    expect(afterPaper.bytes).toBeGreaterThan(0);
    await store.putArtifact("1706.03762", "h", "main", "journey", journey);
    await store.putTranslation("1706.03762", "t", "fr", "engineer", "graph", pickText("graph", graph));
    const all = await store.usage();
    expect(all.papers).toBe(1);
    expect(all.bytes).toBeGreaterThan(afterPaper.bytes + new TextEncoder().encode(JSON.stringify(journey)).length - 1);
    await store.clear();
    expect(await store.usage()).toEqual({ papers: 0, bytes: 0 });
    expect(await store.getPaper("1706.03762")).toBeNull();
  });
});
