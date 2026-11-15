import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "./graph";
import { ARTIFACT_SCHEMA, LEARN_KINDS, pruneArtifact, type LearnKind } from "./learn";

const SAMPLES = join(import.meta.dir, "../../samples");
const graph = Architecture.parse(await Bun.file(join(SAMPLES, "1706.03762.json")).json());
const ids = new Set(graph.blocks.map((b) => b.id));
const sample = async (kind: LearnKind) => Bun.file(join(SAMPLES, `1706.03762.${kind}.json`)).json();

const blockIds = (kind: LearnKind, a: any): string[] =>
  kind === "pytorch" ? a.block_map.map((m: any) => m.block_id) : a.steps.map((s: any) => s.block_id);

describe("Transformer sample artifacts", () => {
  for (const kind of LEARN_KINDS) {
    test(`${kind} matches its schema and only uses the graph's blocks`, async () => {
      const a = ARTIFACT_SCHEMA[kind].parse(await sample(kind));
      const used = blockIds(kind, a);
      expect(used.filter((id) => !ids.has(id))).toEqual([]);
      if (kind !== "pytorch") expect(new Set(used)).toEqual(ids); // the tour visits every block
    });
  }
});

describe("pruneArtifact", () => {
  test("drops walkthrough steps for blocks that aren't in the graph", async () => {
    const w = await sample("walkthrough");
    const bogus = { block_id: "nope", title: "Nope", narration: "n", shape_change: "-" };
    const pruned = pruneArtifact("walkthrough", { ...w, steps: [...w.steps, bogus] }, graph) as typeof w;
    expect(pruned.steps.length).toBe(w.steps.length);
  });

  test("drops pytorch block_map entries for unknown blocks", async () => {
    const p = await sample("pytorch");
    const pruned = pruneArtifact("pytorch", { ...p, block_map: [...p.block_map, { block_id: "nope", symbol: "X" }] }, graph) as typeof p;
    expect(pruned.block_map).toEqual(p.block_map);
  });
});
