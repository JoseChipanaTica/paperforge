import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "./graph";
import { ARTIFACT_SCHEMA, LEARN_KINDS, cleanJourney, pruneArtifact, type JourneyArtifact, type LearnKind, type View } from "./learn";

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

describe("cleanJourney", () => {
  const view = (v: Partial<View>): View => ({ kind: "tensor", caption: "c", rows: [], cols: [], values: [], dims: "[3, 512]", split: 1, ...v });
  const step = (block_id: string, views: View[] = [view({})]) => ({
    block_id,
    title: "t",
    what_happens: "w",
    narration: "n",
    shape_change: "s",
    views,
  });
  const journey = (steps: JourneyArtifact["steps"], extra: Partial<JourneyArtifact> = {}): JourneyArtifact => ({
    modality: "text",
    analogy: { title: "a", text: "b" },
    intro: "i",
    input: { label: "the cat sat", views: [view({ kind: "tokens", rows: ["the", "cat", "sat"] })] },
    steps,
    output: { label: "o", views: [view({ kind: "probs", rows: ["x"], values: [[1]] })] },
    outro: "o",
    ...extra,
  });
  const [a, b] = [graph.blocks[0]!.id, graph.blocks[1]!.id];
  const only = (v: View) => cleanJourney(journey([step(a, [v])]), graph).steps[0]!.views[0]!;

  test("drops steps for unknown blocks", () => {
    expect(cleanJourney(journey([step(a), step("nope"), step(b)]), graph).steps.map((s) => s.block_id)).toEqual([a, b]);
  });

  test("clamps views to 12×12 and vector rows to 8 numbers", () => {
    const rows = Array.from({ length: 20 }, (_, i) => `r${i}`);
    const v = only(view({ kind: "vectors", rows, values: rows.map(() => Array.from({ length: 30 }, (_, i) => i)) }));
    expect(v.rows.length).toBe(12);
    expect(v.values.length).toBe(12);
    expect(v.values[0]!.length).toBe(8);
    const m = only(view({ kind: "matrix", rows, cols: rows, values: rows.map(() => Array.from({ length: 20 }, () => 0.5)) }));
    expect(m.values.length).toBe(12);
    expect(m.values[0]!.length).toBe(12);
    expect(m.cols.length).toBe(12);
  });

  test("pads ragged rows and zeroes non-finite numbers", () => {
    const m = only(view({ kind: "matrix", rows: ["a", "b", "c"], cols: ["a", "b", "c"], values: [[1, 2, 3], [4], [NaN, Infinity, 5]] }));
    expect(m.values).toEqual([[1, 2, 3], [4, 0, 0], [0, 0, 5]]);
  });

  test("clamps split", () => {
    expect(only(view({ kind: "heads", rows: ["a"], values: [[1, 2]], split: 0 })).split).toBe(1);
    expect(only(view({ kind: "heads", rows: ["a"], values: [[1, 2]], split: 99 })).split).toBe(16);
  });

  test("undrawable views become tensor", () => {
    for (const v of [view({ kind: "matrix", caption: "attn", dims: "[3, 3]" }), view({ kind: "probs", caption: "attn", dims: "[3, 3]" }), view({ kind: "tokens", caption: "attn", dims: "[3, 3]" })]) {
      expect(only(v)).toEqual({ kind: "tensor", caption: "attn", rows: [], cols: [], values: [], dims: "[3, 3]", split: 1 });
    }
  });

  test("keeps at most two views per step, input and output too", () => {
    const three = [view({}), view({}), view({})];
    const j = cleanJourney(journey([step(a, three)], { input: { label: "i", views: three }, output: { label: "o", views: three } }), graph);
    expect(j.steps[0]!.views.length).toBe(2);
    expect(j.input.views.length).toBe(2);
    expect(j.output.views.length).toBe(2);
  });
});
