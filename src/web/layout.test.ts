import { expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { layoutGraph } from "./layout";

const transformer = Architecture.parse(
  await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json(),
);

test("Transformer: encoder and decoder become two columns, side by side", () => {
  const layout = layoutGraph(transformer);
  expect(layout.columns).toBe(2);
  const enc = layout.blocks.get("encoder_self_attention")!;
  const dec = layout.blocks.get("decoder_masked_attention")!;
  expect(enc.x).toBeLessThan(dec.x);
  expect(enc.rank).toBe(dec.rank); // aligned rows, like Figure 1
  expect(enc.y).toBeCloseTo(dec.y);
});

test("data flows upward", () => {
  const layout = layoutGraph(transformer);
  for (const e of transformer.edges.filter((e) => e.kind !== "cross")) {
    expect(layout.blocks.get(e.to)!.y).toBeGreaterThan(layout.blocks.get(e.from)!.y);
  }
});

test("groups wrap their members", () => {
  const layout = layoutGraph(transformer);
  const decoder = layout.groups.find((g) => g.id === "decoder")!;
  expect(decoder.repeat).toBe(6);
  const ffn = layout.blocks.get("decoder_ffn")!;
  expect(ffn.y).toBeLessThan(decoder.y + decoder.h / 2);
  expect(ffn.y).toBeGreaterThan(decoder.y - decoder.h / 2);
});

test("parallel branches at the same rank spread sideways; cycles don't hang", () => {
  const layout = layoutGraph({
    groups: [],
    blocks: [
      { id: "x", type: "input", group: "" },
      { id: "a", type: "convolution", group: "" },
      { id: "b", type: "pooling", group: "" },
      { id: "cat", type: "concat", group: "" },
    ],
    edges: [
      { from: "x", to: "a", kind: "data" },
      { from: "x", to: "b", kind: "data" },
      { from: "a", to: "cat", kind: "data" },
      { from: "b", to: "cat", kind: "data" },
      { from: "cat", to: "x", kind: "data" }, // bogus back edge
    ],
  });
  const a = layout.blocks.get("a")!;
  const b = layout.blocks.get("b")!;
  expect(layout.blocks.size).toBe(4);
  if (a.rank === b.rank) expect(Math.abs(a.x - b.x)).toBeGreaterThan(1);
});
