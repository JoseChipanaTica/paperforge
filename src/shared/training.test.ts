import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "./graph";
import { RAISING_TOY, TRANSFORMER_TOY } from "./train/fixtures";
import { train } from "./train/run";
import { TrainingArtifact, cleanTraining, trainingFits } from "./training";

const graph = Architecture.parse(await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json());
const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x));
const allFinite = (m: number[][]) => m.every((r) => r.every(Number.isFinite));

describe("cleanTraining", () => {
  test("the toy Transformer survives cleaning unchanged except lr_toy", () => {
    const input = clone(TRANSFORMER_TOY);
    const out = cleanTraining(clone(input), graph);
    expect(out).not.toBeNull();
    expect({ ...out!, optimizer: { ...out!.optimizer, lr_toy: 0 } }).toEqual({
      ...input,
      optimizer: { ...input.optimizer, lr_toy: 0 },
    });
    expect(() => TrainingArtifact.parse(out)).not.toThrow();
    const t = train(out!);
    expect(t.lossAfter).toBeLessThan(t.lossBefore * 0.99);
  });

  test("oversized matrices are clamped", () => {
    const a = clone(RAISING_TOY);
    const rows = (r: number, c: number, v: number) => Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => v * ((i + j) % 3 - 1)));
    a.toy.nodes[0]!.params[0]!.values = rows(10, 10, 1);
    a.toy.nodes[1]!.params[0]!.values = rows(10, 2, 0.1);
    a.target.values = rows(10, 2, 1);
    a.optimizer.lr_toy = 0.01;
    const out = cleanTraining(a, graph);
    expect(out).not.toBeNull();
    for (const n of out!.toy.nodes)
      for (const p of n.params) {
        expect(p.values.length).toBe(8);
        for (const r of p.values) expect(r.length).toBeLessThanOrEqual(8);
      }
    expect(out!.target.values.length).toBe(8);
  });

  test("vocabulary and heads are clamped", () => {
    const a = clone(TRANSFORMER_TOY);
    a.vocab = [...a.vocab, "x", "y", "z"];
    a.toy.nodes.find((n) => n.id === "enc_attn")!.count = 4;
    const out = cleanTraining(a, graph);
    expect(out).not.toBeNull();
    expect(out!.vocab.length).toBe(8);
    expect(out!.toy.nodes.find((n) => n.id === "enc_attn")!.count).toBe(2);
  });

  test("more than 32 nodes are cut to 32", () => {
    const a = clone(TRANSFORMER_TOY);
    const last = a.toy.nodes.pop()!;
    let prev = a.toy.nodes[a.toy.nodes.length - 1]!.id;
    for (let i = 0; i < 10; i++) {
      a.toy.nodes.push({ id: `drop${i}`, block_id: "final_linear", kind: "dropout", fn: "", inputs: [prev], params: [], count: 1 });
      prev = `drop${i}`;
    }
    a.toy.nodes.push({ ...last, inputs: [prev] });
    const out = cleanTraining(a, graph);
    expect(out === null || out.toy.nodes.length <= 32).toBe(true);
  });

  test("a cycle is unusable (null)", () => {
    const a = clone(TRANSFORMER_TOY);
    a.toy.nodes.find((n) => n.id === "enc_pos")!.inputs = ["enc_attn"];
    expect(cleanTraining(a, graph)).toBeNull();
  });

  test("an input that names no earlier node is unusable", () => {
    const a = clone(TRANSFORMER_TOY);
    a.toy.nodes.find((n) => n.id === "enc_attn")!.inputs = ["nowhere"];
    expect(cleanTraining(a, graph)).toBeNull();
  });

  test("an unknown block's node is dropped", () => {
    const a = clone(TRANSFORMER_TOY);
    a.toy.nodes.splice(3, 0, { id: "stray", block_id: "no_such_block", kind: "dropout", fn: "", inputs: ["enc_pos"], params: [], count: 1 });
    expect(trainingFits(a, graph)).toBe(false);
    const out = cleanTraining(a, graph);
    expect(out).not.toBeNull();
    expect(out!.toy.nodes.some((n) => n.id === "stray")).toBe(false);
    expect(trainingFits(out!, graph)).toBe(true);
  });

  test("a node reading a dropped node becomes a pass-through of what remains", () => {
    const a = clone(TRANSFORMER_TOY);
    // enc_add1 reads [enc_attn, enc_pos]; move enc_attn to an unknown block.
    a.toy.nodes.find((n) => n.id === "enc_attn")!.block_id = "no_such_block";
    const out = cleanTraining(a, graph);
    expect(out).not.toBeNull();
    const add = out!.toy.nodes.find((n) => n.id === "enc_add1")!;
    expect(add.fn).toBe("passthrough");
    expect(add.inputs).toEqual(["enc_pos"]);
  });

  test("an unknown kind becomes a pass-through", () => {
    const a = clone(TRANSFORMER_TOY);
    const relu = a.toy.nodes.find((n) => n.id === "enc_ffn_relu")!;
    relu.kind = "swiglu_gate";
    const out = cleanTraining(a, graph);
    expect(out).not.toBeNull();
    const n = out!.toy.nodes.find((x) => x.id === "enc_ffn_relu")!;
    expect(n).toMatchObject({ kind: "dropout", fn: "passthrough", inputs: ["enc_ffn1"] });
    expect(train(out!).nodes.find((x) => x.id === "enc_ffn_relu")!.passthrough).toBe(true);
  });

  test("a pass-through keeps only its first input", () => {
    const a = clone(TRANSFORMER_TOY);
    const add = a.toy.nodes.find((n) => n.id === "enc_add1")!;
    add.kind = "gated_add";
    const out = cleanTraining(a, graph);
    expect(out).not.toBeNull();
    expect(out!.toy.nodes.find((n) => n.id === "enc_add1")!.inputs).toEqual(["enc_attn"]);
  });

  test("LAMB falls back to SGD with fallback: true", () => {
    const a = clone(TRANSFORMER_TOY);
    a.optimizer.kind = "LAMB";
    const out = cleanTraining(a, graph);
    expect(out).not.toBeNull();
    expect(out!.optimizer.kind).toBe("LAMB");
    expect(out!.optimizer.fallback).toBe(true);
  });

  test("AdamW stays adamw", () => {
    const a = clone(TRANSFORMER_TOY);
    a.optimizer.kind = "AdamW";
    a.optimizer.fallback = true; // Claude's claim is overwritten by what runs
    const out = cleanTraining(a, graph);
    expect(out).not.toBeNull();
    expect(out!.optimizer.kind).toBe("AdamW");
    expect(out!.optimizer.fallback).toBe(false);
  });

  test("Adam eps is clamped away from zero", () => {
    const a = clone(TRANSFORMER_TOY);
    a.optimizer.eps = 0;
    const out = cleanTraining(a, graph);
    expect(out).not.toBeNull();
    expect(out!.optimizer.eps).toBeGreaterThanOrEqual(1e-12);
  });

  test("weights of 1e3 either train finitely or are rejected", () => {
    const a = clone(TRANSFORMER_TOY);
    for (const n of a.toy.nodes) if (n.kind !== "input") for (const p of n.params) p.values = p.values.map((r) => r.map((v) => v * 1e3));
    const out = cleanTraining(a, graph);
    if (out !== null) {
      const t = train(out);
      expect(Number.isFinite(t.lossBefore) && Number.isFinite(t.lossAfter)).toBe(true);
      for (const n of t.nodes) for (const p of n.params) expect(allFinite(p.after)).toBe(true);
    }
  });

  test("a too-large lr_toy is retried at a tenth", () => {
    const a = clone(RAISING_TOY);
    a.optimizer.lr_toy = 0.5; // overshoots; 0.05 descends
    const out = cleanTraining(a, graph);
    expect(out).not.toBeNull();
    expect(out!.optimizer.lr_toy).toBeCloseTo(0.05, 12);
    expect(out!.optimizer.lr).toBe(a.optimizer.lr);
  });

  test("a step that raises the loss keeps lr and reports it", () => {
    const out = cleanTraining(clone(RAISING_TOY), graph);
    expect(out).not.toBeNull();
    expect(out!.optimizer.lr_toy).toBe(RAISING_TOY.optimizer.lr_toy);
    expect(out!.optimizer.lr).toBe(RAISING_TOY.optimizer.lr);
    const t = train(out!);
    expect(t.lossAfter).toBeGreaterThan(t.lossBefore);
  });

  test("target: exactly one of classes / values", () => {
    const both = clone(TRANSFORMER_TOY);
    both.target.values = [[1, 0, 0, 0, 0, 0, 0]];
    expect(cleanTraining(both, graph)).toBeNull();
    const none = clone(TRANSFORMER_TOY);
    none.target.classes = [];
    expect(cleanTraining(none, graph)).toBeNull();
  });

  test("class indices outside the vocabulary or output are unusable", () => {
    const a = clone(TRANSFORMER_TOY);
    a.target.classes = [4, 5, 7];
    expect(cleanTraining(a, graph)).toBeNull();
    const b = clone(TRANSFORMER_TOY);
    b.target.classes = [4, 5, -1];
    expect(cleanTraining(b, graph)).toBeNull();
    const c = clone(TRANSFORMER_TOY);
    c.vocab = [...c.vocab, "extra"]; // vocab 8 but the output is 7 wide
    c.target.classes = [4, 5, 7];
    expect(cleanTraining(c, graph)).toBeNull();
  });

  test("fewer than two graph blocks carrying a node is unusable", () => {
    const a = clone(RAISING_TOY);
    for (const n of a.toy.nodes) n.block_id = "final_linear";
    expect(cleanTraining(a, graph)).toBeNull();
  });

  test("the model's output must survive cleaning as a real node, still last", () => {
    const offGraph = clone(TRANSFORMER_TOY);
    offGraph.toy.nodes.at(-1)!.block_id = "no_such_block";
    expect(cleanTraining(offGraph, graph)).toBeNull();
    const unknown = clone(TRANSFORMER_TOY);
    unknown.toy.nodes.at(-1)!.kind = "sparsemax";
    expect(cleanTraining(unknown, graph)).toBeNull();
    const lostInput = clone(RAISING_TOY);
    const last = lostInput.toy.nodes.at(-1)!;
    lostInput.toy.nodes.find((n) => n.id === last.inputs[0])!.block_id = "no_such_block";
    expect(cleanTraining(lostInput, graph)).toBeNull();
  });

  test("RAISING_TOY parses and fits the sample graph", () => {
    expect(() => TrainingArtifact.parse(RAISING_TOY)).not.toThrow();
    expect(trainingFits(RAISING_TOY, graph)).toBe(true);
  });
});
