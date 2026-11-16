import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../graph";
import { TrainingArtifact, optimizerKind } from "../training";
import { TRANSFORMER_TOY } from "./fixtures";
import { train, type Trace } from "./run";

const graph = Architecture.parse(await Bun.file(join(import.meta.dir, "../../../samples/1706.03762.json")).json());

const node = (t: Trace, id: string) => {
  const n = t.nodes.find((x) => x.id === id);
  if (!n) throw new Error(`no node ${id}`);
  return n;
};
const close = (a: number[][], b: number[][]) => {
  expect(a.length).toBe(b.length);
  a.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(b[i]![j]!, 12)));
};
const clone = <T>(x: T): T => JSON.parse(JSON.stringify(x));

describe("training contract", () => {
  test("TRANSFORMER_TOY parses", () => {
    expect(() => TrainingArtifact.parse(TRANSFORMER_TOY)).not.toThrow();
  });

  test("the fixture stays inside the toy limits and on the sample's blocks", () => {
    const ids = new Set(graph.blocks.map((b) => b.id));
    const { nodes } = TRANSFORMER_TOY.toy;
    expect(nodes.length).toBeLessThanOrEqual(24);
    expect(TRANSFORMER_TOY.vocab.length).toBeLessThanOrEqual(8);
    for (const n of nodes) {
      expect(ids.has(n.block_id)).toBe(true);
      for (const p of n.params) {
        expect(p.values.length).toBeLessThanOrEqual(8);
        for (const row of p.values) expect(row.length).toBeLessThanOrEqual(8);
        if (n.kind !== "input" && p.name !== "gamma") for (const row of p.values) for (const v of row) expect(Math.abs(v)).toBeLessThanOrEqual(0.5);
      }
    }
    // Every block on the journey path carries a node.
    for (const b of graph.blocks) expect(nodes.some((n) => n.block_id === b.id)).toBe(true);
  });

  test("optimizerKind maps the paper's name, else SGD with a fallback", () => {
    expect(optimizerKind("Adam")).toEqual({ kind: "adam", fallback: false });
    expect(optimizerKind("AdamW (decoupled)")).toEqual({ kind: "adamw", fallback: false });
    expect(optimizerKind("SGD with Nesterov momentum")).toEqual({ kind: "momentum", fallback: false });
    expect(optimizerKind("momentum")).toEqual({ kind: "momentum", fallback: false });
    expect(optimizerKind("sgd")).toEqual({ kind: "sgd", fallback: false });
    expect(optimizerKind("LAMB")).toEqual({ kind: "sgd", fallback: true });
  });
});

describe("train", () => {
  test("train is deterministic", () => {
    expect(train(TRANSFORMER_TOY)).toEqual(train(TRANSFORMER_TOY));
  });

  test("one step lowers the toy Transformer's loss", () => {
    const t = train(TRANSFORMER_TOY);
    expect(Number.isFinite(t.lossBefore)).toBe(true);
    expect(t.lossAfter).toBeLessThan(t.lossBefore);
    expect(t.lr).toBe(TRANSFORMER_TOY.optimizer.lr_toy);
    expect(t.note).not.toBe("");
    expect(t.predictionBefore).toEqual(t.nodes[t.nodes.length - 1]!.value);
    expect(t.predictionBefore.length).toBe(3);
    expect(t.predictionBefore[0]!.length).toBe(TRANSFORMER_TOY.vocab.length);
  });

  test("the trace follows the nodes in forward order, with params before/grad/after", () => {
    const t = train(TRANSFORMER_TOY);
    expect(t.nodes.map((n) => n.id)).toEqual(TRANSFORMER_TOY.toy.nodes.map((n) => n.id));
    const lin = node(t, "final_linear");
    const W = lin.params.find((p) => p.name === "W")!;
    expect(W.before).toEqual(TRANSFORMER_TOY.toy.nodes.find((n) => n.id === "final_linear")!.params[0]!.values);
    expect(W.grad.flat().some((g) => g !== 0)).toBe(true);
    expect(W.after).not.toEqual(W.before);
    expect(node(t, "src").params).toEqual([]);
    expect(node(t, "src").kind).toBe("input");
  });

  test("gradient leaving a node equals the gradient arriving at its input", () => {
    const t = train(TRANSFORMER_TOY);
    close(node(t, "softmax").gradIn[0]!, node(t, "final_linear").gradOut);
    close(node(t, "final_linear").gradIn[0]!, node(t, "dec_norm3").gradOut);
  });

  test("residual_add sends the same gradient to both inputs", () => {
    const t = train(TRANSFORMER_TOY);
    const add = node(t, "enc_add1");
    expect(add.gradIn.length).toBe(2);
    close(add.gradIn[0]!, add.gradOut);
    close(add.gradIn[1]!, add.gradOut);
    expect(add.gradOut.flat().some((g) => g !== 0)).toBe(true);
  });

  test("a node feeding two consumers receives the sum of what they send back", () => {
    const t = train(TRANSFORMER_TOY);
    // enc_norm2 feeds both decoder cross-attention (as keys/values) — and only that.
    const cross = node(t, "dec_cross");
    close(node(t, "enc_norm2").gradOut, cross.gradIn[1]!);
    // dec_norm1 feeds dec_cross (queries) and dec_add2 (skip).
    const sum = cross.gradIn[0]!.map((row, i) => row.map((v, j) => v + node(t, "dec_add2").gradIn[0]![i]![j]!));
    close(node(t, "dec_norm1").gradOut, sum);
  });

  test("a shape mismatch throws unusable", () => {
    const a = clone(TRANSFORMER_TOY);
    a.toy.nodes.find((n) => n.id === "final_linear")!.params[0]!.values = [[0.1, 0.2], [0.3, 0.4], [0.1, 0.1]];
    expect(() => train(a)).toThrow(/^unusable:/);
  });

  test("a target that doesn't fit the output throws unusable", () => {
    const a = clone(TRANSFORMER_TOY);
    a.target.classes = [1, 2];
    expect(() => train(a)).toThrow(/^unusable:/);
  });

  test("a non-finite number throws unusable", () => {
    const a = clone(TRANSFORMER_TOY);
    a.toy.nodes.find((n) => n.id === "final_linear")!.params[0]!.values[0]![0] = Number.NaN;
    expect(() => train(a)).toThrow(/^unusable:/);
  });

  test("an input that names no earlier node throws unusable", () => {
    const a = clone(TRANSFORMER_TOY);
    a.toy.nodes.find((n) => n.id === "softmax")!.inputs = ["nowhere"];
    expect(() => train(a)).toThrow(/^unusable:/);
  });

  test("a pass-through node is identity forward and backward", () => {
    const a = clone(TRANSFORMER_TOY);
    const ffn = a.toy.nodes.find((n) => n.id === "enc_ffn")!;
    Object.assign(ffn, { kind: "dropout", fn: "passthrough", params: [] });
    const t = train(a);
    const n = node(t, "enc_ffn");
    expect(n.passthrough).toBe(true);
    close(n.value, node(t, "enc_norm1").value);
    close(n.gradIn[0]!, n.gradOut);
    // A kind the engine lacks is a pass-through too.
    const b = clone(TRANSFORMER_TOY);
    Object.assign(b.toy.nodes.find((x) => x.id === "enc_ffn")!, { kind: "mixture_of_experts", params: [] });
    const u = node(train(b), "enc_ffn");
    expect(u.passthrough).toBe(true);
    expect(u.kind).toBe("dropout");
  });

  test("an unknown optimizer trains with SGD", () => {
    const a = clone(TRANSFORMER_TOY);
    a.optimizer.kind = "LAMB";
    const t = train(a);
    const W = node(t, "final_linear").params[0]!;
    W.before.forEach((row, i) =>
      row.forEach((v, j) => expect(W.after[i]![j]!).toBeCloseTo(v - a.optimizer.lr_toy * W.grad[i]![j]!, 12)),
    );
  });

  test("3-D values are flattened to rows", () => {
    const t = train(TRANSFORMER_TOY);
    for (const n of t.nodes) {
      expect(Array.isArray(n.value[0])).toBe(true);
      expect(typeof n.value[0]![0]).toBe("number");
    }
  });
});
