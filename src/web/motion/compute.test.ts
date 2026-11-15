import { describe, expect, test } from "bun:test";
import type { Op } from "../../shared/motion";
import { computeOp, runChain, type M } from "./compute";

const op = (o: Partial<Op>): Op => ({
  kind: "custom",
  caption: "",
  tokens: [],
  input: [],
  weights: [],
  extra: [],
  fn: "",
  size: 1,
  stride: 1,
  count: 1,
  svg: "",
  ...o,
});
const close = (a: M, b: M) => {
  expect(a.length).toBe(b.length);
  a.forEach((row, i) => row.forEach((v, j) => expect(v).toBeCloseTo(b[i]![j]!, 4)));
};
const sum = (r: number[]) => r.reduce((s, x) => s + x, 0);
const grid = (n: number) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => i * n + j));

describe("convolution family", () => {
  const X = [[1, 2, 3, 0], [0, 1, 2, 3], [3, 0, 1, 2], [2, 3, 0, 1]];
  const diag = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];

  test("a 3×3 kernel writes each window's sum", () => {
    const r = computeOp(op({ kind: "convolution", weights: diag }), X);
    expect(r.ok).toBe(true);
    close(r.output, [[3, 6], [0, 3]]);
    expect(r.stages.positions).toEqual([[0, 0], [0, 1], [1, 0], [1, 1]]);
  });

  test("stride and dilation shrink the output", () => {
    expect(computeOp(op({ kind: "convolution", weights: diag, stride: 2 }), grid(5)).output.length).toBe(2);
    const dilated = computeOp(op({ kind: "convolution", weights: diag, fn: "dilated" }), grid(5));
    expect(dilated.output).toEqual([[0 + 12 + 24]]);
  });

  test("transposed convolution grows to (n−1)·s+k", () => {
    const r = computeOp(op({ kind: "transposed_convolution", weights: diag, stride: 2 }), [[1, 2], [3, 4]]);
    expect(r.output.length).toBe(5);
    expect(r.output[0]!.length).toBe(5);
    expect(r.output[2]![2]).toBe(1 + 4); // overlap of (0,0)'s last tap and (1,1)'s first
  });

  test("pooling picks the max or the mean", () => {
    expect(computeOp(op({ kind: "pooling", fn: "max", size: 2, stride: 2 }), [[1, 3], [2, 4]]).output).toEqual([[4]]);
    expect(computeOp(op({ kind: "pooling", fn: "avg", size: 2, stride: 2 }), [[1, 3], [2, 4]]).output).toEqual([[2.5]]);
    expect(computeOp(op({ kind: "pooling", fn: "max", size: 2, stride: 2 }), [[1, 3], [2, 4]]).stages.picks).toEqual([[1, 1, 0, 0]]);
  });
});

describe("attention", () => {
  const Q = [[1, 0], [0, 1], [1, 1]];
  const K = [[1, 0], [0, 1], [1, 1]];
  const V = [[1, 2], [3, 4], [5, 6]];

  test("weights are a softmax of QKᵀ/√d and mix V", () => {
    const r = computeOp(op({ kind: "attention", weights: Q, extra: [...K, ...V] }), []);
    for (const row of r.stages.weights!) expect(sum(row)).toBeCloseTo(1, 6);
    close(r.stages.scaled!, r.stages.scores!.map((row) => row.map((s) => s / Math.SQRT2)));
    const w = r.stages.weights![0]!;
    close([r.output[0]!], [[w[0]! * 1 + w[1]! * 3 + w[2]! * 5, w[0]! * 2 + w[1]! * 4 + w[2]! * 6]]);
  });

  test("causal attention never looks ahead", () => {
    const r = computeOp(op({ kind: "attention", fn: "causal", weights: Q, extra: [...K, ...V] }), []);
    expect(r.stages.weights![0]!.slice(1)).toEqual([0, 0]);
    expect(r.stages.weights![1]![2]).toBe(0);
  });

  test("identical keys give uniform weights", () => {
    const r = computeOp(op({ kind: "attention", weights: Q, extra: [[1, 1], [1, 1], [1, 1], ...V] }), []);
    for (const w of r.stages.weights![0]!) expect(w).toBeCloseTo(1 / 3, 6);
  });
});

describe("dense ops", () => {
  test("linear: x·W + b", () => {
    const r = computeOp(op({ kind: "linear", weights: [[1, 0, 1], [0, 1, 1]], extra: [[0, 0, 1]] }), [[1, 2]]);
    expect(r.output).toEqual([[1, 2, 4]]);
  });

  test("activations at known points", () => {
    const at = (fn: string, x: number) => computeOp(op({ kind: "activation", fn }), [[x]]).output[0]![0]!;
    expect(at("relu", -1)).toBe(0);
    expect(at("relu", 2)).toBe(2);
    expect(at("sigmoid", 0)).toBeCloseTo(0.5, 6);
    expect(at("tanh", 0)).toBeCloseTo(0, 6);
    expect(at("gelu", 0)).toBeCloseTo(0, 6);
    expect(at("silu", 0)).toBeCloseTo(0, 6);
    expect(at("gelu", 3)).toBeCloseTo(2.996, 2);
  });

  test("layer norm: mean 0, variance 1 per row; rms rescales only; batch per column", () => {
    const x = [[1, 2, 3, 6], [2, 2, 4, 8]];
    for (const row of computeOp(op({ kind: "normalization", fn: "layer" }), x).output) {
      expect(sum(row) / 4).toBeCloseTo(0, 4);
      expect(sum(row.map((v) => v * v)) / 4).toBeCloseTo(1, 3);
    }
    const rms = computeOp(op({ kind: "normalization", fn: "rms" }), x).output[0]!;
    expect(sum(rms) / 4).not.toBeCloseTo(0, 1);
    expect(Math.sqrt(sum(rms.map((v) => v * v)) / 4)).toBeCloseTo(1, 3);
    const batch = computeOp(op({ kind: "normalization", fn: "batch" }), x).output;
    for (let j = 0; j < 4; j++) expect(batch[0]![j]! + batch[1]![j]!).toBeCloseTo(0, 4);
  });

  test("softmax sums to 1", () => {
    expect(sum(computeOp(op({ kind: "softmax" }), [[1, 2, 3]]).output[0]!)).toBeCloseTo(1, 6);
  });

  test("residual add uses the skip input, else the chain's start", () => {
    expect(computeOp(op({ kind: "residual_add", extra: [[1, 1]] }), [[2, 3]]).output).toEqual([[3, 4]]);
    expect(computeOp(op({ kind: "residual_add" }), [[2, 3]], [[10, 10]]).output).toEqual([[12, 13]]);
  });
});

describe("sequence ops", () => {
  test("embedding lookup picks rows by id", () => {
    expect(computeOp(op({ kind: "embedding_lookup", weights: [[0, 0], [1, 1], [2, 2]] }), [[2, 0]]).output).toEqual([[2, 2], [0, 0]]);
  });

  test("positional encoding starts at sin 0 / cos 0", () => {
    const r = computeOp(op({ kind: "positional_encoding" }), [[0, 0, 0, 0], [0, 0, 0, 0]]);
    close([r.stages.pe![0]!], [[0, 1, 0, 1]]);
    expect(r.stages.pe![1]![0]).toBeCloseTo(Math.sin(1), 6);
  });

  test("split heads then concat round-trips", () => {
    const x = [[1, 2, 3, 4], [5, 6, 7, 8]];
    const split = computeOp(op({ kind: "split_heads", count: 2 }), x);
    expect(split.stages.heads).toEqual([[1, 2], [3, 4], [5, 6], [7, 8]]);
    expect(computeOp(op({ kind: "concat" }), split.output).output).toEqual(x);
  });

  test("recurrent: one state per token", () => {
    const r = computeOp(op({ kind: "recurrent", weights: [[0.5, 0], [0, 0.5], [0.1, 0.2], [0.3, 0.4]] }), [[1, 0], [0, 1], [1, 1]]);
    expect(r.ok).toBe(true);
    expect(r.stages.states!.length).toBe(3);
  });

  test("rotary keeps each pair's length", () => {
    const x = [[1, 0, 0.5, 0.5], [1, 0, 0.5, 0.5], [3, 4, 1, 2]];
    const r = computeOp(op({ kind: "rotary_embedding" }), x);
    r.output.forEach((row, t) => {
      for (let i = 0; i < 4; i += 2) expect(Math.hypot(row[i]!, row[i + 1]!)).toBeCloseTo(Math.hypot(x[t]![i]!, x[t]![i + 1]!), 6);
    });
    expect(r.output[0]).toEqual(x[0]!); // position 0 doesn't turn
  });
});

describe("other ops", () => {
  test("dropout zeroes some cells and doubles the rest at p = 0.5, the same way every time", () => {
    const x = [[1, 1, 1, 1, 1, 1, 1, 1]];
    const a = computeOp(op({ kind: "dropout", extra: [[0.5]] }), x);
    const b = computeOp(op({ kind: "dropout", extra: [[0.5]] }), x);
    expect(a.output).toEqual(b.output);
    for (const v of a.output[0]!) expect([0, 2]).toContain(v);
  });

  test("flatten is row-major", () => {
    expect(computeOp(op({ kind: "flatten" }), [[1, 2], [3, 4]]).output).toEqual([[1, 2, 3, 4]]);
  });

  test("patch embedding: (n/size)² patches, each projected", () => {
    const W = [[1, 0], [0, 1], [0, 0], [0, 0]];
    const r = computeOp(op({ kind: "patch_embedding", size: 2, weights: W }), grid(4));
    expect(r.stages.patches!.length).toBe(4);
    expect(r.output).toEqual([[0, 1], [2, 3], [8, 9], [10, 11]]);
  });

  test("a glu gate never makes a value bigger", () => {
    const x = [[2, -3, 0.5]];
    const r = computeOp(op({ kind: "gate", fn: "glu", weights: [[1, 0, 0], [0, 1, 0], [0, 0, 1]] }), x);
    r.output[0]!.forEach((v, i) => expect(Math.abs(v)).toBeLessThanOrEqual(Math.abs(x[0]![i]!)));
  });

  test("moe routing sends each token to its top-k experts", () => {
    const r = computeOp(op({ kind: "moe_routing", count: 2, weights: [[0.1, 0.9, 0.5, 0.2]] }), [[1]]);
    expect([...r.stages.chosen![0]!].sort()).toEqual([1, 2]);
  });

  test("add_noise at ᾱ = 1 is the input", () => {
    expect(computeOp(op({ kind: "add_noise", extra: [[1], [5, 5]] }), [[1, 2]]).output).toEqual([[1, 2]]);
  });

  test("message passing gathers only from neighbours", () => {
    const A = [[0, 1, 0], [1, 0, 1], [0, 1, 0]];
    const r = computeOp(op({ kind: "message_passing", fn: "sum", extra: A }), [[1], [10], [100]]);
    expect(r.output).toEqual([[10], [101], [10]]);
  });
});

describe("chains and failures", () => {
  test("a mismatched shape is ok: false and passes the input through", () => {
    const r = computeOp(op({ kind: "linear", weights: [[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]] }), [[1, 2]]);
    expect(r.ok).toBe(false);
    expect(r.output).toEqual([[1, 2]]);
  });

  test("runChain threads each output into the next op (FFN 4 → 8 → 4)", () => {
    const up = Array.from({ length: 4 }, () => Array(8).fill(0.1));
    const down = Array.from({ length: 8 }, () => Array(4).fill(0.1));
    const chain = runChain([
      op({ kind: "linear", input: [[1, -1, 2, 0]], weights: up }),
      op({ kind: "activation", fn: "relu" }),
      op({ kind: "linear", weights: down }),
    ]);
    expect(chain.map((c) => c.result.ok)).toEqual([true, true, true]);
    expect(chain[1]!.input[0]!.length).toBe(8);
    expect(chain[2]!.result.output[0]!.length).toBe(4);
  });

  test("custom passes its input through", () => {
    expect(computeOp(op({ kind: "custom" }), [[1]]).output).toEqual([[1]]);
  });
});
