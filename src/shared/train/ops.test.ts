import { expect, test } from "bun:test";
import { gradcheck } from "./gradcheck";
import { TOY_KINDS, type ToyKind, runOp, sinusoidal } from "./ops";
import { Tensor, mul, sum, tensor, toArray, withTape } from "./tensor";

/** Seeded LCG, values in [-1, 1] (or [0.5, 1.5] when positive). */
function rng(seed: number) {
  let s = seed;
  return (shape: number[], positive = false) => {
    const n = shape.reduce((a, b) => a * b, 1);
    const d = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      s = (s * 1664525 + 1013904223) % 4294967296;
      const u = s / 4294967296;
      d[i] = positive ? 0.5 + u : 2 * u - 1;
    }
    return new Tensor(d, shape);
  };
}

type Case = { kind: ToyKind; fn?: string; inputs: Tensor[]; params: Record<string, Tensor>; count?: number };

test("each kind's gradient matches finite differences", () => {
  const r = rng(3);
  const attn = () => ({ Wq: r([4, 4]), Wk: r([4, 4]), Wv: r([4, 4]), Wo: r([4, 4]) });
  const cases: Record<string, Case> = {
    embedding: { kind: "embedding_lookup", inputs: [tensor([[2, 0, 2]])], params: { E: r([3, 4]) } },
    positional: { kind: "positional_encoding", inputs: [r([3, 4])], params: {} },
    positionalLearned: { kind: "positional_encoding", fn: "learned", inputs: [r([3, 4])], params: { P: r([4, 4]) } },
    linear: { kind: "linear", inputs: [r([3, 4])], params: { W: r([4, 2]), b: r([1, 2]) } },
    linearNoBias: { kind: "linear", inputs: [r([3, 4])], params: { W: r([4, 3]) } },
    ...Object.fromEntries(
      ["relu", "gelu", "sigmoid", "tanh", "silu"].map((fn) => [
        `activation ${fn}`,
        { kind: "activation", fn, inputs: [mul(r([3, 4]), tensor([3]))], params: {} } as Case,
      ]),
    ),
    ...Object.fromEntries(
      ["layer", "batch"].map((fn) => [
        `normalization ${fn}`,
        { kind: "normalization", fn, inputs: [r([3, 4])], params: { gamma: r([1, 4], true), beta: r([1, 4]) } } as Case,
      ]),
    ),
    "normalization rms": { kind: "normalization", fn: "rms", inputs: [r([3, 4])], params: { gamma: r([4], true) } },
    residual: { kind: "residual_add", inputs: [r([3, 4]), r([3, 4]), r([3, 4])], params: {} },
    softmax: { kind: "softmax", inputs: [r([3, 4])], params: {} },
    dropout: { kind: "dropout", inputs: [r([3, 4])], params: {} },
    convolution: { kind: "convolution", inputs: [r([4, 5])], params: { W: r([2, 3]), b: r([1, 1]) } },
    poolMax: { kind: "pooling", fn: "max", inputs: [r([4, 5])], params: {} },
    poolAvg: { kind: "pooling", fn: "avg", inputs: [r([4, 5])], params: {} },
    flatten: { kind: "flatten", inputs: [r([2, 3])], params: {} },
    concat: { kind: "concat", inputs: [r([3, 2]), r([3, 3])], params: {} },
  };
  for (const fn of ["", "causal", "cross"])
    for (const count of [1, 2])
      cases[`attention ${fn || "self"} ×${count}`] = {
        kind: "attention",
        fn,
        inputs: fn === "cross" ? [mul(r([3, 4]), tensor([2])), r([2, 4])] : [mul(r([3, 4]), tensor([2]))],
        params: attn(),
        count,
      };

  for (const [name, c] of Object.entries(cases)) {
    const names = Object.keys(c.params);
    // Ids aren't differentiable: the embedding's input stays out of the check.
    const nIn = c.kind === "embedding_lookup" ? 0 : c.inputs.length;
    const f = (xs: Tensor[]) =>
      runOp(
        c.kind,
        c.fn ?? "",
        nIn ? xs.slice(0, nIn) : c.inputs,
        Object.fromEntries(names.map((n, i) => [n, xs[nIn + i]!])),
        c.count ?? 1,
      );
    const all = [...c.inputs.slice(0, nIn), ...names.map((n) => c.params[n]!)];
    // A fixed, non-uniform upstream gradient so wrong-offset backwards cannot hide.
    const k = rng(7)(f(all).shape);
    expect(gradcheck((xs) => mul(f(xs), k), all), name).toBeLessThan(1e-5);
  }
});

test("every toy kind runs", () => {
  expect(TOY_KINDS.length).toBe(14);
  const out = runOp("input", "", [], { values: tensor([[0, 1, 2]]) }, 1);
  expect(toArray(out)).toEqual([[0, 1, 2]]);
  expect(out.requiresGrad).toBe(false);
});

test("causal attention ignores later positions", () => {
  const r = rng(11);
  const params = { Wq: r([4, 4]), Wk: r([4, 4]), Wv: r([4, 4]), Wo: r([4, 4]) };
  const x = r([3, 4]);
  const before = toArray(runOp("attention", "causal", [x], params, 2)) as number[][];
  const changed = new Tensor(Float64Array.from(x.data), [3, 4]);
  for (let j = 0; j < 4; j++) changed.data[8 + j] = 5 - j;
  const after = toArray(runOp("attention", "causal", [changed], params, 2)) as number[][];
  expect(after[0]).toEqual(before[0]!);
  expect(after[2]).not.toEqual(before[2]!);
});

test("softmax stays finite for logits of 1e3", () => {
  const out = runOp("softmax", "", [tensor([[1000, 0, -1000]])], {}, 1);
  expect(Array.from(out.data).every(Number.isFinite)).toBe(true);
  expect(out.data.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
});

test("layer norm of an all-zero row is finite", () => {
  const x = tensor([[0, 0, 0, 0], [1, 2, 3, 4]], true);
  const gamma = tensor([[1, 1, 1, 1]], true);
  const beta = tensor([[0, 0, 0, 0]], true);
  const out = withTape((tape) => {
    const y = runOp("normalization", "layer", [x], { gamma, beta }, 1);
    tape.backward(sum(mul(y, y)));
    return y;
  });
  expect(Array.from(out.data).every(Number.isFinite)).toBe(true);
  expect(Array.from(x.grad!).every(Number.isFinite)).toBe(true);
});

test("activations stay finite for inputs of ±1e3", () => {
  for (const fn of ["relu", "gelu", "sigmoid", "tanh", "silu"]) {
    const x = tensor([[1000, -1000, 0]], true);
    const out = withTape((tape) => {
      const y = runOp("activation", fn, [x], {}, 1);
      tape.backward(sum(y));
      return y;
    });
    expect([...out.data, ...x.grad!].every(Number.isFinite), fn).toBe(true);
  }
});

test("sinusoidal positions follow the paper's formula", () => {
  const pe = toArray(sinusoidal(3, 4)) as number[][];
  expect(pe[0]).toEqual([0, 1, 0, 1]);
  expect(pe[2]![0]).toBeCloseTo(Math.sin(2), 12);
  expect(pe[2]![3]).toBeCloseTo(Math.cos(2 / 100), 12);
});

test("shape problems throw unusable", () => {
  const r = rng(5);
  const attn = { Wq: r([4, 3]), Wk: r([4, 3]), Wv: r([4, 3]), Wo: r([3, 4]) };
  expect(() => runOp("attention", "", [r([3, 4])], attn, 2)).toThrow(/^unusable: /);
  expect(() => runOp("linear", "", [r([3, 4])], { W: r([3, 2]) }, 1)).toThrow(/^unusable: /);
  expect(() => runOp("residual_add", "", [r([3, 4]), r([3, 2])], {}, 1)).toThrow(/^unusable: /);
  expect(() => runOp("linear", "", [r([3, 4])], {}, 1)).toThrow(/^unusable: /);
  expect(() => runOp("activation", "swoosh", [r([3, 4])], {}, 1)).toThrow(/^unusable: /);
  expect(() => runOp("convolution", "", [r([2, 2])], { W: r([3, 3]) }, 1)).toThrow(/^unusable: /);
});
