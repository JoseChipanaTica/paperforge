import { expect, test } from "bun:test";
import { gradcheck } from "./gradcheck";
import { lossOf } from "./loss";
import { Tensor, mul, tensor } from "./tensor";

const plain = { smoothing: 0, outputIsProbs: false };
const softmaxRows = (rows: number[][]) =>
  rows.map((r) => {
    const e = r.map((x) => Math.exp(x));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((x) => x / s);
  });

function rng(seed: number) {
  let s = seed;
  return (shape: number[]) => {
    const d = new Float64Array(shape.reduce((a, b) => a * b, 1));
    for (let i = 0; i < d.length; i++) {
      s = (s * 1664525 + 1013904223) % 4294967296;
      d[i] = (2 * s) / 4294967296 - 1;
    }
    return new Tensor(d, shape);
  };
}

test("cross-entropy of a uniform 4-way prediction is ln 4", () => {
  const l = lossOf("cross_entropy", tensor([[0, 0, 0, 0]]), [2], plain);
  expect(l.data[0]).toBeCloseTo(Math.log(4), 4);
});

test("label smoothing 0.1 on a perfect prediction is above 0", () => {
  const l = lossOf("cross_entropy", tensor([[50, 0, 0, 0]]), [0], { smoothing: 0.1, outputIsProbs: false });
  expect(l.data[0]!).toBeGreaterThan(0);
});

test("cross-entropy on softmax probabilities equals cross-entropy on logits", () => {
  const logits = [[1, 2, 0.5], [-1, 0.3, 2]];
  const a = lossOf("cross_entropy", tensor(logits), [1, 2], { smoothing: 0.1, outputIsProbs: false });
  const b = lossOf("cross_entropy", tensor(softmaxRows(logits)), [1, 2], { smoothing: 0.1, outputIsProbs: true });
  expect(b.data[0]).toBeCloseTo(a.data[0]!, 9);
});

test("loss gradients match finite differences", () => {
  const r = rng(7);
  const check = (f: (x: Tensor) => Tensor, x: Tensor) => {
    const c = new Tensor(Float64Array.of(1.7), [1]);
    expect(gradcheck(([t]) => mul(f(t!), c), [x])).toBeLessThan(1e-6);
  };
  check((x) => lossOf("cross_entropy", x, [1, 0, 3], { smoothing: 0.1, outputIsProbs: false }), r([3, 4]));
  check((x) => lossOf("mse", x, [[0.2, -1], [1, 0.5]], plain), r([2, 2]));
  check((x) => lossOf("bce", x, [[1, 0, 1], [0, 1, 1]], plain), r([2, 3]));
  const probs = new Tensor(Float64Array.from([0.2, 0.7, 0.4, 0.9]), [2, 2]);
  check((x) => lossOf("bce", x, [[1, 0], [0, 1]], { smoothing: 0, outputIsProbs: true }), probs);
});

test("bce stays finite on perfect or zero probabilities", () => {
  const l = lossOf("bce", tensor([[1, 0]]), [[0, 1]], { smoothing: 0, outputIsProbs: true });
  expect(Number.isFinite(l.data[0]!)).toBe(true);
});
