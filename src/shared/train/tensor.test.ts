import { expect, test } from "bun:test";
import { gradcheck } from "./gradcheck";
import {
  Tensor, add, concat, div, exp, gather, log, matmul, maxAxis, mean, mul, reshape, slice, sqrt, sub, sum,
  tensor, toArray, transpose, where, withTape,
} from "./tensor";

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

test("matmul forward matches a hand-computed 2×3 · 3×2", () => {
  const out = matmul(tensor([[1, 2, 3], [4, 5, 6]]), tensor([[1, 0], [0, 1], [1, 1]]));
  expect(toArray(out)).toEqual([[4, 5], [10, 11]]);
});

test("every primitive's gradient matches finite differences", () => {
  const r = rng(42);
  const checks: Record<string, [(x: Tensor[]) => Tensor, Tensor[]]> = {
    matmul2: [([a, b]) => matmul(a!, b!), [r([2, 3]), r([3, 4])]],
    matmul3: [([a, b]) => matmul(a!, b!), [r([2, 3, 4]), r([2, 4, 2])]],
    add: [([a, b]) => add(a!, b!), [r([2, 3]), r([2, 3])]],
    addRow: [([a, b]) => add(a!, b!), [r([2, 3]), r([3])]],
    sub: [([a, b]) => sub(a!, b!), [r([2, 3]), r([3])]],
    mul: [([a, b]) => mul(a!, b!), [r([2, 3]), r([2, 3])]],
    mulCol: [([a, b]) => mul(a!, b!), [r([2, 3]), r([2, 1])]],
    div: [([a, b]) => div(a!, b!), [r([2, 3]), r([3], true)]],
    divFull: [([a, b]) => div(a!, b!), [r([2, 3]), r([2, 3], true)]],
    exp: [([a]) => exp(a!), [r([2, 3])]],
    log: [([a]) => log(a!), [r([2, 3], true)]],
    sqrt: [([a]) => sqrt(a!), [r([2, 3], true)]],
    maxAxis: [([a]) => mul(maxAxis(a!, 1), maxAxis(a!, 1)), [r([3, 4])]],
    sumAll: [([a]) => mul(sum(a!), sum(a!)), [r([2, 3])]],
    sumAxis: [([a]) => mul(sum(a!, 1), sum(a!, 1)), [r([2, 3, 2])]],
    mean: [([a]) => mul(mean(a!, 0), mean(a!, 0)), [r([3, 2])]],
    transpose: [([a, b]) => mul(transpose(a!), b!), [r([2, 3, 4]), r([2, 4, 3])]],
    reshape: [([a, b]) => mul(reshape(a!, [3, 2]), b!), [r([2, 3]), r([3, 2])]],
    slice: [([a]) => mul(slice(a!, 1, 1, 3), slice(a!, 1, 0, 2)), [r([2, 4])]],
    concat: [([a, b]) => mul(concat([a!, b!], 1), concat([b!, a!], 1)), [r([2, 2]), r([2, 3])]],
    gather: [([t]) => mul(gather(t!, [2, 0, 2]), gather(t!, [1, 1, 0])), [r([3, 4])]],
    where: [([a]) => mul(where([[true, false], [true, true]], a!, 0.5), a!), [r([2, 2, 2])]],
  };
  for (const [name, [f, inputs]] of Object.entries(checks)) {
    const err = gradcheck(f, inputs);
    expect(err, name).toBeLessThan(1e-6);
  }
});

test("gradients accumulate when a tensor is used twice", () => {
  const x = tensor([[1, 2], [3, 4]], true);
  withTape((tape) => tape.backward(sum(add(x, x))));
  expect(Array.from(x.grad!)).toEqual([2, 2, 2, 2]);
});

test("broadcast add of a row vector sums its gradient over rows", () => {
  const a = tensor([[1, 2, 3], [4, 5, 6]], true);
  const b = tensor([10, 20, 30], true);
  const out = withTape((tape) => {
    const y = add(a, b);
    tape.backward(sum(y));
    return y;
  });
  expect(toArray(out)).toEqual([[11, 22, 33], [14, 25, 36]]);
  expect(Array.from(b.grad!)).toEqual([2, 2, 2]);
  expect(Array.from(a.grad!)).toEqual([1, 1, 1, 1, 1, 1]);
});
