import { expect, test } from "bun:test";
import { type OptimizerSpec, step } from "./optim";
import { Tensor, tensor } from "./tensor";

const base: OptimizerSpec = { kind: "sgd", lr: 0.1, momentum: 0.9, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay: 0.01 };
const param = (w: number[], g: number[]) => {
  const t = tensor([w], true);
  t.grad = Float64Array.from(g);
  return t;
};

test("sgd: w − lr·g", () => {
  const { after, note } = step(base, { W: param([1, -2], [0.5, 4]) });
  expect(after.W![0]![0]).toBeCloseTo(0.95, 12);
  expect(after.W![0]![1]).toBeCloseTo(-2.4, 12);
  expect(note).toBe("");
});

test("step does not mutate its inputs", () => {
  const w = param([1, 2], [1, 1]);
  step(base, { W: w });
  expect(Array.from(w.data)).toEqual([1, 2]);
});

test("momentum first step equals sgd", () => {
  const p = { W: param([1, -2], [0.5, 4]) };
  expect(step({ ...base, kind: "momentum" }, p).after).toEqual(step(base, p).after);
});

test("adam first step moves each weight by ≈ lr·sign(g)", () => {
  const { after, note } = step({ ...base, kind: "adam", lr: 0.01 }, { W: param([1, 1], [0.001, 10]) });
  expect(Math.abs(1 - after.W![0]![0]!)).toBeCloseTo(0.01, 4);
  expect(Math.abs(1 - after.W![0]![0]!) / 0.01).toBeGreaterThan(0.99);
  expect(Math.abs(1 - after.W![0]![1]!) / 0.01).toBeGreaterThan(0.99);
  expect(Math.abs(1 - after.W![0]![1]!) / 0.01).toBeLessThan(1.01);
  expect(note).toContain("about the learning rate");
});

test("adamw also decays the weight by lr·wd·w", () => {
  const adam = step({ ...base, kind: "adam", lr: 0.01 }, { W: param([2], [3]) }).after.W![0]![0]!;
  const adamw = step({ ...base, kind: "adamw", lr: 0.01, weight_decay: 0.1 }, { W: param([2], [3]) }).after.W![0]![0]!;
  expect(adam - adamw).toBeCloseTo(0.01 * 0.1 * 2, 12);
});
