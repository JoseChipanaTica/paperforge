import { Tape, type Tensor, sum, withTape } from "./tensor";

/**
 * Max relative error between the tape's gradients of `sum(f(inputs))` and central finite differences.
 * Marks every input as requiring grad and overwrites its `.grad`.
 */
export function gradcheck(f: (inputs: Tensor[]) => Tensor, inputs: Tensor[], eps = 1e-5): number {
  for (const t of inputs) {
    t.requiresGrad = true;
    t.grad = null;
  }
  withTape((tape: Tape) => tape.backward(sum(f(inputs))));
  const value = () => f(inputs).data.reduce((a, b) => a + b, 0);
  let worst = 0;
  for (const t of inputs) {
    for (let i = 0; i < t.size; i++) {
      const keep = t.data[i]!;
      t.data[i] = keep + eps;
      const hi = value();
      t.data[i] = keep - eps;
      const lo = value();
      t.data[i] = keep;
      const numeric = (hi - lo) / (2 * eps);
      const analytic = t.grad?.[i] ?? 0;
      worst = Math.max(worst, Math.abs(analytic - numeric) / Math.max(1, Math.abs(analytic), Math.abs(numeric)));
    }
  }
  return worst;
}
