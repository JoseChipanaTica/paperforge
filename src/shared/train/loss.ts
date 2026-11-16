import { Tensor, add, exp, gradOf, log, make, maxAxis, mul, sub, sum } from "./tensor";

export type LossKind = "cross_entropy" | "mse" | "bce";
export type LossOpts = { smoothing: number; outputIsProbs: boolean };

const P_MIN = 1e-12;
const constant = (data: Float64Array, shape: number[]) => new Tensor(data, [...shape]);
const scalar = (v: number) => new Tensor(Float64Array.of(v), [1]);
const flat = (target: number[] | number[][]) => (target as unknown[]).flat() as number[];

/** Clamp to [1e-12, 1 − 1e-12] so log never sees 0; the gradient passes only where nothing was clamped. */
function clampProb(p: Tensor): Tensor {
  const out = p.data.map((x) => Math.min(1 - P_MIN, Math.max(P_MIN, x)));
  return make(out, [...p.shape], [p], (o) => {
    if (!p.requiresGrad) return;
    const gp = gradOf(p);
    const g = o.grad!;
    for (let i = 0; i < g.length; i++) if (p.data[i] === out[i]) gp[i]! += g[i]!;
  });
}

/** Stable log-softmax over the last axis (the row max is a detached constant). */
function logSoftmax(x: Tensor): Tensor {
  const m = maxAxis(x, 1, true);
  const shifted = sub(x, constant(Float64Array.from(m.data), m.shape));
  return sub(shifted, log(sum(exp(shifted), 1, true)));
}

/**
 * The loss of `output` against `target`, as a scalar tensor on the current tape.
 * Cross-entropy: `output` is [rows, V] logits (or probabilities), `target` one class index per row,
 * averaged over rows; smoothing puts `1 − s` on the true class and `s / (V − 1)` on each other one.
 * MSE: mean squared error. BCE: mean binary cross-entropy (probabilities or logits).
 */
export function lossOf(
  kind: LossKind,
  output: Tensor,
  target: number[] | number[][],
  opts: LossOpts = { smoothing: 0, outputIsProbs: false },
): Tensor {
  const t = flat(target);
  if (kind === "cross_entropy") {
    if (output.shape.length !== 2) throw new Error("cross_entropy: output must be [rows, classes]");
    const [n, v] = output.shape as [number, number];
    if (t.length !== n) throw new Error("cross_entropy: one class index per row");
    const q = new Float64Array(n * v);
    const other = v > 1 ? opts.smoothing / (v - 1) : 0;
    for (let i = 0; i < n; i++) {
      q.fill(other, i * v, (i + 1) * v);
      q[i * v + t[i]!] = 1 - opts.smoothing;
    }
    const logp = opts.outputIsProbs ? log(clampProb(output)) : logSoftmax(output);
    return mul(sum(mul(logp, constant(q, [n, v]))), scalar(-1 / n));
  }
  if (t.length !== output.size) throw new Error(`${kind}: target does not match output`);
  const y = constant(Float64Array.from(t), output.shape);
  const mean = (x: Tensor) => mul(sum(x), scalar(1 / x.size));
  if (kind === "mse") {
    const d = sub(output, y);
    return mean(mul(d, d));
  }
  if (opts.outputIsProbs) {
    const p = clampProb(output);
    const one = constant(new Float64Array(output.size).fill(1), output.shape);
    return mean(mul(add(mul(y, log(p)), mul(sub(one, y), log(sub(one, p)))), scalar(-1)));
  }
  // logits: softplus(x) − x·y, with softplus(x) = m + log(exp(−m) + exp(x − m)), m = max(x, 0) detached
  const m = constant(output.data.map((x) => Math.max(x, 0)), output.shape);
  const softplus = add(m, log(add(exp(mul(m, scalar(-1))), exp(sub(output, m)))));
  return mean(sub(softplus, mul(output, y)));
}
