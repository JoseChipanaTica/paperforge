import type { Tensor } from "./tensor";

export type OptimizerSpec = {
  kind: "sgd" | "momentum" | "adam" | "adamw";
  lr: number;
  momentum: number;
  beta1: number;
  beta2: number;
  eps: number;
  weight_decay: number;
};

const ADAM_NOTE = "On the first step Adam moves every weight by about the learning rate, whatever the size of its gradient.";

/** Rows of the last axis (1-D becomes one row). */
function rows(t: Tensor, values: ArrayLike<number>): number[][] {
  const width = t.shape[t.shape.length - 1] ?? 1;
  const out: number[][] = [];
  for (let i = 0; i < values.length; i += width) out.push(Array.from({ length: width }, (_, j) => values[i + j]!));
  return out;
}

/**
 * One optimizer step from zero state, using each param's `.grad`. Inputs are not mutated:
 * the new values come back in `after`. First step: momentum buffer = g; Adam m̂ = g, v̂ = g²,
 * so Δ = lr·g / (|g| + eps). Only AdamW decays weights (lr·wd·w).
 */
export function step(spec: OptimizerSpec, params: Record<string, Tensor>): { after: Record<string, number[][]>; note: string } {
  const after: Record<string, number[][]> = {};
  for (const [name, w] of Object.entries(params)) {
    const next = new Float64Array(w.size);
    for (let i = 0; i < w.size; i++) {
      const g = w.grad?.[i] ?? 0;
      const x = w.data[i]!;
      let delta: number;
      if (spec.kind === "sgd" || spec.kind === "momentum") delta = spec.lr * g;
      else delta = (spec.lr * g) / (Math.sqrt(g * g) + spec.eps);
      next[i] = x - delta - (spec.kind === "adamw" ? spec.lr * spec.weight_decay * x : 0);
    }
    after[name] = rows(w, next);
  }
  return { after, note: spec.kind === "adam" || spec.kind === "adamw" ? ADAM_NOTE : "" };
}
