import type { Op } from "../../shared/motion";

/**
 * The arithmetic behind every animation, recomputed in the browser from the small inputs Claude
 * gives, so every number on screen is right and agrees with the next. Each op returns its output
 * plus the intermediates its template shows (stage names in the plan's table). Shapes that don't
 * fit give ok: false and pass the input through, so a chain keeps going.
 */

export type M = number[][];
export interface Result {
  output: M;
  stages: Record<string, M>;
  ok: boolean;
}

const zeros = (r: number, c: number): M => Array.from({ length: r }, () => Array(c).fill(0));
const cols = (m: M) => m[0]?.length ?? 0;
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));
const fail = (input: M): Result => ({ output: input, stages: {}, ok: false });
const done = (output: M, stages: Record<string, M> = {}): Result => ({ output, stages, ok: true });

function matmul(a: M, b: M): M | null {
  if (!a.length || !b.length || cols(a) !== b.length) return null;
  return a.map((row) => Array.from({ length: cols(b) }, (_, j) => row.reduce((s, v, k) => s + v * b[k]![j]!, 0)));
}

function softmaxRow(row: number[]): number[] {
  const finite = row.filter(Number.isFinite);
  if (!finite.length) return row.map(() => 0);
  const top = Math.max(...finite);
  const e = row.map((v) => (Number.isFinite(v) ? Math.exp(v - top) : 0));
  const total = e.reduce((s, v) => s + v, 0);
  return e.map((v) => v / total);
}

const ACTIVATIONS: Record<string, (x: number) => number> = {
  relu: (x) => Math.max(0, x),
  sigmoid,
  tanh: Math.tanh,
  silu: (x) => x * sigmoid(x),
  gelu: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
};

/** A tiny seeded PRNG (mulberry32), so dropout and filler noise look the same every replay. */
function prng(seed: number) {
  let a = seed;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function convolution(op: Op, x: M): Result {
  const k = op.weights;
  const kh = k.length;
  const kw = cols(k);
  const dil = op.fn === "dilated" ? 2 : 1;
  const [eh, ew] = [(kh - 1) * dil + 1, (kw - 1) * dil + 1];
  const s = op.stride;
  if (!kh || x.length < eh || cols(x) < ew) return fail(x);
  const oh = Math.floor((x.length - eh) / s) + 1;
  const ow = Math.floor((cols(x) - ew) / s) + 1;
  const out = zeros(oh, ow);
  const positions: M = [];
  for (let i = 0; i < oh; i++)
    for (let j = 0; j < ow; j++) {
      positions.push([i * s, j * s]);
      for (let a = 0; a < kh; a++) for (let b = 0; b < kw; b++) out[i]![j]! += x[i * s + a * dil]![j * s + b * dil]! * k[a]![b]!;
    }
  return done(out, { sums: out, positions });
}

function transposed(op: Op, x: M): Result {
  const k = op.weights;
  const s = op.stride;
  if (!k.length || !x.length) return fail(x);
  const out = zeros((x.length - 1) * s + k.length, (cols(x) - 1) * s + cols(k));
  const stamps: M = [];
  x.forEach((row, i) =>
    row.forEach((v, j) => {
      stamps.push([i, j, v]);
      k.forEach((krow, a) => krow.forEach((kv, b) => (out[i * s + a]![j * s + b]! += v * kv)));
    }),
  );
  return done(out, { stamps, output: out });
}

function pooling(op: Op, x: M): Result {
  const w = Math.max(op.size, 1);
  const s = op.stride;
  if (x.length < w || cols(x) < w) return fail(x);
  const oh = Math.floor((x.length - w) / s) + 1;
  const ow = Math.floor((cols(x) - w) / s) + 1;
  const out = zeros(oh, ow);
  const picks: M = [];
  for (let i = 0; i < oh; i++)
    for (let j = 0; j < ow; j++) {
      let best = [-Infinity, i * s, j * s];
      let total = 0;
      for (let a = 0; a < w; a++)
        for (let b = 0; b < w; b++) {
          const v = x[i * s + a]![j * s + b]!;
          total += v;
          if (v > best[0]!) best = [v, i * s + a, j * s + b];
        }
      out[i]![j] = op.fn === "avg" ? total / (w * w) : best[0]!;
      picks.push(op.fn === "avg" ? [i * s, j * s, i, j] : [best[1]!, best[2]!, i, j]);
    }
  return done(out, { picks });
}

function attention(op: Op, x: M): Result {
  const q = op.weights;
  const half = Math.floor(op.extra.length / 2);
  const k = op.extra.slice(0, half);
  const v = op.extra.slice(half, half * 2);
  const d = cols(q);
  const scores = half && cols(k) === d ? matmul(q, transpose(k)) : null;
  if (!scores) return fail(x);
  const scaled = scores.map((row) => row.map((sc) => sc / Math.sqrt(d)));
  const masked = op.fn === "causal" ? scaled.map((row, i) => row.map((sc, j) => (j > i ? -Infinity : sc))) : scaled;
  const weights = masked.map(softmaxRow);
  const output = matmul(weights, v);
  if (!output) return fail(x);
  return done(output, { q, k, v, scores, scaled, weights, output });
}

const transpose = (m: M): M => Array.from({ length: cols(m) }, (_, j) => m.map((r) => r[j]!));

function linear(op: Op, x: M): Result {
  const out = matmul(x, op.weights);
  if (!out) return fail(x);
  const bias = op.extra[0];
  const output = bias && bias.length === cols(out) ? out.map((r) => r.map((v, j) => v + bias[j]!)) : out;
  return done(output, { products: out, output });
}

function normalization(op: Op, x: M): Result {
  if (!x.length) return fail(x);
  const eps = 1e-5;
  const affine = (m: M) => {
    const [g, b] = op.extra;
    return m.map((r) => r.map((v, j) => v * (g?.[j] ?? 1) + (b?.[j] ?? 0)));
  };
  if (op.fn === "batch") {
    const t = transpose(x);
    const mean = t.map((c) => c.reduce((s, v) => s + v, 0) / c.length);
    const variance = t.map((c, j) => c.reduce((s, v) => s + (v - mean[j]!) ** 2, 0) / c.length);
    const normalized = x.map((r) => r.map((v, j) => (v - mean[j]!) / Math.sqrt(variance[j]! + eps)));
    return done(affine(normalized), { mean: [mean], variance: [variance], normalized });
  }
  const mean = x.map((r) => (op.fn === "rms" ? 0 : r.reduce((s, v) => s + v, 0) / r.length));
  const variance = x.map((r, i) => r.reduce((s, v) => s + (v - mean[i]!) ** 2, 0) / r.length);
  const normalized = x.map((r, i) => r.map((v) => (v - mean[i]!) / Math.sqrt(variance[i]! + eps)));
  return done(affine(normalized), { mean: mean.map((m) => [m]), variance: variance.map((v) => [v]), normalized });
}

function residual(op: Op, x: M, start: M | undefined): Result {
  const b = op.extra.length ? op.extra : (start ?? []);
  if (b.length !== x.length || cols(b) !== cols(x)) return fail(x);
  const output = x.map((r, i) => r.map((v, j) => v + b[i]![j]!));
  return done(output, { a: x, b, output });
}

function softmax(x: M): Result {
  if (!x.length) return fail(x);
  const exp = x.map((r) => r.map((v) => Math.exp(v - Math.max(...r))));
  const sum = exp.map((r) => [r.reduce((s, v) => s + v, 0)]);
  const output = x.map(softmaxRow);
  return done(output, { exp, sum, output });
}

function embedding(op: Op, x: M): Result {
  const ids = (x[0] ?? []).map((v) => Math.max(0, Math.min(op.weights.length - 1, Math.round(v))));
  if (!ids.length || !op.weights.length) return fail(x);
  const rows = ids.map((id) => op.weights[id]!.slice());
  return done(rows, { ids: [ids], rows });
}

function positional(x: M): Result {
  const d = cols(x);
  if (!d) return fail(x);
  const pe = x.map((_, pos) => Array.from({ length: d }, (_, i) => (i % 2 === 0 ? Math.sin(pos / 10000 ** (i / d)) : Math.cos(pos / 10000 ** ((i - 1) / d)))));
  const output = x.map((r, p) => r.map((v, i) => v + pe[p]![i]!));
  return done(output, { pe, output });
}

function splitHeads(op: Op, x: M): Result {
  const h = Math.max(1, op.count);
  const d = cols(x);
  if (!d || d % h !== 0) return fail(x);
  const w = d / h;
  const heads = x.flatMap((r) => Array.from({ length: h }, (_, k) => r.slice(k * w, (k + 1) * w)));
  return done(x, { heads });
}

function recurrent(op: Op, x: M): Result {
  const dh = cols(op.weights);
  const dx = cols(x);
  if (!dh || op.weights.length !== dx + dh) return fail(x);
  let h = op.extra[0]?.length === dh ? op.extra[0]!.slice() : Array(dh).fill(0);
  const states: M = [];
  for (const row of x) {
    const joined = [...row, ...h];
    h = Array.from({ length: dh }, (_, j) => Math.tanh(joined.reduce((s, v, k) => s + v * op.weights[k]![j]!, 0)));
    states.push(h);
  }
  return done(states, { states });
}

function dropout(op: Op, x: M): Result {
  const p = Math.max(0, Math.min(0.9, op.extra[0]?.[0] ?? 0.5));
  const rand = prng(7);
  const mask = x.map((r) => r.map(() => (rand() < p ? 0 : 1)));
  const output = x.map((r, i) => r.map((v, j) => (v * mask[i]![j]!) / (1 - p)));
  return done(output, { mask, output });
}

function patchEmbedding(op: Op, x: M): Result {
  const p = Math.max(1, op.size);
  if (x.length < p || cols(x) < p) return fail(x);
  const patches: M = [];
  for (let i = 0; i + p <= x.length; i += p)
    for (let j = 0; j + p <= cols(x); j += p) patches.push(x.slice(i, i + p).flatMap((r) => r.slice(j, j + p)));
  const output = matmul(patches, op.weights);
  if (!output) return fail(x);
  return done(output, { patches, output });
}

function gate(op: Op, x: M): Result {
  const pre = op.weights.length ? matmul(x, op.weights) : x;
  if (!pre || cols(pre) !== cols(x)) return fail(x);
  const g = pre.map((r) => r.map((v) => (op.fn === "swiglu" ? ACTIVATIONS.silu!(v) : sigmoid(v))));
  if (op.fn === "lstm" || op.fn === "gru") {
    // One step of the cell: keep `forget` of the old state, write the rest from the new input.
    const c0 = op.extra[0]?.length === cols(x) ? op.extra[0]! : Array(cols(x)).fill(0);
    const inputGate = g.map((r) => r.map((v) => 1 - v));
    const cell = x.map((r, i) => r.map((v, j) => g[i]![j]! * c0[j]! + inputGate[i]![j]! * Math.tanh(v)));
    const output = op.fn === "lstm" ? cell.map((r) => r.map(Math.tanh)) : cell;
    return done(output, { gate: g, signal: x, forget: g, inputGate, cell, output });
  }
  const output = x.map((r, i) => r.map((v, j) => v * Math.min(1, Math.abs(g[i]![j]!))));
  return done(output, { gate: g, signal: x, output });
}

function moe(op: Op, x: M): Result {
  const logits = matmul(x, op.weights);
  if (!logits) return fail(x);
  const scores = logits.map(softmaxRow);
  const k = Math.min(Math.max(1, op.count), cols(scores));
  const chosen = scores.map((r) => r.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i));
  return done(x, { scores, chosen, output: x });
}

function rotary(x: M): Result {
  const d = cols(x);
  if (d < 2) return fail(x);
  const angles = x.map((_, pos) => Array.from({ length: Math.floor(d / 2) }, (_, i) => pos * 10000 ** ((-2 * i) / d)));
  const output = x.map((r, pos) => {
    const out = r.slice();
    angles[pos]!.forEach((t, i) => {
      if (t === 0) return;
      const [a, b] = [r[2 * i]!, r[2 * i + 1]!];
      out[2 * i] = a * Math.cos(t) - b * Math.sin(t);
      out[2 * i + 1] = a * Math.sin(t) + b * Math.cos(t);
    });
    return out;
  });
  return done(output, { angles, output });
}

function addNoise(op: Op, x: M): Result {
  const abar = Math.max(0, Math.min(1, op.extra[0]?.[0] ?? 0.5));
  const rand = prng(11);
  const given = op.extra.slice(1);
  const noise = x.map((r, i) => r.map((_, j) => given[i]?.[j] ?? (rand() + rand() + rand() - 1.5) * 1.4));
  const output = x.map((r, i) => r.map((v, j) => Math.sqrt(abar) * v + Math.sqrt(1 - abar) * noise[i]![j]!));
  return done(output, { noise, output });
}

function messagePassing(op: Op, x: M): Result {
  const a = op.extra;
  if (a.length !== x.length) return fail(x);
  const messages = x.map((_, i) => {
    const from = x.filter((_, j) => (a[i]![j] ?? 0) > 0);
    if (!from.length) return Array(cols(x)).fill(0);
    return Array.from({ length: cols(x) }, (_, c) => {
      const vals = from.map((r) => r[c]!);
      if (op.fn === "max") return Math.max(...vals);
      const total = vals.reduce((s, v) => s + v, 0);
      return op.fn === "mean" ? total / vals.length : total;
    });
  });
  const output = op.weights.length ? (matmul(messages, op.weights) ?? messages) : messages;
  return done(output, { messages, output });
}

/** One op on its input (`start` is the step's original input, for a residual with no skip given). */
export function computeOp(op: Op, input: M, start?: M): Result {
  try {
    switch (op.kind) {
      case "convolution":
        return convolution(op, input);
      case "transposed_convolution":
        return transposed(op, input);
      case "pooling":
        return pooling(op, input);
      case "attention":
        return attention(op, input);
      case "linear":
        return linear(op, input);
      case "activation": {
        const f = ACTIVATIONS[op.fn] ?? ACTIVATIONS.relu!;
        const after = input.map((r) => r.map(f));
        return done(after, { before: input, after });
      }
      case "normalization":
        return normalization(op, input);
      case "residual_add":
        return residual(op, input, start);
      case "softmax":
        return softmax(input);
      case "embedding_lookup":
        return embedding(op, input);
      case "positional_encoding":
        return positional(input);
      case "split_heads":
        return splitHeads(op, input);
      case "concat":
        return done(input, { output: input });
      case "recurrent":
        return recurrent(op, input);
      case "dropout":
        return dropout(op, input);
      case "flatten":
        return done([input.flat()], { output: [input.flat()] });
      case "patch_embedding":
        return patchEmbedding(op, input);
      case "gate":
        return gate(op, input);
      case "moe_routing":
        return moe(op, input);
      case "rotary_embedding":
        return rotary(input);
      case "add_noise":
        return addNoise(op, input);
      case "message_passing":
        return messagePassing(op, input);
      case "custom":
        return done(input);
    }
  } catch {
    return fail(input);
  }
}

/** Run a step's ops in order: each gets its own input if it brought one, else the previous output. */
export function runChain(ops: Op[]): { op: Op; input: M; result: Result }[] {
  const start = ops[0]?.input ?? [];
  let current = start;
  return ops.map((op) => {
    const input = op.input.length ? op.input : current;
    const result = computeOp(op, input, start);
    current = result.output;
    return { op, input, result };
  });
}
