// The toy model's operations: each contract kind composed from the tape's primitives (convolution and pooling
// carry their own backward). Rows are sequence positions (or batch items), columns are features.

import {
  Tensor, add, concat, div, exp, gather, gradOf, make, matmul, maxAxis, mean, mul, reshape, slice, sqrt, sub, sum,
  transpose, where,
} from "./tensor";

/** Every op the engine can run; anything else becomes a pass-through upstream. */
export const TOY_KINDS = [
  "input", "embedding_lookup", "positional_encoding", "linear", "activation", "normalization", "residual_add",
  "attention", "softmax", "dropout", "convolution", "pooling", "flatten", "concat",
] as const;
export type ToyKind = (typeof TOY_KINDS)[number];

const EPS = 1e-5;
/** Finite stand-in for −∞ in a causal mask, so masked scores still have a finite gradient (zero). */
const MASKED = -1e9;

const unusable = (reason: string) => new Error(`unusable: ${reason}`);
const scalar = (v: number) => new Tensor(Float64Array.of(v), [1]);
const constant = (data: Float64Array, shape: number[]) => new Tensor(data, [...shape]);
/** A copy recorded on the tape with an identity backward, so every node owns its output. */
const identity = (x: Tensor) => reshape(x, x.shape);

function rows(x: Tensor, what: string): [number, number] {
  if (x.shape.length !== 2) throw unusable(`${what} must be a matrix, got [${x.shape}]`);
  return x.shape as [number, number];
}

function need(params: Record<string, Tensor>, name: string, kind: string): Tensor {
  const p = params[name];
  if (!p) throw unusable(`${kind} needs a parameter "${name}"`);
  return p;
}

function one(inputs: Tensor[], kind: string, n = 1): Tensor[] {
  if (inputs.length < n) throw unusable(`${kind} needs ${n} input${n > 1 ? "s" : ""}, got ${inputs.length}`);
  return inputs;
}

/** Flatten to one row, apply `f` (whose masks are over [1, size]), restore the shape. */
function elementwise(x: Tensor, f: (flat: Tensor, mask: (p: (v: number) => boolean) => boolean[][]) => Tensor) {
  const flat = reshape(x, [1, x.size]);
  const mask = (p: (v: number) => boolean) => [Array.from(flat.data, p)];
  return reshape(f(flat, mask), x.shape);
}

/** σ(x) without overflow: e = exp(−|x|) ≤ 1, σ = (x ≥ 0 ? 1 : e) / (1 + e). */
function sigmoid(x: Tensor): Tensor {
  return elementwise(x, (flat, mask) => {
    const negAbs = constant(flat.data.map((v) => (v >= 0 ? -1 : 1)), flat.shape);
    const e = exp(mul(flat, negAbs));
    return div(where(mask((v) => v < 0), e, 1), add(e, scalar(1)));
  });
}

/** tanh(x) = 2σ(2x) − 1. */
const tanh = (x: Tensor) => sub(mul(sigmoid(mul(x, scalar(2))), scalar(2)), scalar(1));

function activation(x: Tensor, fn: string): Tensor {
  switch (fn) {
    case "relu":
    case "":
      return elementwise(x, (flat, mask) => where(mask((v) => v > 0), flat, 0));
    case "gelu": {
      // 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
      const inner = mul(add(x, mul(mul(mul(x, x), x), scalar(0.044715))), scalar(Math.sqrt(2 / Math.PI)));
      return mul(mul(x, add(tanh(inner), scalar(1))), scalar(0.5));
    }
    case "sigmoid":
      return sigmoid(x);
    case "tanh":
      return tanh(x);
    case "silu":
    case "swish":
      return mul(x, sigmoid(x));
    default:
      throw unusable(`unknown activation "${fn}"`);
  }
}

/** Softmax over the last axis, max-subtracted (the max is a constant: softmax doesn't depend on it). */
function softmax(x: Tensor): Tensor {
  const last = x.shape.length - 1;
  const m = maxAxis(x, last, true);
  const e = exp(sub(x, constant(Float64Array.from(m.data), m.shape)));
  return div(e, sum(e, last, true));
}

function normalization(x: Tensor, fn: string, params: Record<string, Tensor>): Tensor {
  rows(x, "normalization input");
  const gamma = need(params, "gamma", "normalization");
  if (fn === "rms") {
    const ms = mean(mul(x, x), 1, true);
    return mul(div(x, sqrt(add(ms, scalar(EPS)))), gamma);
  }
  if (fn !== "layer" && fn !== "batch" && fn !== "") throw unusable(`unknown normalization "${fn}"`);
  const axis = fn === "batch" ? 0 : 1;
  const xc = sub(x, mean(x, axis, true));
  const variance = mean(mul(xc, xc), axis, true);
  const y = div(xc, sqrt(add(variance, scalar(EPS))));
  return add(mul(y, gamma), need(params, "beta", "normalization"));
}

function attention(fn: string, inputs: Tensor[], params: Record<string, Tensor>, count: number): Tensor {
  const cross = fn === "cross";
  const causal = fn === "causal" || fn === "masked";
  if (!cross && !causal && fn !== "" && fn !== "self") throw unusable(`unknown attention "${fn}"`);
  const [xq] = one(inputs, "attention", cross ? 2 : 1);
  const xkv = cross ? inputs[1]! : xq!;
  const s = rows(xq!, "attention queries")[0];
  const t = rows(xkv, "attention keys")[0];
  const q = matmul(xq!, need(params, "Wq", "attention"));
  const k = matmul(xkv, need(params, "Wk", "attention"));
  const v = matmul(xkv, need(params, "Wv", "attention"));
  const d = q.shape[1]!;
  if (!Number.isInteger(count) || count < 1 || d % count !== 0)
    throw unusable(`width ${d} does not split into ${count} heads`);
  if (k.shape[1] !== d) throw unusable(`Wk gives width ${k.shape[1]}, Wq ${d}`);
  if (v.shape[1]! % count !== 0) throw unusable(`Wv width ${v.shape[1]} does not split into ${count} heads`);
  const dk = d / count;
  const dv = v.shape[1]! / count;
  const mask = Array.from({ length: s }, (_, i) => Array.from({ length: t }, (_, j) => j <= i));
  const heads: Tensor[] = [];
  for (let h = 0; h < count; h++) {
    const part = (m: Tensor, w: number) => slice(m, 1, h * w, (h + 1) * w);
    let scores = mul(matmul(part(q, dk), transpose(part(k, dk))), scalar(1 / Math.sqrt(dk)));
    if (causal) scores = where(mask, scores, MASKED);
    heads.push(matmul(softmax(scores), part(v, dv)));
  }
  return matmul(concat(heads, 1), need(params, "Wo", "attention"));
}

/** Sinusoidal table: PE(pos, 2i) = sin(pos / 10000^(2i/d)), PE(pos, 2i+1) = cos(…). */
export function sinusoidal(seq: number, d: number): Tensor {
  const out = new Float64Array(seq * d);
  for (let pos = 0; pos < seq; pos++)
    for (let j = 0; j < d; j++) {
      const angle = pos / 10000 ** ((2 * Math.floor(j / 2)) / d);
      out[pos * d + j] = j % 2 === 0 ? Math.sin(angle) : Math.cos(angle);
    }
  return constant(out, [seq, d]);
}

/** Valid 2-D cross-correlation of an [H, W] map with a [kh, kw] kernel, stride 1. */
function convolution(x: Tensor, w: Tensor): Tensor {
  const [h, wd] = rows(x, "convolution input");
  const [kh, kw] = rows(w, "convolution kernel");
  if (kh > h || kw > wd) throw unusable(`kernel [${w.shape}] is larger than the input [${x.shape}]`);
  const oh = h - kh + 1;
  const ow = wd - kw + 1;
  const out = new Float64Array(oh * ow);
  for (let i = 0; i < oh; i++)
    for (let j = 0; j < ow; j++) {
      let acc = 0;
      for (let a = 0; a < kh; a++) for (let b = 0; b < kw; b++) acc += x.data[(i + a) * wd + j + b]! * w.data[a * kw + b]!;
      out[i * ow + j] = acc;
    }
  return make(out, [oh, ow], [x, w], (o) => {
    const g = o.grad!;
    const gx = x.requiresGrad ? gradOf(x) : null;
    const gw = w.requiresGrad ? gradOf(w) : null;
    for (let i = 0; i < oh; i++)
      for (let j = 0; j < ow; j++) {
        const gij = g[i * ow + j]!;
        for (let a = 0; a < kh; a++)
          for (let b = 0; b < kw; b++) {
            if (gx) gx[(i + a) * wd + j + b]! += gij * w.data[a * kw + b]!;
            if (gw) gw[a * kw + b]! += gij * x.data[(i + a) * wd + j + b]!;
          }
      }
  });
}

/** Max or average over non-overlapping 2×2 windows (1 along an axis of length 1); a leftover edge is dropped. */
function pooling(x: Tensor, fn: string): Tensor {
  if (fn !== "max" && fn !== "avg" && fn !== "average" && fn !== "") throw unusable(`unknown pooling "${fn}"`);
  const isMax = fn === "max" || fn === "";
  const [h, w] = rows(x, "pooling input");
  const ph = Math.min(2, h);
  const pw = Math.min(2, w);
  const oh = Math.floor(h / ph);
  const ow = Math.floor(w / pw);
  const out = new Float64Array(oh * ow);
  const arg = new Int32Array(oh * ow);
  const cells = (i: number, j: number) => {
    const at: number[] = [];
    for (let a = 0; a < ph; a++) for (let b = 0; b < pw; b++) at.push((i * ph + a) * w + j * pw + b);
    return at;
  };
  for (let i = 0; i < oh; i++)
    for (let j = 0; j < ow; j++) {
      const at = cells(i, j);
      if (isMax) {
        let best = at[0]!;
        for (const k of at) if (x.data[k]! > x.data[best]!) best = k;
        arg[i * ow + j] = best;
        out[i * ow + j] = x.data[best]!;
      } else out[i * ow + j] = at.reduce((s, k) => s + x.data[k]!, 0) / at.length;
    }
  return make(out, [oh, ow], [x], (o) => {
    if (!x.requiresGrad) return;
    const gx = gradOf(x);
    for (let i = 0; i < oh; i++)
      for (let j = 0; j < ow; j++) {
        const g = o.grad![i * ow + j]!;
        if (isMax) gx[arg[i * ow + j]!]! += g;
        else for (const k of cells(i, j)) gx[k]! += g / (ph * pw);
      }
  });
}

function sameShape(ts: Tensor[], kind: string) {
  const s = ts[0]!.shape.join("×");
  for (const t of ts) if (t.shape.join("×") !== s) throw unusable(`${kind}: [${t.shape}] does not match [${ts[0]!.shape}]`);
}

function run(kind: ToyKind, fn: string, inputs: Tensor[], params: Record<string, Tensor>, count: number): Tensor {
  switch (kind) {
    case "input": {
      const v = need(params, "values", kind);
      return new Tensor(Float64Array.from(v.data), [...v.shape]);
    }
    case "embedding_lookup": {
      const [x] = one(inputs, kind);
      const table = need(params, "E", kind);
      rows(table, "embedding table");
      const width = x!.shape[x!.shape.length - 1]!;
      return gather(table, Array.from(x!.data.subarray(0, width), (v) => Math.round(v)));
    }
    case "positional_encoding": {
      const [x] = one(inputs, kind);
      const [seq, d] = rows(x!, "positional encoding input");
      if (fn !== "learned") return add(x!, sinusoidal(seq, d));
      const p = need(params, "P", kind);
      if (rows(p, "P")[0] < seq) throw unusable(`P has ${p.shape[0]} positions, the input ${seq}`);
      return add(x!, p.shape[0] === seq ? p : slice(p, 0, 0, seq));
    }
    case "linear": {
      const [x] = one(inputs, kind);
      rows(x!, "linear input");
      const y = matmul(x!, need(params, "W", kind));
      return params.b ? add(y, params.b) : y;
    }
    case "activation":
      return activation(one(inputs, kind)[0]!, fn);
    case "normalization":
      return normalization(one(inputs, kind)[0]!, fn, params);
    case "residual_add": {
      one(inputs, kind);
      sameShape(inputs, kind);
      return inputs.length === 1 ? identity(inputs[0]!) : inputs.slice(1).reduce((s, t) => add(s, t), inputs[0]!);
    }
    case "attention":
      return attention(fn, inputs, params, count);
    case "softmax":
      return softmax(one(inputs, kind)[0]!);
    case "dropout":
      return identity(one(inputs, kind)[0]!);
    case "convolution": {
      const y = convolution(one(inputs, kind)[0]!, need(params, "W", kind));
      return params.b ? add(y, params.b) : y;
    }
    case "pooling":
      return pooling(one(inputs, kind)[0]!, fn);
    case "flatten": {
      const [x] = one(inputs, kind);
      return reshape(x!, [1, x!.size]);
    }
    case "concat": {
      one(inputs, kind);
      return inputs.length === 1 ? identity(inputs[0]!) : concat(inputs, inputs[0]!.shape.length - 1);
    }
  }
}

/**
 * Run one toy op on the current tape. `fn` picks the variant (activation, normalization, attention, pooling,
 * positional encoding), `count` is the number of attention heads. Any shape problem throws `Error("unusable: …")`.
 */
export function runOp(
  kind: ToyKind,
  fn: string,
  inputs: Tensor[],
  params: Record<string, Tensor>,
  count: number,
): Tensor {
  try {
    return run(kind, fn.trim().toLowerCase(), inputs, params, count);
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    throw msg.startsWith("unusable:") ? e : unusable(`${kind}: ${msg}`);
  }
}
