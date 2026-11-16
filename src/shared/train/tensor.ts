// A tiny reverse-mode autograd: row-major Float64Array tensors and a tape of backward closures.
// Plain TypeScript, no DOM, so the browser and the server run the same engine.

export class Tensor {
  grad: Float64Array | null = null;
  constructor(
    public data: Float64Array,
    public shape: number[],
    public requiresGrad = false,
  ) {}
  get size(): number {
    return this.data.length;
  }
}

type Nested = number[] | number[][] | number[][][];

const sizeOf = (shape: number[]) => shape.reduce((a, b) => a * b, 1);

/** Build a tensor from nested arrays (1-D, 2-D or 3-D). */
export function tensor(values: Nested, requiresGrad = false): Tensor {
  const shape: number[] = [];
  let probe: unknown = values;
  while (Array.isArray(probe)) {
    shape.push(probe.length);
    probe = probe[0];
  }
  const flat: number[] = [];
  const walk = (v: unknown) => {
    if (Array.isArray(v)) v.forEach(walk);
    else flat.push(v as number);
  };
  walk(values);
  return new Tensor(Float64Array.from(flat), shape, requiresGrad);
}

/** Back to nested arrays (1-D, 2-D or 3-D). */
export function toArray(t: Tensor): Nested {
  const [a = 0, b = 0, c = 0] = t.shape;
  const d = t.data;
  if (t.shape.length === 1) return Array.from(d);
  if (t.shape.length === 2) return Array.from({ length: a }, (_, i) => Array.from(d.subarray(i * b, (i + 1) * b)));
  if (t.shape.length === 3)
    return Array.from({ length: a }, (_, i) =>
      Array.from({ length: b }, (_, j) => Array.from(d.subarray((i * b + j) * c, (i * b + j + 1) * c))),
    );
  throw new Error(`toArray: unsupported rank ${t.shape.length}`);
}

type Entry = { out: Tensor; backward: () => void };

const stack: Tape[] = [];

export class Tape {
  entries: Entry[] = [];

  /** Replay the recorded ops in reverse from `loss`, accumulating into every `.grad`. */
  backward(loss: Tensor): void {
    loss.grad = new Float64Array(loss.size).fill(1);
    for (let i = this.entries.length - 1; i >= 0; i--) {
      const e = this.entries[i]!;
      if (e.out.grad) e.backward();
    }
  }
}

/** Run `fn` with a fresh tape; ops called inside record onto it. */
export function withTape<T>(fn: (tape: Tape) => T): T {
  const tape = new Tape();
  stack.push(tape);
  try {
    return fn(tape);
  } finally {
    stack.pop();
  }
}

function make(data: Float64Array, shape: number[], inputs: Tensor[], backward: (out: Tensor) => void): Tensor {
  const out = new Tensor(data, shape, inputs.some((t) => t.requiresGrad));
  const tape = stack[stack.length - 1];
  if (tape && out.requiresGrad) tape.entries.push({ out, backward: () => backward(out) });
  return out;
}

/** The gradient buffer of `t`, created on first use. */
function gradOf(t: Tensor): Float64Array {
  return (t.grad ??= new Float64Array(t.size));
}

// ---- elementwise with broadcasting of the second operand ----

/** For each element of `a`, the index into `b` it pairs with (b's shape padded left with 1s). */
function broadcastIndex(a: number[], b: number[]): Int32Array {
  if (b.length > a.length) throw new Error(`broadcast: ${b} does not fit ${a}`);
  const bs = [...Array(a.length - b.length).fill(1), ...b] as number[];
  bs.forEach((d, i) => {
    if (d !== 1 && d !== a[i]) throw new Error(`broadcast: [${b}] does not fit [${a}]`);
  });
  const stride = new Array(a.length).fill(0);
  let s = 1;
  for (let i = a.length - 1; i >= 0; i--) {
    stride[i] = bs[i] === 1 ? 0 : s;
    s *= bs[i]!;
  }
  const n = sizeOf(a);
  const idx = new Int32Array(n);
  const pos = new Array(a.length).fill(0);
  for (let k = 0; k < n; k++) {
    let j = 0;
    for (let d = 0; d < a.length; d++) j += pos[d] * stride[d];
    idx[k] = j;
    for (let d = a.length - 1; d >= 0; d--) {
      if (++pos[d] < a[d]!) break;
      pos[d] = 0;
    }
  }
  return idx;
}

function binary(
  a: Tensor,
  b: Tensor,
  f: (x: number, y: number) => number,
  da: (x: number, y: number, g: number) => number,
  db: (x: number, y: number, g: number) => number,
): Tensor {
  const idx = broadcastIndex(a.shape, b.shape);
  const out = new Float64Array(a.size);
  for (let i = 0; i < out.length; i++) out[i] = f(a.data[i]!, b.data[idx[i]!]!);
  return make(out, [...a.shape], [a, b], (o) => {
    const g = o.grad!;
    const ga = a.requiresGrad ? gradOf(a) : null;
    const gb = b.requiresGrad ? gradOf(b) : null;
    for (let i = 0; i < g.length; i++) {
      const x = a.data[i]!;
      const y = b.data[idx[i]!]!;
      if (ga) ga[i]! += da(x, y, g[i]!);
      if (gb) gb[idx[i]!]! += db(x, y, g[i]!);
    }
  });
}

/** a + b; `b` may be a row vector (or any shape that broadcasts onto `a`). */
export const add = (a: Tensor, b: Tensor) =>
  binary(a, b, (x, y) => x + y, (_x, _y, g) => g, (_x, _y, g) => g);
export const sub = (a: Tensor, b: Tensor) =>
  binary(a, b, (x, y) => x - y, (_x, _y, g) => g, (_x, _y, g) => -g);
export const mul = (a: Tensor, b: Tensor) =>
  binary(a, b, (x, y) => x * y, (_x, y, g) => g * y, (x, _y, g) => g * x);
export const div = (a: Tensor, b: Tensor) =>
  binary(a, b, (x, y) => x / y, (_x, y, g) => g / y, (x, y, g) => (-g * x) / (y * y));

function unary(a: Tensor, f: (x: number) => number, df: (x: number, y: number) => number): Tensor {
  const out = a.data.map(f);
  return make(out, [...a.shape], [a], (o) => {
    if (!a.requiresGrad) return;
    const ga = gradOf(a);
    const g = o.grad!;
    for (let i = 0; i < g.length; i++) ga[i]! += g[i]! * df(a.data[i]!, o.data[i]!);
  });
}

export const exp = (a: Tensor) => unary(a, Math.exp, (_x, y) => y);
export const log = (a: Tensor) => unary(a, Math.log, (x) => 1 / x);
export const sqrt = (a: Tensor) => unary(a, Math.sqrt, (_x, y) => 0.5 / y);

// ---- matmul ----

/** 2-D matrix product, or 3-D batched over the first axis. */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const batched = a.shape.length === 3;
  if (a.shape.length !== b.shape.length || (!batched && a.shape.length !== 2))
    throw new Error(`matmul: unsupported shapes [${a.shape}] · [${b.shape}]`);
  const nb = batched ? a.shape[0]! : 1;
  const [m, k] = a.shape.slice(-2) as [number, number];
  const [k2, n] = b.shape.slice(-2) as [number, number];
  if (k !== k2 || (batched && b.shape[0] !== nb)) throw new Error(`matmul: [${a.shape}] · [${b.shape}]`);
  const out = new Float64Array(nb * m * n);
  for (let s = 0; s < nb; s++)
    for (let i = 0; i < m; i++)
      for (let p = 0; p < k; p++) {
        const x = a.data[(s * m + i) * k + p]!;
        for (let j = 0; j < n; j++) out[(s * m + i) * n + j]! += x * b.data[(s * k + p) * n + j]!;
      }
  return make(out, batched ? [nb, m, n] : [m, n], [a, b], (o) => {
    const g = o.grad!;
    const ga = a.requiresGrad ? gradOf(a) : null;
    const gb = b.requiresGrad ? gradOf(b) : null;
    for (let s = 0; s < nb; s++)
      for (let i = 0; i < m; i++)
        for (let j = 0; j < n; j++) {
          const gij = g[(s * m + i) * n + j]!;
          for (let p = 0; p < k; p++) {
            if (ga) ga[(s * m + i) * k + p]! += gij * b.data[(s * k + p) * n + j]!;
            if (gb) gb[(s * k + p) * n + j]! += gij * a.data[(s * m + i) * k + p]!;
          }
        }
  });
}

// ---- reductions ----

function split(shape: number[], axis: number): [number, number, number] {
  if (axis < 0 || axis >= shape.length) throw new Error(`axis ${axis} out of range for [${shape}]`);
  return [sizeOf(shape.slice(0, axis)), shape[axis]!, sizeOf(shape.slice(axis + 1))];
}

function reducedShape(shape: number[], axis: number, keepdim: boolean): number[] {
  const s = keepdim ? shape.map((d, i) => (i === axis ? 1 : d)) : shape.filter((_, i) => i !== axis);
  return s.length ? s : [1];
}

function reduceSum(t: Tensor, axis: number | undefined, keepdim: boolean, scaleByN: boolean): Tensor {
  if (axis === undefined) {
    const n = t.size;
    const total = t.data.reduce((x, y) => x + y, 0);
    const c = scaleByN ? 1 / n : 1;
    return make(Float64Array.of(total * c), [1], [t], (o) => {
      if (!t.requiresGrad) return;
      const gt = gradOf(t);
      for (let i = 0; i < n; i++) gt[i]! += o.grad![0]! * c;
    });
  }
  const [outer, n, inner] = split(t.shape, axis);
  const c = scaleByN ? 1 / n : 1;
  const out = new Float64Array(outer * inner);
  for (let o = 0; o < outer; o++)
    for (let j = 0; j < n; j++) for (let k = 0; k < inner; k++) out[o * inner + k]! += t.data[(o * n + j) * inner + k]! * c;
  return make(out, reducedShape(t.shape, axis, keepdim), [t], (res) => {
    if (!t.requiresGrad) return;
    const gt = gradOf(t);
    for (let o = 0; o < outer; o++)
      for (let j = 0; j < n; j++) for (let k = 0; k < inner; k++) gt[(o * n + j) * inner + k]! += res.grad![o * inner + k]! * c;
  });
}

/** Sum over `axis` (all elements when omitted, giving shape [1]). */
export const sum = (t: Tensor, axis?: number, keepdim = false) => reduceSum(t, axis, keepdim, false);
export const mean = (t: Tensor, axis?: number, keepdim = false) => reduceSum(t, axis, keepdim, true);

/** Max over `axis`; the gradient goes to the (first) argmax. */
export function maxAxis(t: Tensor, axis: number, keepdim = false): Tensor {
  const [outer, n, inner] = split(t.shape, axis);
  const out = new Float64Array(outer * inner);
  const arg = new Int32Array(outer * inner);
  for (let o = 0; o < outer; o++)
    for (let k = 0; k < inner; k++) {
      let best = -Infinity;
      let bi = 0;
      for (let j = 0; j < n; j++) {
        const i = (o * n + j) * inner + k;
        if (t.data[i]! > best) {
          best = t.data[i]!;
          bi = i;
        }
      }
      out[o * inner + k] = best;
      arg[o * inner + k] = bi;
    }
  return make(out, reducedShape(t.shape, axis, keepdim), [t], (res) => {
    if (!t.requiresGrad) return;
    const gt = gradOf(t);
    for (let i = 0; i < arg.length; i++) gt[arg[i]!]! += res.grad![i]!;
  });
}

// ---- shape ops ----

/** Swap the last two axes. */
export function transpose(t: Tensor): Tensor {
  const r = t.shape.length;
  if (r < 2) throw new Error("transpose needs rank >= 2");
  const m = t.shape[r - 2]!;
  const n = t.shape[r - 1]!;
  const batch = t.size / (m * n);
  const out = new Float64Array(t.size);
  for (let s = 0; s < batch; s++)
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[(s * n + j) * m + i] = t.data[(s * m + i) * n + j]!;
  return make(out, [...t.shape.slice(0, -2), n, m], [t], (o) => {
    if (!t.requiresGrad) return;
    const gt = gradOf(t);
    for (let s = 0; s < batch; s++)
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) gt[(s * m + i) * n + j]! += o.grad![(s * n + j) * m + i]!;
  });
}

export function reshape(t: Tensor, shape: number[]): Tensor {
  if (sizeOf(shape) !== t.size) throw new Error(`reshape: [${t.shape}] to [${shape}]`);
  return make(Float64Array.from(t.data), [...shape], [t], (o) => {
    if (!t.requiresGrad) return;
    const gt = gradOf(t);
    for (let i = 0; i < gt.length; i++) gt[i]! += o.grad![i]!;
  });
}

/** Elements `start..end` (end exclusive) along `axis`. */
export function slice(t: Tensor, axis: number, start: number, end: number): Tensor {
  const [outer, n, inner] = split(t.shape, axis);
  if (start < 0 || end > n || start >= end) throw new Error(`slice: ${start}..${end} of ${n}`);
  const w = end - start;
  const out = new Float64Array(outer * w * inner);
  for (let o = 0; o < outer; o++)
    for (let j = 0; j < w; j++)
      for (let k = 0; k < inner; k++) out[(o * w + j) * inner + k] = t.data[(o * n + start + j) * inner + k]!;
  return make(out, t.shape.map((d, i) => (i === axis ? w : d)), [t], (res) => {
    if (!t.requiresGrad) return;
    const gt = gradOf(t);
    for (let o = 0; o < outer; o++)
      for (let j = 0; j < w; j++)
        for (let k = 0; k < inner; k++) gt[(o * n + start + j) * inner + k]! += res.grad![(o * w + j) * inner + k]!;
  });
}

/** Join tensors along `axis`. */
export function concat(ts: Tensor[], axis: number): Tensor {
  const first = ts[0];
  if (!first) throw new Error("concat: nothing to join");
  const outer = sizeOf(first.shape.slice(0, axis));
  const inner = sizeOf(first.shape.slice(axis + 1));
  ts.forEach((t) => {
    if (t.shape.length !== first.shape.length || t.shape.some((d, i) => i !== axis && d !== first.shape[i]))
      throw new Error(`concat: [${t.shape}] does not match [${first.shape}]`);
  });
  const total = ts.reduce((s, t) => s + t.shape[axis]!, 0);
  const out = new Float64Array(outer * total * inner);
  let off = 0;
  for (const t of ts) {
    const w = t.shape[axis]!;
    for (let o = 0; o < outer; o++)
      out.set(t.data.subarray(o * w * inner, (o + 1) * w * inner), (o * total + off) * inner);
    off += w;
  }
  return make(out, first.shape.map((d, i) => (i === axis ? total : d)), ts, (res) => {
    let at = 0;
    for (const t of ts) {
      const w = t.shape[axis]!;
      if (t.requiresGrad) {
        const gt = gradOf(t);
        for (let o = 0; o < outer; o++)
          for (let i = 0; i < w * inner; i++) gt[o * w * inner + i]! += res.grad![(o * total + at) * inner + i]!;
      }
      at += w;
    }
  });
}

/** Rows `ids` of a [vocab, dim] table (an embedding lookup). */
export function gather(table: Tensor, ids: number[]): Tensor {
  const [v, d] = table.shape as [number, number];
  const out = new Float64Array(ids.length * d);
  ids.forEach((id, r) => {
    if (id < 0 || id >= v) throw new Error(`gather: id ${id} outside 0..${v - 1}`);
    out.set(table.data.subarray(id * d, (id + 1) * d), r * d);
  });
  return make(out, [ids.length, d], [table], (o) => {
    if (!table.requiresGrad) return;
    const gt = gradOf(table);
    ids.forEach((id, r) => {
      for (let j = 0; j < d; j++) gt[id * d + j]! += o.grad![r * d + j]!;
    });
  });
}

/** Keep `a` where `mask[i][j]` is true (over the last two axes), else `fill`. */
export function where(mask: boolean[][], a: Tensor, fill: number): Tensor {
  const m = a.shape[a.shape.length - 2]!;
  const n = a.shape[a.shape.length - 1]!;
  if (mask.length !== m || mask.some((row) => row.length !== n)) throw new Error(`where: mask does not fit [${a.shape}]`);
  const keep = (i: number) => mask[Math.floor(i / n) % m]![i % n]!;
  const out = a.data.map((x, i) => (keep(i) ? x : fill));
  return make(out, [...a.shape], [a], (o) => {
    if (!a.requiresGrad) return;
    const ga = gradOf(a);
    for (let i = 0; i < ga.length; i++) if (keep(i)) ga[i]! += o.grad![i]!;
  });
}
