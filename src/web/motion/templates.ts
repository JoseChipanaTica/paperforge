import type { Op } from "../../shared/motion";
import type { M, Result } from "./compute";
import { COLOR, SLICES, bars, box, dot, fit, fmt, frame, grid, label, line, range, rowLabels, shade, W, type Frame, type Item } from "./frames";

/**
 * One animation per op, written as frames from the computed result (compute.ts): the window that
 * slides, the cell being written, the weights that mix. Every number shown comes from `r.stages`,
 * so the drawing can't disagree with the math. A result that didn't compute gets the placeholder.
 */

type Template = (op: Op, input: M, r: Result) => Frame[];

const STEP = 420; // ms per animated beat
const MAX_BEATS = 16; // longer sweeps jump to the end after this many
const cols = (m: M) => m[0]?.length ?? 0;
const first = (m: M | undefined): number[] => m?.[0] ?? [];
const names = (op: Op, n: number, prefix = "") => Array.from({ length: n }, (_, i) => op.tokens[i] ?? `${prefix}${i + 1}`);

/** The captioned placeholder: an op no template covers, or one whose numbers didn't line up. */
export function placeholderFrames(caption: string): Frame[] {
  return [
    frame(
      300,
      box("ph-card", 40, 40, 280, 120, { dash: true, stroke: COLOR.faint }),
      label("ph-text", caption.length > 44 ? `${caption.slice(0, 43)}…` : caption || "No animation for this block", W / 2, 100, { size: 10, color: COLOR.soft }),
    ),
  ];
}

/** Beats for a sweep: the first MAX_BEATS one by one, then the rest at once. */
function sweep<T>(items: T[], beat: (done: T[], current: T | null) => Frame): Frame[] {
  const frames: Frame[] = [];
  const n = Math.min(items.length, MAX_BEATS);
  for (let k = 0; k < n; k++) frames.push(beat(items.slice(0, k + 1), items[k]!));
  if (items.length > n) frames.push(beat(items, null));
  return frames;
}

const convolution: Template = (op, x, r) => {
  const out = r.stages.sums!;
  const pos = r.stages.positions!;
  const k = op.weights;
  const dil = op.fn === "dilated" ? 2 : 1;
  const c = fit(x.length, cols(x), 140, 170);
  const kc = fit(k.length, cols(k), 60, 60, 18);
  const oc = fit(out.length, cols(out), 110, 150);
  const [ix, iy, kx, ky, ox, oy] = [16, 22, 168, 70, 236, 22];
  const xs = range(x);
  const os = range(out);
  const base = [
    label("in-l", "input", ix, 12, { anchor: "start", size: 9, color: COLOR.soft }),
    grid("in", x, ix, iy, c, { scale: xs }),
    label("k-l", op.fn === "depthwise" ? "kernel / channel" : "kernel", kx, ky - 10, { anchor: "start", size: 9, color: COLOR.soft }),
    grid("k", k, kx, ky, kc, { fill: () => COLOR.mark }),
    label("out-l", "output", ox, 12, { anchor: "start", size: 9, color: COLOR.soft }),
  ];
  const cells = pos.map((p, idx) => ({ p, i: Math.floor(idx / cols(out)), j: idx % cols(out) }));
  const frames = [frame(STEP, base, grid("out", out.map((row) => row.map(() => 0)), ox, oy, oc, { fill: () => COLOR.paper, numbers: false }))];
  frames.push(
    ...sweep(cells, (done, cur) => {
      const filled = new Set(done.map((d) => `${d.i}-${d.j}`));
      return frame(
        STEP,
        base,
        grid("out", out, ox, oy, oc, { scale: os, opacity: (i, j) => (filled.has(`${i}-${j}`) ? 1 : 0.12) }),
        cur && box("win", ix + cur.p[1]! * c - 1, iy + cur.p[0]! * c - 1, ((k.length - 1) * dil + 1) * c + 1, ((cols(k) - 1) * dil + 1) * c + 1, { stroke: COLOR.result, width: 2, dash: dil > 1 }),
        cur && box("hit", ox + cur.j * oc - 1, oy + cur.i * oc - 1, oc + 1, oc + 1, { stroke: COLOR.result, width: 2 }),
        cur && label("sum", `Σ window × kernel = ${fmt(out[cur.i]![cur.j]!)}`, 236, 190, { anchor: "start", size: 9 }),
      );
    }),
  );
  if (op.count > 1) {
    const ghosts = [3, 2, 1].map((g) => box(`ghost-${g}`, ox + g * 5, oy - g * 5, cols(out) * oc, out.length * oc, { stroke: COLOR.soft, opacity: 0.6 }));
    frames.push(frame(STEP * 2, base, ghosts, grid("out", out, ox, oy, oc, { scale: os }), label("sum", `× ${op.count} filters, one map each`, 236, 190, { anchor: "start", size: 9, weight: 600 })));
  }
  return frames;
};

const transposed: Template = (op, x, r) => {
  const k = op.weights;
  const s = op.stride;
  const out = r.stages.output!;
  const c = fit(x.length, cols(x), 90, 90);
  const oc = fit(out.length, cols(out), 170, 170, 18);
  const [ix, iy, ox, oy] = [16, 40, 170, 22];
  const os = range(out);
  const partial = out.map((row) => row.map(() => 0));
  const base = [label("in-l", "input", ix, 28, { anchor: "start", size: 9, color: COLOR.soft }), grid("in", x, ix, iy, c), label("out-l", "output (overlaps add)", ox, 12, { anchor: "start", size: 9, color: COLOR.soft })];
  return sweep(r.stages.stamps!, (_done, cur) => {
    if (cur) {
      const [i, j, v] = cur as [number, number, number];
      k.forEach((row, a) => row.forEach((kv, b) => (partial[i * s + a]![j * s + b]! += v * kv)));
    }
    const shown = cur ? partial : out;
    return frame(
      STEP,
      base,
      grid("out", shown, ox, oy, oc, { scale: os, opacity: (a, b) => (shown[a]![b] === 0 ? 0.25 : 1) }),
      cur && box("src", ix + cur[1]! * c - 1, iy + cur[0]! * c - 1, c + 1, c + 1, { stroke: COLOR.result, width: 2 }),
      cur && box("stamp", ox + cur[1]! * s * oc - 1, oy + cur[0]! * s * oc - 1, cols(k) * oc + 1, k.length * oc + 1, { stroke: COLOR.result, width: 2 }),
    );
  });
};

const pooling: Template = (op, x, r) => {
  const picks = r.stages.picks!;
  const c = fit(x.length, cols(x), 150, 170);
  const out = r.output;
  const oc = fit(out.length, cols(out), 100, 120);
  const [ix, iy, ox, oy] = [16, 22, 220, 40];
  const xs = range(x);
  const base = [label("in-l", "input", ix, 12, { anchor: "start", size: 9, color: COLOR.soft }), grid("in", x, ix, iy, c, { scale: xs }), label("out-l", op.fn === "avg" ? "average" : "max", ox, 28, { anchor: "start", size: 9, color: COLOR.soft })];
  return sweep(picks, (done, cur) => {
    const filled = new Set(done.map((p) => `${p[2]}-${p[3]}`));
    return frame(
      STEP,
      base,
      grid("out", out, ox, oy, oc, { scale: xs, opacity: (i, j) => (filled.has(`${i}-${j}`) ? 1 : 0.12) }),
      cur && box("win", ix + (op.fn === "avg" ? cur[1]! : cur[1]! - ((cur[1]! - cur[3]! * op.stride) % op.size)) * c - 1, iy + (op.fn === "avg" ? cur[0]! : cur[0]! - ((cur[0]! - cur[2]! * op.stride) % op.size)) * c - 1, op.size * c + 1, op.size * c + 1, { stroke: COLOR.result, width: 2 }),
      cur && op.fn !== "avg" && dot("pick", ix + cur[1]! * c + c / 2, iy + cur[0]! * c + c / 2, Math.min(5, c / 3)),
    );
  });
};

const attention: Template = (op, _x, r) => {
  const { q, k, v, scores, weights, output } = r.stages as Record<string, M>;
  const n = scores!.length;
  const m = cols(scores!);
  const rowsL = names(op, n, "q");
  const colsL = op.fn === "cross" ? Array.from({ length: m }, (_, j) => op.tokens[n + j] ?? `k${j + 1}`) : names(op, m, "k");
  const c = fit(n, m, 120, 120);
  const [gx, gy] = [70, 36];
  const vc = fit(1, cols(v!), 70, 20, 14);
  const header = [...rowLabels("rl", rowsL, gx, gy, c), ...colsL.map((t, j) => label(`cl-${j}`, t.slice(0, 6), gx + j * c + c / 2, gy - 8, { size: 8 }))];
  const sc = range(scores!);
  const masked = (i: number, j: number) => op.fn === "causal" && j > i;
  const frames: Frame[] = [
    frame(STEP, header, label("t", "Q · Kᵀ: how well each query matches each key", 10, 206, { anchor: "start", size: 9 }), grid("q", q!, 8, 160, fit(q!.length, cols(q!), 50, 40, 10), { numbers: false }), grid("kk", k!, 70, 160, fit(k!.length, cols(k!), 50, 40, 10), { numbers: false })),
  ];
  for (let i = 0; i < n; i++)
    frames.push(frame(STEP, header, label("t", `scores for “${rowsL[i]}”`, 10, 206, { anchor: "start", size: 9 }), grid("s", scores!.map((row, a) => row.map((s) => (a <= i ? s : 0))), gx, gy, c, { scale: sc, opacity: (a) => (a <= i ? 1 : 0.1) })));
  frames.push(frame(STEP * 1.5, header, label("t", `÷ √${cols(q!)} then softmax: each row becomes weights that sum to 1`, 10, 206, { anchor: "start", size: 9 }), grid("s", weights!, gx, gy, c, { scale: [0, 1], fill: (i, j, w) => (masked(i, j) ? COLOR.faint : shade(w, 0, 1)) })));
  for (let i = 0; i < n; i++) {
    frames.push(
      frame(
        STEP * 1.5,
        header,
        label("t", `“${rowsL[i]}” = Σ weight × V`, 10, 206, { anchor: "start", size: 9 }),
        grid("s", weights!, gx, gy, c, { scale: [0, 1], fill: (a, j, w) => (masked(a, j) ? COLOR.faint : shade(w, 0, 1)), opacity: (a) => (a === i ? 1 : 0.35) }),
        label("v-l", "V", 214, 16, { size: 9, color: COLOR.soft }),
        grid("v", v!, 214, 26, vc, { numbers: false, opacity: (a) => 0.25 + 0.75 * weights![i]![a]! }),
        label("o-l", "output", 214, 26 + m * vc + 14, { size: 9, color: COLOR.soft, anchor: "start" }),
        grid("o", output!.slice(0, i + 1), 214, 26 + m * vc + 24, vc, { numbers: false }),
      ),
    );
  }
  return frames;
};

const linear: Template = (op, x, r) => {
  const w = op.weights;
  const out = r.output;
  const xin = first(x);
  const c = fit(w.length, cols(w), 150, 130, 18);
  const [wx, wy] = [110, 50];
  const base = [label("x-l", "x", 100, 20, { anchor: "end", size: 9 }), grid("x", [xin], 110, 12, c), label("w-l", "W", 100, wy + (w.length * c) / 2, { anchor: "end", size: 9 }), grid("w", w, wx, wy, c, { numbers: false })];
  return sweep(
    Array.from({ length: cols(w) }, (_, j) => j),
    (done, j) =>
      frame(
        STEP,
        base,
        j !== null && box("col", wx + j * c - 1, wy - 1, c + 1, w.length * c + 1, { stroke: COLOR.result, width: 2 }),
        label("y-l", "x·W" + (op.extra[0] ? " + b" : ""), 100, wy + w.length * c + 18, { anchor: "end", size: 9 }),
        grid("y", [first(out)], wx, wy + w.length * c + 10, c, { opacity: (_i, jj) => (done.includes(jj) ? 1 : 0.1) }),
      ),
  );
};

const CURVES: Record<string, (v: number) => number> = {
  relu: (v) => Math.max(0, v),
  sigmoid: (v) => 1 / (1 + Math.exp(-v)),
  tanh: Math.tanh,
  silu: (v) => v / (1 + Math.exp(-v)),
  gelu: (v) => 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3))),
};

const activation: Template = (op, x, r) => {
  const before = first(x).slice(0, 8);
  const after = first(r.output).slice(0, 8);
  const f = CURVES[op.fn] ?? CURVES.relu!;
  const span = Math.max(3, ...before.map(Math.abs));
  const [px, py, pw, ph] = [200, 30, 140, 120];
  const lo = Math.min(0, f(-span));
  const hi = Math.max(1, f(span));
  const sx = (v: number) => px + ((v + span) / (2 * span)) * pw;
  const sy = (v: number) => py + ph - ((v - lo) / (hi - lo)) * ph;
  const pts = Array.from({ length: 41 }, (_, i) => -span + (i / 40) * 2 * span).map((v) => `${sx(v).toFixed(1)},${sy(f(v)).toFixed(1)}`).join(" ");
  const c = fit(1, before.length, 170, 30);
  const plot = [
    box("axes", px, py, pw, ph, { stroke: COLOR.faint }),
    line("zero", px, sy(0), px + pw, sy(0), { color: COLOR.faint }),
    { id: "curve", tag: "polyline", attrs: { points: pts, fill: "none", stroke: COLOR.ink, "stroke-width": 1.5 } } as Item,
    label("fn", op.fn || "relu", px + pw / 2, py + ph + 14, { size: 10, weight: 600 }),
  ];
  const s = range([before, after]);
  return [
    frame(STEP, plot, label("b-l", "in", 16, 46, { anchor: "start", size: 9 }), grid("v", [before], 16, 56, c, { scale: s })),
    frame(STEP * 2, plot, label("b-l", "in", 16, 46, { anchor: "start", size: 9 }), grid("v", [before], 16, 56, c, { scale: s }), before.map((v, i) => dot(`p-${i}`, sx(v), sy(f(v)), 3.5))),
    frame(STEP * 2, plot, label("b-l", "out", 16, 46, { anchor: "start", size: 9 }), grid("v", [after], 16, 56, c, { scale: s }), before.map((v, i) => dot(`p-${i}`, sx(v), sy(f(v)), 3.5))),
  ];
};

/** Vertical bars around a zero line, one per value: for normalization and softmax. */
function columns(id: string, values: number[], x: number, mid: number, w: number, h: number, max: number, color: string = COLOR.result): Item[] {
  const bw = w / Math.max(values.length, 1);
  return values.flatMap((v, i) => {
    const len = (Math.abs(v) / Math.max(max, 1e-9)) * h;
    return [box(`${id}-${i}`, x + i * bw + 1, v >= 0 ? mid - len : mid, bw - 3, Math.max(1, len), { fill: color, width: 0.5 }), label(`${id}-v-${i}`, fmt(v), x + i * bw + bw / 2, v >= 0 ? mid - len - 6 : mid + len + 7, { size: 7.5 })];
  });
}

const normalization: Template = (op, x, r) => {
  const row = first(x).slice(0, 8);
  const norm = first(r.output).slice(0, 8);
  const mean = op.fn === "batch" ? (r.stages.mean?.[0]?.[0] ?? 0) : (r.stages.mean?.[0]?.[0] ?? 0);
  const max = Math.max(...row.map(Math.abs), ...norm.map(Math.abs));
  const [bx, mid, bw, bh] = [40, 110, 280, 70];
  const axis = line("axis", bx, mid, bx + bw, mid, { color: COLOR.faint });
  const meanY = mid - (mean / Math.max(max, 1e-9)) * bh;
  const what = op.fn === "rms" ? "divide by the root mean square" : op.fn === "batch" ? "each feature across the batch: subtract its mean, divide by its spread" : "subtract the mean, divide by the spread";
  return [
    frame(STEP, axis, columns("b", row, bx, mid, bw, bh, max, COLOR.mark), label("t", "the values", W / 2, 206, { size: 9 })),
    frame(STEP * 2, axis, columns("b", row, bx, mid, bw, bh, max, COLOR.mark), op.fn !== "rms" && line("mean", bx, meanY, bx + bw, meanY, { color: COLOR.result, width: 1.5 }), op.fn !== "rms" && label("mean-l", `mean ${fmt(mean)}`, bx + bw, meanY - 7, { size: 8, anchor: "end", color: COLOR.result }), label("t", what, W / 2, 206, { size: 9 })),
    frame(STEP * 2, axis, columns("b", norm, bx, mid, bw, bh, max), label("t", op.fn === "rms" ? "same shape, unit size" : "centred on 0, spread 1", W / 2, 206, { size: 9 })),
  ];
};

const residual: Template = (_op, _x, r) => {
  const a = first(r.stages.a).slice(0, 8);
  const b = first(r.stages.b).slice(0, 8);
  const sum = first(r.output).slice(0, 8);
  const c = fit(1, a.length, 220, 26);
  const s = range([a, b, sum]);
  const x0 = 90;
  const rows = (k: number) => [label("a-l", "block out", x0 - 6, 50, { anchor: "end", size: 9 }), grid("a", [a], x0, 40, c, { scale: s }), label("b-l", "skip (input)", x0 - 6, 50 + c + 14, { anchor: "end", size: 9 }), grid("b", [b], x0, 40 + c + 4 + (k ? 10 : 0), c, { scale: s })];
  return [
    frame(STEP, rows(1)),
    frame(STEP * 2, rows(0), label("plus", "+", x0 - 70, 40 + c + 2, { size: 14, weight: 700 }), line("bar", x0, 40 + 2 * c + 12, x0 + a.length * c, 40 + 2 * c + 12), label("s-l", "sum", x0 - 6, 40 + 2 * c + 30, { anchor: "end", size: 9 }), grid("s", [sum], x0, 40 + 2 * c + 20, c, { scale: s })),
  ];
};

const softmax: Template = (op, x, r) => {
  const row = first(x).slice(0, 8);
  const exp = first(r.stages.exp).slice(0, 8);
  const out = first(r.output).slice(0, 8);
  const labels = names(op, row.length, "");
  const [bx, by, bw, bh] = [70, 24, 220, 160];
  return [
    frame(STEP, bars("b", row, bx, by, bw, bh, labels), label("t", "scores", W / 2, 206, { size: 9 })),
    frame(STEP * 2, bars("b", exp, bx, by, bw, bh, labels), label("t", "eˣ: all positive, big ones grow most", W / 2, 206, { size: 9 })),
    frame(STEP * 2, bars("b", out, bx, by, bw, bh, labels, 1), label("t", `÷ Σ: probabilities that sum to 1`, W / 2, 206, { size: 9, weight: 600 })),
  ];
};

const embedding: Template = (op, _x, r) => {
  const ids = first(r.stages.ids);
  const table = op.weights;
  const c = fit(table.length, cols(table), 120, 170, 16);
  const [tx, ty, ox] = [60, 22, 230];
  const base = [label("t-l", "embedding table", tx, 12, { anchor: "start", size: 9, color: COLOR.soft }), grid("tab", table, tx, ty, c, { numbers: false }), ...table.map((_, i) => label(`id-${i}`, String(i), tx - 6, ty + i * c + c / 2, { size: 8, anchor: "end" }))];
  return sweep(
    ids.map((id, t) => ({ id, t })),
    (done, cur) =>
      frame(
        STEP * 1.2,
        base,
        cur && box("row", tx - 1, ty + cur.id * c - 1, cols(table) * c + 1, c + 1, { stroke: COLOR.result, width: 2 }),
        done.flatMap(({ id, t }) => [label(`tok-${t}`, `${names(op, ids.length)[t]} → ${id}`, ox - 6, ty + t * (c + 6) + c / 2, { anchor: "end", size: 8.5 }), ...grid(`o-${t}`, [table[id]!], ox, ty + t * (c + 6), c, { numbers: false })]),
      ),
  );
};

const positional: Template = (op, x, r) => {
  const pe = r.stages.pe!;
  const out = r.output;
  const n = Math.min(x.length, 6);
  const c = fit(n, cols(x), 100, 110, 16);
  const s = range([...x, ...out]);
  const labels = names(op, n, "pos ");
  const waves = [0, 1].map((dim) => {
    const pts = Array.from({ length: 31 }, (_, i) => {
      const p = (i / 30) * Math.max(n - 1, 1);
      const v = dim % 2 === 0 ? Math.sin(p) : Math.cos(p);
      return `${(20 + (i / 30) * 320).toFixed(1)},${(176 - v * 14).toFixed(1)}`;
    });
    return { id: `wave-${dim}`, tag: "polyline", attrs: { points: pts.join(" "), fill: "none", stroke: dim ? COLOR.soft : COLOR.result, "stroke-width": 1.2 } } as Item;
  });
  return [
    frame(STEP, rowLabels("rl", labels, 60, 20, c), grid("x", x.slice(0, n), 60, 20, c, { scale: s, numbers: false })),
    frame(STEP * 2, rowLabels("rl", labels, 60, 20, c), grid("x", x.slice(0, n), 60, 20, c, { scale: s, numbers: false }), label("plus", "+", 175, 20 + (n * c) / 2, { size: 14 }), grid("pe", pe.slice(0, n), 190, 20, c, { scale: [-1, 1], numbers: false }), waves, label("t", "sin / cos at each position", W / 2, 206, { size: 9 })),
    frame(STEP * 2, rowLabels("rl", labels, 60, 20, c), grid("x", out.slice(0, n), 60, 20, c, { scale: s, numbers: false }), waves, label("t", "each word now carries where it sits", W / 2, 206, { size: 9 })),
  ];
};

/** One vector cut into `count` slices (split) or put back together (concat). */
function slices(op: Op, row: number[], apart: number): Item[] {
  const h = Math.max(1, op.count);
  const per = Math.max(1, Math.ceil(row.length / h));
  const c = fit(1, row.length + h, 300, 26);
  return row.map((v, j) => {
    const k = Math.floor(j / per);
    return { id: `c-${j}`, tag: "rect", attrs: { x: 30 + j * c + k * apart, y: 90, width: c - 1, height: c - 1, fill: SLICES[k % SLICES.length]!, stroke: COLOR.ink, "stroke-width": 0.5, opacity: 0.35 + 0.65 * Math.min(1, Math.abs(v) / Math.max(...row.map(Math.abs), 1e-9)) } } as Item;
  });
}

const splitHeads: Template = (op, x) => {
  const row = first(x).slice(0, 8);
  const n = Math.max(1, op.count);
  return [
    frame(STEP, slices({ ...op, count: n }, row, 0), label("t", "one vector per word", W / 2, 150, { size: 9 })),
    frame(STEP * 2, slices({ ...op, count: n }, row, 10), label("t", `cut into ${n} heads, each works on its own slice`, W / 2, 150, { size: 9 })),
  ];
};

const concat: Template = (op, x) => {
  const row = first(x).slice(0, 8);
  const n = Math.max(1, op.count);
  return [frame(STEP, slices({ ...op, count: n }, row, 10), label("t", "the heads' results", W / 2, 150, { size: 9 })), frame(STEP * 2, slices({ ...op, count: n }, row, 0), label("t", "joined back into one vector", W / 2, 150, { size: 9 }))];
};

const recurrent: Template = (op, x, r) => {
  const states = r.stages.states!;
  const n = Math.min(states.length, 6);
  const step = 300 / n;
  const c = fit(1, cols(states), step - 14, 18, 12);
  const toks = names(op, n);
  return sweep(
    Array.from({ length: n }, (_, t) => t),
    (done) =>
      frame(
        STEP * 1.3,
        toks.map((t, i) => label(`tok-${i}`, t.slice(0, 7), 30 + i * step + step / 2, 160, { size: 9 })),
        done.flatMap((t) => [
          ...grid(`h-${t}`, [states[t]!], 30 + t * step + 4, 80, c, { scale: [-1, 1], numbers: false }),
          line(`up-${t}`, 30 + t * step + step / 2, 150, 30 + t * step + step / 2, 82 + c + 4, { color: COLOR.soft }),
          ...(t > 0 ? [line(`carry-${t}`, 30 + t * step - 6, 80 + c / 2, 30 + t * step + 4, 80 + c / 2, { color: COLOR.result, width: 1.5 })] : []),
        ]),
        label("t", "the state carries from word to word", W / 2, 206, { size: 9 }),
      ),
  );
};

const dropout: Template = (op, x, r) => {
  const row = first(x).slice(0, 8);
  const mask = first(r.stages.mask).slice(0, 8);
  const out = first(r.output).slice(0, 8);
  const c = fit(1, row.length, 280, 30);
  const s = range([row, out]);
  const p = op.extra[0]?.[0] ?? 0.5;
  return [
    frame(STEP, grid("v", [row], 40, 80, c, { scale: s })),
    frame(STEP * 2, grid("v", [row], 40, 80, c, { scale: s, opacity: (_i, j) => (mask[j] ? 1 : 0.12) }), label("t", `drop each value with chance ${fmt(p)} (training only)`, W / 2, 150, { size: 9 })),
    frame(STEP * 2, grid("v", [out], 40, 80, c, { scale: s, opacity: (_i, j) => (mask[j] ? 1 : 0.12) }), label("t", `survivors × ${fmt(1 / (1 - p))} keep the total the same`, W / 2, 150, { size: 9 })),
  ];
};

const flatten: Template = (_op, x) => {
  const c = Math.min(fit(x.length, cols(x), 120, 120), 320 / Math.max(1, x.flat().length));
  const big = fit(x.length, cols(x), 120, 120);
  const cells = (flat: boolean): Item[] =>
    x.flatMap((row, i) => row.map((v, j) => ({ id: `c-${i}-${j}`, tag: "rect", attrs: { x: flat ? 20 + (i * cols(x) + j) * c : 120 + j * big, y: flat ? 160 : 20 + i * big, width: (flat ? c : big) - 1, height: (flat ? c : big) - 1, fill: shade(v, ...range(x)), stroke: COLOR.ink, "stroke-width": 0.5 } }) as Item));
  return [frame(STEP, cells(false)), frame(STEP * 3, cells(true), label("t", "row after row, into one long vector", W / 2, 206, { size: 9 }))];
};

const patchEmbedding: Template = (op, x, r) => {
  const p = Math.max(1, op.size);
  const patches = r.stages.patches!;
  const out = r.output;
  const c = fit(x.length, cols(x), 140, 160);
  const per = Math.floor(cols(x) / p);
  const oc = fit(out.length, cols(out), 120, 160, 14);
  const base = [label("in-l", "image", 16, 12, { anchor: "start", size: 9, color: COLOR.soft }), grid("in", x, 16, 22, c, { numbers: false })];
  return sweep(
    patches.map((_, k) => k),
    (done, k) =>
      frame(
        STEP * 1.2,
        base,
        k !== null && box("patch", 16 + (k % per) * p * c - 1, 22 + Math.floor(k / per) * p * c - 1, p * c + 1, p * c + 1, { stroke: COLOR.result, width: 2 }),
        label("o-l", "one vector per patch", 220, 12, { anchor: "start", size: 9, color: COLOR.soft }),
        done.flatMap((d) => [label(`pn-${d}`, `#${d + 1}`, 214, 22 + d * oc + oc / 2, { anchor: "end", size: 8 }), ...grid(`o-${d}`, [out[d]!], 220, 22 + d * oc, oc, { numbers: false })]),
      ),
  );
};

const gate: Template = (op, x, r) => {
  const signal = first(x).slice(0, 8);
  const g = first(r.stages.gate).slice(0, 8);
  const out = first(r.output).slice(0, 8);
  const c = fit(1, signal.length, 220, 26);
  const s = range([signal, out]);
  const x0 = 110;
  const word = op.fn === "lstm" ? "forget gate" : op.fn === "gru" ? "update gate" : "gate";
  return [
    frame(STEP, label("s-l", "signal", x0 - 6, 40 + c / 2, { anchor: "end", size: 9 }), grid("s", [signal], x0, 40, c, { scale: s })),
    frame(STEP * 2, label("s-l", "signal", x0 - 6, 40 + c / 2, { anchor: "end", size: 9 }), grid("s", [signal], x0, 40, c, { scale: s }), label("g-l", word, x0 - 6, 90 + c / 2, { anchor: "end", size: 9 }), grid("g", [g], x0, 90, c, { scale: [0, 1] })),
    frame(STEP * 2, label("s-l", "signal", x0 - 6, 40 + c / 2, { anchor: "end", size: 9 }), grid("s", [signal], x0, 40, c, { scale: s }), label("g-l", word, x0 - 6, 90 + c / 2, { anchor: "end", size: 9 }), grid("g", [g], x0, 90, c, { scale: [0, 1] }), label("o-l", op.fn === "lstm" || op.fn === "gru" ? "new state" : "passed", x0 - 6, 140 + c / 2, { anchor: "end", size: 9 }), grid("o", [out], x0, 140, c, { scale: s }), label("t", "each value goes through as far as its gate is open (0–1)", W / 2, 206, { size: 9 })),
  ];
};

const moe: Template = (op, _x, r) => {
  const scores = r.stages.scores!;
  const chosen = r.stages.chosen!;
  const n = Math.min(scores.length, 5);
  const e = cols(scores);
  const toks = names(op, n);
  const ey = (j: number) => 24 + j * (170 / Math.max(e, 1));
  const ty = (i: number) => 24 + i * (170 / Math.max(n, 1));
  const experts = Array.from({ length: e }, (_, j) => [box(`e-${j}`, 250, ey(j), 70, 16, { fill: COLOR.paper }), label(`el-${j}`, `expert ${j + 1}`, 285, ey(j) + 8, { size: 8.5 })]).flat();
  const tokens = toks.map((t, i) => label(`t-${i}`, t.slice(0, 8), 60, ty(i) + 8, { size: 9, anchor: "end" }));
  return sweep(
    Array.from({ length: n }, (_, i) => i),
    (_done, i) =>
      frame(
        STEP * 1.5,
        experts,
        tokens,
        i !== null && chosen[i]!.map((j) => line(`route-${j}`, 66, ty(i) + 8, 250, ey(j) + 8, { color: COLOR.result, width: 1.5 + 3 * scores[i]![j]! })),
        i !== null && scores[i]!.map((w, j) => box(`sc-${j}`, 326, ey(j) + 3, w * 30, 10, { fill: chosen[i]!.includes(j) ? COLOR.result : COLOR.faint, width: 0.5 })),
        label("t", `the router sends each token to its top ${Math.min(op.count, e)} expert${op.count > 1 ? "s" : ""}`, W / 2, 210, { size: 9 }),
      ),
  );
};

const rotary: Template = (op, x, r) => {
  const n = Math.min(x.length, 5);
  const angles = r.stages.angles!;
  const step = 300 / n;
  const hand = (id: string, a: number, b: number, cx: number, color: string): Item => {
    const len = Math.hypot(a, b) || 1;
    return line(id, cx, 100, cx + (a / len) * 26, 100 - (b / len) * 26, { color, width: 2 });
  };
  const dials = Array.from({ length: n }, (_, t) => [{ id: `dial-${t}`, tag: "circle", attrs: { cx: 30 + t * step + step / 2, cy: 100, r: 28, fill: COLOR.paper, stroke: COLOR.faint, "stroke-width": 1 } } as Item, label(`pos-${t}`, op.tokens[t] ?? `pos ${t}`, 30 + t * step + step / 2, 145, { size: 9 })]).flat();
  const hands = (m: M, color: string) => Array.from({ length: n }, (_, t) => hand(`h-${t}`, m[t]![0]!, m[t]![1]!, 30 + t * step + step / 2, color));
  return [
    frame(STEP, dials, hands(x, COLOR.soft), label("t", "the first pair of numbers in each query, as a hand", W / 2, 190, { size: 9 })),
    frame(STEP * 3, dials, hands(r.output, COLOR.result), Array.from({ length: n }, (_, t) => label(`a-${t}`, `${fmt((angles[t]![0]! * 180) / Math.PI)}°`, 30 + t * step + step / 2, 160, { size: 8, color: COLOR.soft })), label("t", "turned by an angle that grows with position", W / 2, 190, { size: 9 })),
  ];
};

const addNoise: Template = (op, x, r) => {
  const noise = r.stages.noise!;
  const out = r.output;
  const c = fit(x.length, cols(x), 100, 120);
  const s = range([...x, ...noise, ...out]);
  const abar = op.extra[0]?.[0] ?? 0.5;
  const lbl = (id: string, t: string, xx: number) => label(id, t, xx, 14, { size: 9, color: COLOR.soft });
  return [
    frame(STEP, lbl("l1", "image", 70), grid("img", x, 20, 24, c, { scale: s, numbers: false })),
    frame(STEP * 1.5, lbl("l1", "image", 70), grid("img", x, 20, 24, c, { scale: s, numbers: false }), lbl("l2", "noise ε", 180), grid("n", noise, 130, 24, c, { scale: s, numbers: false })),
    frame(STEP * 2, lbl("l1", "image", 70), grid("img", x, 20, 24, c, { scale: s, numbers: false }), lbl("l2", "noise ε", 180), grid("n", noise, 130, 24, c, { scale: s, numbers: false }), lbl("l3", `x_t (ᾱ = ${fmt(abar)})`, 290), grid("o", out, 240, 24, c, { scale: s, numbers: false }), label("t", `√ᾱ · image + √(1−ᾱ) · noise`, W / 2, 200, { size: 9 })),
  ];
};

const messagePassing: Template = (op, x, r) => {
  const a = op.extra;
  const n = Math.min(x.length, 8);
  const at = (i: number) => [180 + 70 * Math.cos((2 * Math.PI * i) / n - Math.PI / 2), 100 + 70 * Math.sin((2 * Math.PI * i) / n - Math.PI / 2)] as const;
  const edges: Item[] = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if ((a[i]?.[j] ?? 0) > 0 || (a[j]?.[i] ?? 0) > 0) edges.push(line(`e-${i}-${j}`, ...at(i), ...at(j), { color: COLOR.faint }));
  const nodes = (vals: number[][], lit: number | null) =>
    Array.from({ length: n }, (_, i) => [dot(`n-${i}`, ...at(i), 13, i === lit ? COLOR.result : COLOR.mark), label(`nv-${i}`, fmt(vals[i]?.[0] ?? 0), ...at(i), { size: 8 }), label(`nl-${i}`, (op.tokens[i] ?? `n${i + 1}`).slice(0, 6), at(i)[0], at(i)[1] + 22, { size: 8, color: COLOR.soft })]).flat();
  return [
    frame(STEP, edges, nodes(x, null)),
    ...sweep(
      Array.from({ length: n }, (_, i) => i),
      (done, i) =>
        frame(
          STEP * 1.2,
          edges,
          i !== null && Array.from({ length: n }, (_, j) => j).filter((j) => (a[i]?.[j] ?? 0) > 0).map((j) => line(`m-${j}`, ...at(j), ...at(i), { color: COLOR.result, width: 2 })),
          nodes(x.map((row, k) => (done.includes(k) ? r.output[k]! : row)), i),
          label("t", `each node gathers its neighbours (${op.fn || "sum"})`, W / 2, 208, { size: 9 }),
        ),
    ),
  ];
};

const TEMPLATES: Partial<Record<Op["kind"], Template>> = {
  convolution,
  transposed_convolution: transposed,
  pooling,
  attention,
  linear,
  activation,
  normalization,
  residual_add: residual,
  softmax,
  embedding_lookup: embedding,
  positional_encoding: positional,
  split_heads: splitHeads,
  concat,
  recurrent,
  dropout,
  flatten,
  patch_embedding: patchEmbedding,
  gate,
  moe_routing: moe,
  rotary_embedding: rotary,
  add_noise: addNoise,
  message_passing: messagePassing,
};

/** The frames for one op: its template, or the captioned placeholder when it can't be drawn. */
export function framesFor(op: Op, input: M, result: Result): Frame[] {
  const template = TEMPLATES[op.kind];
  if (!template || !result.ok) return placeholderFrames(op.caption);
  try {
    const frames = template(op, input, result);
    return frames.length ? frames : placeholderFrames(op.caption);
  } catch {
    return placeholderFrames(op.caption);
  }
}
