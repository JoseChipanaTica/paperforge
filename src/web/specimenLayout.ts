import type { View } from "../shared/learn";

/**
 * Where each piece of a specimen goes: a view (tokens, vectors, a heatmap…) as flat cells in
 * world units, centred on (0, 0). Pure, so specimen.ts only turns cells into meshes. A cell's
 * key is stable by row label, so "cat" keeps its place from one step to the next.
 */

export interface Cell {
  key: string;
  x: number;
  y: number;
  w: number;
  h: number;
  d: number;
  shade: number; // 0..1 along the highlighter → vermilion ramp
  label?: string; // drawn beside the cell (row labels, tokens, top class)
  tip: string; // shown on hover
  slice?: number; // heads: which head this cell belongs to
}

export interface Laid {
  cells: Cell[];
  /** Room its row labels need to the left of its cells (layoutViews' `rooms`), 0 when not measured. */
  labelRoom?: number;
  width: number;
  height: number;
  caption: string;
  more?: string; // "…512": the vector is wider than the slice shown
}

export const MAX_CELLS = 300;
const S = 0.22; // cell side
const GAP = 0.04;
const STEP = S + GAP;
const DEPTH = 0.08;
const VIEW_GAP = 0.3;
/** Cell to cell, in layout units: what a column label has to fit in. */
export const CELL_PITCH = STEP;
/** A row label sits this far left of its cell (specimen.ts draws it there). */
export const ROW_LABEL_GAP = 0.06;

// The labels are DOM text in JetBrains Mono (styles/specimen.css): every glyph is 0.6 em wide.
const MONO_EM = 0.6;
export const ROW_LABEL_PX = 11;
export const COL_LABEL_PX = 9.5;
const COL_LINE_PX = 12; // a column label's line, when it stands on end
/** How wide a label prints, in screen pixels. */
export const labelPx = (text: string, fontPx: number) => [...text].length * MONO_EM * fontPx;

/**
 * Column labels over a matrix: side by side while the longest fits its column, standing on end when it doesn't but a
 * line does, else none (the cell's tooltip still names its column). `pitchPx`: one column's width on screen now.
 */
export function colLabelMode(cols: string[], pitchPx: number): "flat" | "rotated" | "none" {
  if (!cols.length) return "none";
  const widest = Math.max(...cols.map((c) => labelPx(c, COL_LABEL_PX)));
  if (widest + 4 <= pitchPx) return "flat";
  return COL_LINE_PX <= pitchPx ? "rotated" : "none";
}

/** The labels a view prints to the left of its rows (tokens and patches print theirs on the cells). */
export function rowLabelsOf(v: View): string[] {
  if (v.kind === "tokens" || v.kind === "patches") return [];
  return lay(v).cells.flatMap((c) => (c.label ? [c.label] : []));
}

/**
 * Layout units left of a view's cells that its row labels need, so the card is measured with them and none is
 * clipped. `pxPerUnit`: how many screen pixels one layout unit covers at the card's size.
 */
export function rowLabelRoom(labels: string[], pxPerUnit: number): number {
  if (!labels.length || !(pxPerUnit > 0)) return 0;
  return ROW_LABEL_GAP + (Math.max(...labels.map((l) => labelPx(l, ROW_LABEL_PX))) + 4) / pxPerUnit;
}

/** "[B, 3, 512]" → [4, 3, 512]: symbolic sizes get a token size so they still draw. */
export const parseDims = (dims: string): number[] =>
  (dims.match(/[A-Za-z_][\w]*|\d+/g) ?? []).map((t) => (/^\d+$/.test(t) ? Number(t) : 4));

/** A value on the card: small magnitudes (gradients, ~1e-3) as a compact exponent so they don't all read "0.00". */
export const formatValue = (n: number) =>
  Math.abs(n) >= 100 || Number.isInteger(n) ? String(n) : Math.abs(n) < 0.01 ? n.toExponential(1) : n.toFixed(2);
const fmt = formatValue;

/** Shades every value of a view between its own min and max. */
function shader(values: number[][]) {
  const all = values.flat();
  const lo = Math.min(...all);
  const hi = Math.max(...all);
  return (v: number | undefined) => (v === undefined || !(hi > lo) ? 0.5 : (v - lo) / (hi - lo));
}

const cell = (key: string, col: number, row: number, shade: number, tip: string, extra: Partial<Cell> = {}): Cell => ({
  key,
  x: col * STEP,
  y: -row * STEP,
  w: S,
  h: S,
  d: DEPTH,
  shade,
  tip,
  ...extra,
});

function lay(v: View): Omit<Laid, "width" | "height" | "caption"> {
  const shade = shader(v.values);
  const lastDim = parseDims(v.dims).at(-1);
  switch (v.kind) {
    case "tokens":
      return {
        cells: v.rows.map((r, i) =>
          cell(`${r}:0`, i * 3, 0, shade(v.values[0]?.[i]), v.values[0]?.[i] !== undefined ? `${r} → ${v.values[0]![i]}` : r, {
            w: STEP * 3 - GAP,
            h: S * 1.4,
            label: r,
          }),
        ),
      };
    case "vectors":
    case "heads": {
      const split = v.kind === "heads" ? v.split : 1;
      const cells: Cell[] = [];
      v.rows.forEach((r, i) => {
        const nums = v.values[i] ?? [];
        // Heads need at least one cell per head, even when fewer numbers were given.
        const count = Math.max(nums.length, split, 1);
        for (let j = 0; j < count; j++) {
          const val = nums[Math.floor((j * nums.length) / count)];
          const slice = Math.floor((j * split) / count);
          cells.push(
            cell(`${r}:${j}`, j + slice * 0.35, i, shade(val), `${r}${split > 1 ? `, head ${slice + 1}` : ""}: ${val === undefined ? "…" : fmt(val)}`, {
              ...(j === 0 && { label: r }),
              ...(split > 1 && { slice }),
            }),
          );
        }
      });
      const shown = Math.max(...v.values.map((r) => r.length), 0);
      return { cells, ...(lastDim && lastDim > shown && { more: `…${lastDim}` }) };
    }
    case "matrix":
      return {
        cells: v.values.flatMap((row, i) =>
          row.map((val, j) =>
            cell(`${v.rows[i] ?? i}:${v.cols[j] ?? j}`, j, i, shade(val), `${v.rows[i] ?? i} → ${v.cols[j] ?? j}: ${fmt(val)}`, {
              ...(j === 0 && v.rows[i] && { label: v.rows[i] }),
            }),
          ),
        ),
      };
    case "patches": {
      const grid = v.values.length ? v.values : squareOf(v.rows.length);
      const cols = Math.max(...grid.map((r) => r.length), 1);
      return {
        cells: grid.flatMap((row, i) =>
          row.map((val, j) => {
            const label = v.rows[i * cols + j];
            return cell(`p${i}:${j}`, j, i, shade(val), label ? `${label}: ${fmt(val)}` : fmt(val), { ...(label && { label }) });
          }),
        ),
      };
    }
    case "signal": {
      if (v.values.length > 1) {
        // A spectrogram: time across, frequency up.
        return { cells: v.values.flatMap((row, i) => row.map((val, j) => cell(`s${i}:${j}`, j, i, shade(val), fmt(val)))) };
      }
      const row = v.values[0] ?? [];
      const peak = Math.max(...row.map(Math.abs), 1e-9);
      return {
        cells: row.map((val, j) => {
          const h = Math.max(0.04, (Math.abs(val) / peak) * S * 3);
          return cell(`s0:${j}`, j, 0, shade(val), fmt(val), { w: S * 0.6, h, y: val >= 0 ? h / 2 : -h / 2 });
        }),
      };
    }
    case "frames": {
      const cells: Cell[] = [];
      v.values.forEach((flat, f) => {
        const side = Math.max(1, Math.ceil(Math.sqrt(flat.length)));
        flat.forEach((val, k) => {
          const [i, j] = [Math.floor(k / side), k % side];
          // Each frame a little behind and to the right of the last, like a flipbook.
          cells.push(
            cell(`f${f}:${k}`, j + f * 0.35, i + f * 0.35, shade(val), `${v.rows[f] ?? `frame ${f + 1}`}: ${fmt(val)}`, {
              ...(k === 0 && v.rows[f] && { label: v.rows[f] }),
            }),
          );
        });
      });
      return { cells };
    }
    case "probs": {
      const ps = v.values[0] ?? [];
      return {
        cells: v.rows.map((r, i) => {
          const p = Math.max(0, Math.min(1, ps[i] ?? 0));
          const w = 0.1 + p * 1.8;
          return cell(`${r}:0`, 0, i * 1.2, p, `${r}: ${(p * 100).toFixed(0)}%`, { w, x: w / 2, label: r });
        }),
      };
    }
    case "tensor": {
      const [d, h, w] = [...parseDims(v.dims)].reverse().concat([4, 4, 4]).slice(0, 3).map(size) as [number, number, number];
      return { cells: [{ key: "t", x: 0, y: 0, w, h, d, shade: 0.5, tip: v.dims || "tensor" }] };
    }
  }
}

/** Box size for one tensor dimension: log-scaled so 512 and 50 000 both fit. */
const size = (dim: number) => Math.max(0.2, Math.min(3, 0.35 * Math.log2(dim + 1)));

/** Patches named but not given values: a square of mid-grey cells. */
function squareOf(n: number): number[][] {
  const side = Math.max(1, Math.ceil(Math.sqrt(n)));
  return Array.from({ length: Math.ceil(Math.max(n, 1) / side) }, () => Array(side).fill(0.5));
}

export function layoutView(v: View): Laid {
  let { cells, more } = lay(v);
  if (cells.length === 0) cells = [{ key: "empty", x: 0, y: 0, w: S, h: S, d: DEPTH, shade: 0.5, tip: v.dims || v.kind }];
  cells = cells.slice(0, MAX_CELLS);
  const minX = Math.min(...cells.map((c) => c.x - c.w / 2));
  const maxX = Math.max(...cells.map((c) => c.x + c.w / 2));
  const minY = Math.min(...cells.map((c) => c.y - c.h / 2));
  const maxY = Math.max(...cells.map((c) => c.y + c.h / 2));
  const [cx, cy] = [(minX + maxX) / 2, (minY + maxY) / 2];
  return {
    cells: cells.map((c) => ({ ...c, x: c.x - cx, y: c.y - cy })),
    width: maxX - minX,
    height: maxY - minY,
    caption: v.caption,
    ...(more && { more }),
  };
}

/**
 * Up to two views side by side, the pair centred on (0, 0). `rooms`: per view, the room its row labels need (rowLabelRoom);
 * the second view moves right only when its labels wouldn't fit in the usual gap.
 */
export function layoutViews(vs: View[], rooms: number[] = []): Laid[] {
  const laid = vs.slice(0, 2).map(layoutView);
  const gap = (i: number) => Math.max(VIEW_GAP, (rooms[i] ?? 0) + GAP);
  const total = laid.reduce((sum, l, i) => sum + l.width + (i > 0 ? gap(i) : 0), 0);
  let left = -total / 2;
  return laid.map((l, i) => {
    if (i > 0) left += gap(i);
    const dx = left + l.width / 2;
    left += l.width;
    return { ...l, labelRoom: rooms[i] ?? 0, cells: l.cells.map((c) => ({ ...c, x: c.x + dx })) };
  });
}
