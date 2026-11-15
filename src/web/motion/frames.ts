import { HIGHLIGHTER, INK, SHEET, VERMILION } from "../../shared/palette";

/**
 * The frame model every animation is written in: a frame is the full set of SVG items on screen
 * at one moment, and `dur` is how long the tween into it takes. The player matches items by id,
 * so an item that keeps its id moves (a cell sliding into a flattened row) instead of popping.
 */

export const W = 360;
export const H = 220;

export interface Item {
  id: string;
  tag: "rect" | "text" | "line" | "circle" | "path" | "polyline";
  attrs: Record<string, string | number>;
  text?: string;
}

export interface Frame {
  dur: number;
  items: Item[];
}

export const COLOR = { ink: INK, paper: SHEET, mark: HIGHLIGHTER, result: VERMILION, faint: "#c9bfa9", soft: "#5d574b" } as const;
const PALE = "#fbf1cf";

const hex = (c: string) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));
const mix = (a: string, b: string, t: number) => {
  const [x, y] = [hex(a), hex(b)];
  return `#${x.map((v, i) => Math.round(v + (y[i]! - v) * t).toString(16).padStart(2, "0")).join("")}`;
};

/** A value's colour on the printed ramp: pale → highlighter → vermilion. */
export function shade(v: number, lo: number, hi: number): string {
  const t = hi > lo ? Math.max(0, Math.min(1, (v - lo) / (hi - lo))) : 0.5;
  return t < 0.5 ? mix(PALE, HIGHLIGHTER, t * 2) : mix(HIGHLIGHTER, VERMILION, (t - 0.5) * 2);
}

/** Lighter prints of one hue, one per head / expert / slice. */
export const SLICES = ["#b8324a", "#c2410c", "#d97706", "#3f6212", "#0f766e", "#1d4ed8", "#6d28d9", "#9d174d"];

export const fmt = (v: number) => (Math.abs(v) >= 100 ? v.toFixed(0) : Math.abs(v) >= 10 ? v.toFixed(1) : v.toFixed(Math.abs(v) < 1 && v !== 0 ? 2 : 1));
export const range = (m: number[][]) => {
  const all = m.flat().filter(Number.isFinite);
  return all.length ? ([Math.min(...all), Math.max(...all)] as const) : ([0, 1] as const);
};

/** The largest cell that fits rows × cols in a box, capped so small matrices stay readable. */
export const fit = (rows: number, cols: number, w: number, h: number, max = 22) => Math.max(6, Math.min(max, w / Math.max(cols, 1), h / Math.max(rows, 1)));

export function label(id: string, text: string, x: number, y: number, o: { size?: number; anchor?: string; color?: string; weight?: number } = {}): Item {
  return {
    id,
    tag: "text",
    text,
    attrs: { x, y, "font-size": o.size ?? 10, "text-anchor": o.anchor ?? "middle", fill: o.color ?? COLOR.ink, "font-family": "JetBrains Mono, monospace", "font-weight": o.weight ?? 400, "dominant-baseline": "middle" },
  };
}

export function box(id: string, x: number, y: number, w: number, h: number, o: { fill?: string; stroke?: string; dash?: boolean; width?: number; opacity?: number } = {}): Item {
  return {
    id,
    tag: "rect",
    attrs: {
      x,
      y,
      width: Math.max(0, w),
      height: Math.max(0, h),
      fill: o.fill ?? "none",
      stroke: o.stroke ?? COLOR.ink,
      "stroke-width": o.width ?? 1,
      opacity: o.opacity ?? 1,
      ...(o.dash && { "stroke-dasharray": "3 2" }),
    },
  };
}

export function line(id: string, x1: number, y1: number, x2: number, y2: number, o: { color?: string; width?: number; opacity?: number } = {}): Item {
  return { id, tag: "line", attrs: { x1, y1, x2, y2, stroke: o.color ?? COLOR.ink, "stroke-width": o.width ?? 1, opacity: o.opacity ?? 1 } };
}

export function dot(id: string, cx: number, cy: number, r: number, fill: string = COLOR.result): Item {
  return { id, tag: "circle", attrs: { cx, cy, r, fill, stroke: COLOR.ink, "stroke-width": 0.75 } };
}

/**
 * A matrix as shaded cells (numbers printed when the cells are big enough). Ids are
 * `${id}-${i}-${j}`, so the same matrix drawn in two frames keeps its cells.
 */
export function grid(
  id: string,
  m: number[][],
  x: number,
  y: number,
  cell: number,
  o: { scale?: readonly [number, number]; numbers?: boolean; fill?: (i: number, j: number, v: number) => string; opacity?: (i: number, j: number) => number } = {},
): Item[] {
  const [lo, hi] = o.scale ?? range(m);
  const items: Item[] = [];
  // One number size for the whole grid, sized for its longest number (monospace digits ≈ 0.6 em).
  const longest = Math.max(1, ...m.flat().map((v) => fmt(v).length));
  const size = Math.min(9, (cell - 3) / (longest * 0.6));
  m.forEach((row, i) =>
    row.forEach((v, j) => {
      const cx = x + j * cell;
      const cy = y + i * cell;
      items.push({
        id: `${id}-${i}-${j}`,
        tag: "rect",
        attrs: { x: cx, y: cy, width: cell - 1, height: cell - 1, fill: o.fill?.(i, j, v) ?? shade(v, lo, hi), stroke: COLOR.ink, "stroke-width": 0.5, opacity: o.opacity?.(i, j) ?? 1 },
      });
      if ((o.numbers ?? cell >= 18) && Number.isFinite(v)) items.push(label(`${id}-n-${i}-${j}`, fmt(v), cx + (cell - 1) / 2, cy + (cell - 1) / 2, { size }));
    }),
  );
  return items;
}

/** Horizontal bars, one per value (probabilities, gate openings), with optional labels on the left. */
export function bars(id: string, values: number[], x: number, y: number, w: number, h: number, labels: string[] = [], max = Math.max(1e-9, ...values.map(Math.abs))): Item[] {
  const row = h / Math.max(values.length, 1);
  return values.flatMap((v, i) => {
    const len = Math.max(1, (Math.abs(v) / max) * w);
    const items: Item[] = [box(`${id}-${i}`, x, y + i * row + 1, len, row - 3, { fill: v >= 0 ? COLOR.result : COLOR.soft, stroke: COLOR.ink, width: 0.5 })];
    items.push(label(`${id}-v-${i}`, fmt(v), x + len + 3, y + i * row + row / 2, { size: 8, anchor: "start" }));
    if (labels[i] !== undefined) items.push(label(`${id}-l-${i}`, labels[i]!, x - 4, y + i * row + row / 2, { size: 9, anchor: "end" }));
    return items;
  });
}

/** Row labels to the left of a grid. */
export const rowLabels = (id: string, labels: string[], x: number, y: number, cell: number): Item[] =>
  labels.map((t, i) => label(`${id}-${i}`, t.slice(0, 8), x - 4, y + i * cell + cell / 2, { size: 9, anchor: "end" }));

/** Anything a frame can be built from: items, lists of them (nested), or nothing. */
export type Part = Item | Part[] | false | null | undefined;
const flatten = (p: Part): Item[] => (!p ? [] : Array.isArray(p) ? p.flatMap(flatten) : [p]);

/** A frame made of several item lists. */
export const frame = (dur: number, ...parts: Part[]): Frame => ({ dur, items: parts.flatMap(flatten) });
