import { describe, expect, test } from "bun:test";
import { VIEW_KINDS, type View } from "../shared/learn";
import { MAX_CELLS, layoutView, layoutViews, parseDims } from "./specimenLayout";

const view = (v: Partial<View>): View => ({ kind: "tensor", caption: "c", rows: [], cols: [], values: [], dims: "[3, 512]", split: 1, ...v });
const grid = (n: number, m: number) => Array.from({ length: n }, (_, i) => Array.from({ length: m }, (_, j) => (i * m + j) / (n * m)));
const labels = (n: number) => Array.from({ length: n }, (_, i) => `r${i}`);
const finite = (l: ReturnType<typeof layoutView>) =>
  [l.width, l.height, ...l.cells.flatMap((c) => [c.x, c.y, c.w, c.h, c.d, c.shade])].every(Number.isFinite);

/** The biggest view cleanJourney lets through, for each kind. */
function biggest(kind: View["kind"]): View {
  switch (kind) {
    case "vectors":
    case "heads":
      return view({ kind, rows: labels(12), values: grid(12, 8), split: kind === "heads" ? 8 : 1 });
    case "frames":
      return view({ kind, rows: labels(12), values: grid(12, 12) });
    case "signal":
      return view({ kind, values: grid(12, 12) });
    case "tensor":
      return view({ kind, dims: "[512, 512, 512]" });
    default:
      return view({ kind, rows: labels(12), cols: labels(12), values: grid(12, 12) });
  }
}

describe("layoutView", () => {
  test("every kind stays under the cell cap, with finite geometry", () => {
    for (const kind of VIEW_KINDS) {
      const l = layoutView(biggest(kind));
      expect(l.cells.length).toBeGreaterThan(0);
      expect(l.cells.length).toBeLessThanOrEqual(MAX_CELLS);
      expect(finite(l)).toBe(true);
    }
  });

  test("heads slices cover every row", () => {
    const l = layoutView(view({ kind: "heads", rows: ["the", "cat", "sat"], values: grid(3, 8), split: 8 }));
    for (const row of ["the", "cat", "sat"]) {
      const slices = l.cells.filter((c) => c.key.startsWith(`${row}:`)).map((c) => c.slice);
      expect(new Set(slices)).toEqual(new Set([0, 1, 2, 3, 4, 5, 6, 7]));
    }
  });

  test("rows keep their key and height across steps", () => {
    const rows = ["the", "cat", "sat"];
    const a = layoutView(view({ kind: "vectors", rows, values: grid(3, 6) }));
    const b = layoutView(view({ kind: "vectors", rows, values: grid(3, 6).map((r) => r.map((x) => -x)) }));
    expect(new Set(a.cells.map((c) => c.key))).toEqual(new Set(b.cells.map((c) => c.key)));
    const y = (l: typeof a, row: string) => l.cells.find((c) => c.key === `${row}:0`)!.y;
    for (const row of rows) expect(y(a, row)).toBe(y(b, row));
  });

  test("tensor handles symbolic and huge dims", () => {
    for (const dims of ["[B, n, d_model]", "[1, 50000]", ""]) {
      const l = layoutView(view({ kind: "tensor", dims }));
      expect(l.cells.length).toBeGreaterThan(0);
      expect(finite(l)).toBe(true);
      expect(l.width).toBeLessThanOrEqual(3);
      expect(l.height).toBeLessThanOrEqual(3);
    }
  });

  test("an empty view lays out as a placeholder", () => {
    const l = layoutView(view({ kind: "matrix", values: [] }));
    expect(l.cells.length).toBeGreaterThan(0);
    expect(finite(l)).toBe(true);
  });

  test("probs bars scale with probability", () => {
    const l = layoutView(view({ kind: "probs", rows: ["Katze", "Hund", "Haus"], values: [[0.7, 0.2, 0.1]] }));
    expect(l.cells[0]!.w).toBeGreaterThan(l.cells[1]!.w);
    expect(l.cells[0]!.label).toBe("Katze");
  });

  test("vectors show the true width", () => {
    expect(layoutView(view({ kind: "vectors", rows: ["a"], values: [[1, 2]], dims: "[3, 512]" })).more).toBe("…512");
  });
});

describe("layoutViews", () => {
  test("two views sit side by side without overlapping", () => {
    const [a, b] = layoutViews([view({ kind: "tokens", rows: ["a", "b"] }), view({ kind: "tokens", rows: ["c"] })]);
    const right = (l: typeof a) => Math.max(...l!.cells.map((c) => c.x + c.w / 2));
    const left = (l: typeof a) => Math.min(...l!.cells.map((c) => c.x - c.w / 2));
    expect(left(b)).toBeGreaterThan(right(a));
  });
});

describe("parseDims", () => {
  test("numbers stay, symbols become 4", () => {
    expect(parseDims("[B, 3, 512]")).toEqual([4, 3, 512]);
    expect(parseDims("")).toEqual([]);
  });
});
