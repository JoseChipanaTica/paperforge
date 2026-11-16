import { describe, expect, test } from "bun:test";
import { VIEW_KINDS, type View } from "../shared/learn";
import { MAX_CELLS, colLabelMode, formatValue, labelPx, layoutView, layoutViews, parseDims, rowLabelRoom, rowLabelsOf } from "./specimenLayout";

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

describe("formatValue", () => {
  test("small magnitudes read as a compact exponent, not 0.00", () => {
    expect(formatValue(0.0012)).toBe("1.2e-3");
    expect(formatValue(-0.00056)).toBe("-5.6e-4");
    expect(formatValue(0.009)).toBe("9.0e-3");
  });

  test("other values are unchanged", () => {
    expect(formatValue(0)).toBe("0");
    expect(formatValue(3)).toBe("3");
    expect(formatValue(0.01)).toBe("0.01");
    expect(formatValue(0.5)).toBe("0.50");
    expect(formatValue(-1.234)).toBe("-1.23");
    expect(formatValue(123.4)).toBe("123.4");
  });
});

describe("labels that don't fit", () => {
  const vocab = ["<s>", "the", "cat", "sat", "die", "Katze", "saß", "<eos>"];

  test("column labels sit flat while the longest fits its column", () => {
    expect(colLabelMode(["the", "cat", "sat"], 40)).toBe("flat");
  });

  test("too wide for their columns, they stand on end, and vanish when even a line won't fit", () => {
    expect(colLabelMode(vocab, 20)).toBe("rotated");
    expect(colLabelMode(["E[cat,3]", "E[sat,0]"], 30)).toBe("rotated");
    expect(colLabelMode(vocab, 8)).toBe("none");
    expect(colLabelMode([], 40)).toBe("none");
  });

  test("row labels get room for the longest one, more as the card shrinks", () => {
    const rows = rowLabelsOf(view({ kind: "matrix", rows: ["<s>", "die", "Katze"], cols: [], values: grid(3, 3) }));
    expect(rows).toEqual(["<s>", "die", "Katze"]);
    const room = rowLabelRoom(rows, 100);
    expect(room).toBeGreaterThan(rowLabelRoom(["die"], 100));
    expect(rowLabelRoom(rows, 50)).toBeGreaterThan(room);
    expect(room * 100).toBeGreaterThanOrEqual(labelPx("Katze", 11)); // the whole word, in pixels
    expect(rowLabelRoom([], 100)).toBe(0);
    expect(rowLabelsOf(view({ kind: "tokens", rows: ["a"] }))).toEqual([]);
  });

  test("the second view moves over only when its row labels need more than the gap", () => {
    const vs = [view({ kind: "tokens", rows: ["a", "b"] }), view({ kind: "matrix", rows: ["Katze"], values: grid(1, 3) })];
    const left = (l: { cells: { x: number; w: number }[] }) => Math.min(...l.cells.map((c) => c.x - c.w / 2));
    const right = (l: { cells: { x: number; w: number }[] }) => Math.max(...l.cells.map((c) => c.x + c.w / 2));
    const plain = layoutViews(vs);
    expect(layoutViews(vs, [0, 0.1])).toEqual(plain.map((l) => ({ ...l, labelRoom: l === plain[0] ? 0 : 0.1 })));
    const [a, b] = layoutViews(vs, [0, 0.9]);
    expect(left(b!) - right(a!)).toBeGreaterThanOrEqual(0.9);
    expect(b!.labelRoom).toBe(0.9);
  });
});
