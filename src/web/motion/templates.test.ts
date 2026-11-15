import { describe, expect, test } from "bun:test";
import { OP_KINDS, type Op, type OpKind } from "../../shared/motion";
import { computeOp } from "./compute";
import { H, W } from "./frames";
import { framesFor } from "./templates";

const base = (o: Partial<Op>): Op => ({
  kind: "custom",
  caption: "What happens here",
  tokens: ["the", "cat", "sat"],
  input: [],
  weights: [],
  extra: [],
  fn: "",
  size: 1,
  stride: 1,
  count: 1,
  svg: "",
  ...o,
});
const grid = (r: number, c: number) => Array.from({ length: r }, (_, i) => Array.from({ length: c }, (_, j) => ((i * c + j) % 7) / 3 - 1));
const eye = (n: number) => Array.from({ length: n }, (_, i) => Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)));

/** A representative op of each kind, with the input it would get. */
function sampleOp(kind: OpKind): [Op, number[][]] {
  switch (kind) {
    case "convolution":
      return [base({ kind, weights: grid(3, 3), count: 64 }), grid(6, 6)];
    case "transposed_convolution":
      return [base({ kind, weights: grid(3, 3), stride: 2 }), grid(2, 2)];
    case "pooling":
      return [base({ kind, fn: "max", size: 2, stride: 2 }), grid(4, 4)];
    case "attention":
      return [base({ kind, weights: grid(3, 4), extra: [...grid(3, 4), ...grid(3, 4)] }), []];
    case "linear":
      return [base({ kind, weights: grid(4, 6) }), grid(1, 4)];
    case "embedding_lookup":
      return [base({ kind, weights: grid(5, 4) }), [[2, 0, 4]]];
    case "split_heads":
    case "concat":
      return [base({ kind, count: 2 }), grid(3, 4)];
    case "recurrent":
      return [base({ kind, weights: grid(4, 2) }), grid(3, 2)];
    case "dropout":
      return [base({ kind, extra: [[0.5]] }), grid(1, 8)];
    case "flatten":
      return [base({ kind }), grid(3, 3)];
    case "patch_embedding":
      return [base({ kind, size: 2, weights: grid(4, 3) }), grid(4, 4)];
    case "gate":
      return [base({ kind, fn: "glu", weights: eye(4) }), grid(1, 4)];
    case "moe_routing":
      return [base({ kind, count: 2, weights: grid(4, 4) }), grid(3, 4)];
    case "add_noise":
      return [base({ kind, extra: [[0.4]] }), grid(4, 4)];
    case "message_passing":
      return [base({ kind, fn: "sum", extra: [[0, 1, 0], [1, 0, 1], [0, 1, 0]] }), grid(3, 2)];
    case "activation":
      return [base({ kind, fn: "gelu" }), grid(1, 6)];
    case "normalization":
      return [base({ kind, fn: "layer" }), grid(2, 6)];
    case "residual_add":
      return [base({ kind, extra: grid(1, 4) }), grid(1, 4)];
    default:
      return [base({ kind }), grid(3, 4)];
  }
}

const within = (v: number, max: number) => v >= -1 && v <= max + 1;

describe("framesFor", () => {
  for (const kind of OP_KINDS) {
    test(`${kind}: finite frames inside the viewBox, under 12 s`, () => {
      const [op, input] = sampleOp(kind);
      const result = computeOp(op, input);
      if (kind !== "custom") expect(result.ok).toBe(true);
      const frames = framesFor(op, input, result);
      expect(frames.length).toBeGreaterThanOrEqual(1);
      expect(frames.length).toBeLessThanOrEqual(40);
      expect(frames.reduce((s, f) => s + f.dur, 0)).toBeLessThanOrEqual(12_000);
      for (const f of frames) {
        const ids = f.items.map((i) => i.id);
        expect(new Set(ids).size).toBe(ids.length);
        for (const item of f.items) {
          for (const [k, v] of Object.entries(item.attrs)) if (typeof v === "number") expect(Number.isFinite(v)).toBe(true);
          for (const k of ["x", "cx", "x1", "x2"]) if (typeof item.attrs[k] === "number") expect(within(item.attrs[k] as number, W)).toBe(true);
          for (const k of ["y", "cy", "y1", "y2"]) if (typeof item.attrs[k] === "number") expect(within(item.attrs[k] as number, H)).toBe(true);
        }
      }
    });
  }

  test("a result that didn't compute shows the captioned placeholder", () => {
    const op = base({ kind: "linear", caption: "Mix the features", weights: grid(3, 3) });
    const frames = framesFor(op, grid(1, 2), computeOp(op, grid(1, 2)));
    expect(frames.length).toBe(1);
    expect(frames[0]!.items.some((i) => i.text === "Mix the features")).toBe(true);
  });

  test("convolution writes each window's sum into its output cell", () => {
    const [op, input] = sampleOp("convolution");
    const result = computeOp(op, input);
    const frames = framesFor(op, input, result);
    const sums = result.stages.sums!;
    // Beat k (after the empty first frame) shows output cell k at full opacity.
    const beat = frames[3]!;
    const cell = beat.items.find((i) => i.id === "out-0-2")!;
    expect(cell.attrs.opacity).toBe(1);
    expect(beat.items.find((i) => i.id === "out-n-0-2")?.text).toBe(sums[0]![2]!.toFixed(Math.abs(sums[0]![2]!) < 1 && sums[0]![2] !== 0 ? 2 : 1));
    expect(frames.at(-1)!.items.some((i) => i.text?.includes("64 filters"))).toBe(true);
  });

  test("attention ends on the weights compute produced", () => {
    const [op, input] = sampleOp("attention");
    const result = computeOp(op, input);
    const frames = framesFor(op, input, result);
    const weightsFrame = frames.find((f) => f.items.some((i) => i.text?.includes("softmax")))!;
    const w = result.stages.weights!;
    expect(weightsFrame.items.find((i) => i.id === "s-n-1-2")?.text).toBe(w[1]![2]!.toFixed(2));
  });
});
