import { z } from "zod";
import type { Architecture } from "./graph";
import type { JourneyArtifact } from "./learn";
import { SVG_MAX, sanitizeSvg } from "./svg";

/**
 * Step motion: what each block does to the walkthrough's toy input, as a short chain of operations
 * the browser animates (and recomputes, so the arithmetic on screen is always right). Claude gives
 * only small inputs; an op no template fits is a sanitized SVG of Claude's own.
 */

export const OP_KINDS = [
  "convolution",
  "transposed_convolution",
  "pooling",
  "attention",
  "linear",
  "activation",
  "normalization",
  "residual_add",
  "softmax",
  "embedding_lookup",
  "positional_encoding",
  "split_heads",
  "concat",
  "recurrent",
  "dropout",
  "flatten",
  "patch_embedding",
  "gate",
  "moe_routing",
  "rotary_embedding",
  "add_noise",
  "message_passing",
  "custom",
] as const;
export type OpKind = (typeof OP_KINDS)[number];

const Matrix = z.array(z.array(z.number()));

export const Op = z.object({
  kind: z.enum(OP_KINDS),
  caption: z.string().describe("One short sentence on what this op does here, e.g. 'Slide a 3×3 kernel over the image'"),
  tokens: z.array(z.string()).describe("Item labels: words, patch names, node names, expert names"),
  input: Matrix.describe(
    "≤ 8×8: the data entering the chain (rows = items). Only the first op of a step needs it; later ops receive the previous op's output. For convolution/pooling/patch_embedding/add_noise: an image patch grid. For embedding_lookup: one row of token ids",
  ),
  weights: Matrix.describe(
    "≤ 8×8: the kernel (convolution, transposed_convolution), weight matrix in × out (linear, patch_embedding projection, recurrent), embedding table (embedding_lookup), Q rows (attention; empty means the queries are the input), router weights d × experts (moe_routing), gate weights (gate); otherwise []",
  ),
  extra: Matrix.describe(
    "The second operand: attention: K rows then V rows (same count each); linear: [[bias…]]; normalization: [[γ…],[β…]] or []; residual_add: the skip input; dropout: [[p]]; add_noise: [[ᾱ]] then noise rows; message_passing: the adjacency matrix; recurrent: [[h0…]]; otherwise []",
  ),
  fn: z
    .string()
    .describe(
      "Variant: convolution '' | 'depthwise' | 'dilated'; pooling 'max' | 'avg'; attention '' | 'causal' | 'cross'; activation 'relu' | 'gelu' | 'sigmoid' | 'tanh' | 'silu'; normalization 'layer' | 'batch' | 'rms'; gate 'glu' | 'swiglu' | 'lstm' | 'gru'; message_passing 'sum' | 'mean' | 'max'; otherwise ''",
    ),
  size: z.number().int().describe("Kernel / window / patch size (1–5); otherwise 1"),
  stride: z.number().int().describe("Stride (1–4); otherwise 1"),
  count: z.number().int().describe("How many: filters (convolution), heads (split_heads), experts top-k (moe_routing); otherwise 1"),
  svg: z.string().describe("custom only: an animated SVG (SMIL), viewBox '0 0 360 220'; otherwise ''"),
});
export type Op = z.infer<typeof Op>;

export const MotionArtifact = z.object({
  steps: z
    .array(
      z.object({
        block_id: z.string().describe("A walkthrough step's block_id, in the walkthrough's order"),
        ops: z.array(Op).describe("1–3 ops, played in order; each op's output is the next op's input"),
      }),
    )
    .describe("One entry per walkthrough step"),
});
export type MotionArtifact = z.infer<typeof MotionArtifact>;

export const MOTION_LIMITS = { side: 8, ops: 3, svg: SVG_MAX } as const;

const finite = (n: number) => (Number.isFinite(n) ? n : 0);
const clampInt = (n: number, lo: number, hi: number) => Math.max(lo, Math.min(hi, Math.round(finite(n)) || lo));

/** At most `maxRows` × 8, rectangular, finite. */
function cleanMatrix(m: number[][], maxRows: number = MOTION_LIMITS.side): number[][] {
  const rows = m.slice(0, maxRows).map((r) => r.slice(0, MOTION_LIMITS.side).map(finite));
  const w = Math.max(0, ...rows.map((r) => r.length));
  return rows.map((r) => [...r, ...Array(w - r.length).fill(0)]);
}

const empty = (m: number[][]) => m.length === 0 || m.every((r) => r.length === 0);

/** Ops that read the chain's input (the first op of a step must bring its own). */
const READS_INPUT = new Set<OpKind>(OP_KINDS.filter((k) => k !== "custom"));

/** True when an op can't be computed from what it carries. */
function missing(o: Op, first: boolean): boolean {
  // Attention can bring its queries as Q (weights) instead of an input.
  if (first && READS_INPUT.has(o.kind) && empty(o.input) && !(o.kind === "attention" && !empty(o.weights))) return true;
  switch (o.kind) {
    case "convolution":
    case "transposed_convolution":
    case "linear":
    case "embedding_lookup":
    case "patch_embedding":
      return empty(o.weights);
    case "attention":
      return o.extra.length < 2;
    case "message_passing":
      return empty(o.extra);
    default:
      return false;
  }
}

export function cleanOp(o: Op, first = true): Op {
  const op: Op = {
    ...o,
    input: cleanMatrix(o.input),
    weights: cleanMatrix(o.weights),
    // Attention stacks K and V rows, add_noise ᾱ and the noise: up to 8 rows each.
    extra: cleanMatrix(o.extra, o.kind === "attention" || o.kind === "add_noise" ? 2 * MOTION_LIMITS.side : MOTION_LIMITS.side),
    tokens: o.tokens.slice(0, 16),
    size: clampInt(o.size, 1, 5),
    stride: clampInt(o.stride, 1, 4),
    count: clampInt(o.count, 1, 512),
    svg: o.kind === "custom" ? sanitizeSvg(o.svg) : "",
  };
  if (op.kind !== "custom" && missing(op, first)) return { ...op, kind: "custom", svg: "" };
  return op;
}

/** Keep motion drawable: steps on the walkthrough's blocks, in its order, at most 3 clean ops each. */
export function cleanMotion(m: MotionArtifact, graph: Architecture, journey: JourneyArtifact | null): MotionArtifact {
  const blocks = new Set(graph.blocks.map((b) => b.id));
  const order = journey ? journey.steps.map((s) => s.block_id) : graph.blocks.map((b) => b.id);
  const rank = new Map(order.map((id, i) => [id, i] as const));
  const seen = new Set<string>();
  const steps = m.steps
    .filter((s) => blocks.has(s.block_id) && rank.has(s.block_id) && !seen.has(s.block_id) && seen.add(s.block_id))
    .sort((a, b) => rank.get(a.block_id)! - rank.get(b.block_id)!)
    .map((s) => ({ ...s, ops: s.ops.slice(0, MOTION_LIMITS.ops).map((o, i) => cleanOp(o, i === 0)) }))
    .filter((s) => s.ops.length > 0);
  return { steps };
}

/** True when every animated step is a step of this walkthrough (a sample made for another graph isn't). */
export const motionFits = (m: MotionArtifact, journey: JourneyArtifact) => {
  const steps = new Set(journey.steps.map((s) => s.block_id));
  return m.steps.length > 0 && m.steps.every((s) => steps.has(s.block_id));
};
