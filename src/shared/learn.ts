import { z } from "zod";
import type { Architecture } from "./graph";
import { MotionArtifact, cleanMotion } from "./motion";
import { TrainingArtifact, cleanTraining } from "./training";

/**
 * Learning artifacts: what Claude writes to help someone understand a forged paper.
 * Every step points at a block_id in the graph so the 3D model can follow along.
 * Same rule as graph.ts: no constraints structured outputs can't express.
 */

export const LEARN_KINDS = ["pytorch", "journey", "motion", "training"] as const;
export const LearnKind = z.enum(LEARN_KINDS);
export type LearnKind = z.infer<typeof LearnKind>;

export const PytorchArtifact = z.object({
  filename: z.string().describe("Python file name, e.g. 'transformer.py'"),
  code: z
    .string()
    .describe(
      "Complete PyTorch code: nn.Modules using the paper's hyperparameters, tensor-shape comments, and a __main__ block that runs one forward pass on random input and prints the output shape",
    ),
  block_map: z
    .array(
      z.object({
        block_id: z.string().describe("id of a block in <graph>"),
        symbol: z.string().describe("Name of the class or function in the code that implements it"),
      }),
    )
    .describe("One entry per block in <graph>"),
  notes: z.array(z.string()).describe("Assumptions made where the paper is silent"),
});
export type PytorchArtifact = z.infer<typeof PytorchArtifact>;

export const VIEW_KINDS = ["tokens", "vectors", "heads", "matrix", "patches", "signal", "frames", "probs", "tensor"] as const;
export const MODALITIES = ["text", "image", "audio", "video", "multimodal", "other"] as const;

/** What the data looks like at one point of the journey: a small, illustrative slice of the real tensor. */
export const View = z.object({
  kind: z
    .enum(VIEW_KINDS)
    .describe(
      "tokens: labelled tiles; vectors: one strip of numbers per item; heads: vectors cut into `split` heads; matrix: a heatmap (e.g. attention weights); patches: an image as a grid of cells; signal: a waveform (one row) or spectrogram (grid); frames: a stack of patch grids, one per video frame; probs: output probabilities; tensor: a plain box when nothing else fits",
    ),
  caption: z.string().describe("Short caption, e.g. 'attention weights for “sat”'"),
  rows: z
    .array(z.string())
    .describe("Item labels: tokens (tokens, vectors, heads, matrix rows), one label per cell row-major (patches), frame names (frames), classes (probs); empty for signal and tensor"),
  cols: z.array(z.string()).describe("Column labels: matrix keys or head names; usually empty"),
  values: z
    .array(z.array(z.number()))
    .describe(
      "Illustrative numbers, at most 12×12: tokens: [] or one row of ids; vectors/heads: one row of ≤ 8 numbers per item; matrix: rows × cols; patches: the intensity grid; signal: one row of samples or a grid; frames: one flattened grid per frame; probs: one row; tensor: []",
    ),
  dims: z.string().describe("The real tensor shape this slice comes from, e.g. '[3, 512]'"),
  split: z.number().int().describe("heads: the number of heads h; otherwise 1"),
});
export type View = z.infer<typeof View>;

const Moment = z.object({
  label: z.string().describe("The input or output in words, e.g. 'the cat sat'"),
  views: z.array(View).describe("1–2 views"),
});

export const JourneyArtifact = z.object({
  modality: z.enum(MODALITIES),
  analogy: z.object({
    title: z.string().describe("Short name for the everyday analogy"),
    text: z.string().describe("The analogy in a short paragraph a non-expert understands"),
  }),
  intro: z.string().describe("One or two sentences: what the reader will watch happen"),
  input: Moment.describe("The tiny concrete input, before any block"),
  steps: z
    .array(
      z.object({
        block_id: z.string().describe("id of a block in <graph>"),
        title: z.string().describe("Short step title"),
        what_happens: z.string().describe("One plain sentence on what this block does to the input"),
        narration: z.string().describe("2–3 sentences on what happens to the data here and why the paper does it this way"),
        shape_change: z.string().describe("Tensor shape in → out, e.g. '[B, 3, 512] → [B, 3, 512]'"),
        views: z.array(View).describe("1–2 views: the data after this block"),
      }),
    )
    .describe("One step per block, in data-flow order"),
  output: Moment.describe("The final result"),
  outro: z.string().describe("One sentence to remember"),
});
export type JourneyArtifact = z.infer<typeof JourneyArtifact>;

const MAX_SIDE = 12;
const MAX_VECTOR = 8;
const MAX_VIEWS = 2;

const finite = (n: number) => (Number.isFinite(n) ? n : 0);

/** A view a renderer can always draw: clamped, rectangular where it must be, or a plain tensor. */
export function cleanView(v: View): View {
  // Patches label every cell, and a frame row is a whole flattened grid: both hold up to 12 × 12.
  const rows = v.rows.slice(0, v.kind === "patches" ? MAX_SIDE * MAX_SIDE : MAX_SIDE);
  const cols = v.cols.slice(0, MAX_SIDE);
  const width = v.kind === "vectors" || v.kind === "heads" ? MAX_VECTOR : v.kind === "frames" ? MAX_SIDE * MAX_SIDE : MAX_SIDE;
  let values = v.values.slice(0, MAX_SIDE).map((r) => r.slice(0, width).map(finite));
  if (v.kind === "matrix") {
    // A heatmap is rectangular: pad short rows to the widest (or to the column labels).
    const w = Math.min(MAX_SIDE, Math.max(cols.length, ...values.map((r) => r.length)));
    values = values.map((r) => [...r, ...Array(w - r.length).fill(0)]);
  }
  const split = Math.max(1, Math.min(16, Math.round(finite(v.split)) || 1));
  const drawable =
    v.kind === "tensor" ||
    (["tokens", "vectors", "heads", "probs"].includes(v.kind) && rows.length > 0) ||
    (v.kind === "patches" && (rows.length > 0 || values.some((r) => r.length > 0))) ||
    (["matrix", "frames"].includes(v.kind) && values.some((r) => r.length > 0)) ||
    (v.kind === "signal" && values.some((r) => r.length > 0));
  if (!drawable) return { kind: "tensor", caption: v.caption, rows: [], cols: [], values: [], dims: v.dims, split: 1 };
  return { ...v, rows, cols, values, split: v.kind === "heads" ? split : 1 };
}

const cleanViews = (vs: View[]) => vs.slice(0, MAX_VIEWS).map(cleanView);

/** Keep a journey drawable: steps on blocks the graph has, views clamped, at most two per moment. */
export function cleanJourney(a: JourneyArtifact, graph: Architecture): JourneyArtifact {
  const ids = new Set(graph.blocks.map((b) => b.id));
  return {
    ...a,
    input: { ...a.input, views: cleanViews(a.input.views) },
    steps: a.steps.filter((s) => ids.has(s.block_id)).map((s) => ({ ...s, views: cleanViews(s.views) })),
    output: { ...a.output, views: cleanViews(a.output.views) },
  };
}

/**
 * What enters and leaves a block in a walkthrough: the step before's views (the input's, for the
 * first block) and the block's own. A block's inside walkthrough starts and ends on these.
 */
export function blockEnds(journey: JourneyArtifact | null | undefined, blockId: string): { entering: View[] | null; leaving: View[] | null } {
  const at = journey?.steps.findIndex((s) => s.block_id === blockId) ?? -1;
  if (!journey || at < 0) return { entering: null, leaving: null };
  return { entering: at > 0 ? journey.steps[at - 1]!.views : journey.input.views, leaving: journey.steps[at]!.views };
}

export const ARTIFACT_SCHEMA = {
  pytorch: PytorchArtifact,
  journey: JourneyArtifact,
  motion: MotionArtifact,
  training: TrainingArtifact,
} as const;

export type Artifact<K extends LearnKind> = z.infer<(typeof ARTIFACT_SCHEMA)[K]>;
export type AnyArtifact = PytorchArtifact | JourneyArtifact | MotionArtifact | TrainingArtifact;

/** What a block's inside view can have of its own: a walkthrough of the inside graph, and its motion. */
export type InsideLearnKind = "journey" | "motion";

/** Drop steps (or code mappings) that point at blocks the graph doesn't have; journeys are also clamped. */
export function pruneArtifact(kind: LearnKind, artifact: AnyArtifact, graph: Architecture): AnyArtifact {
  if (kind === "journey") return cleanJourney(artifact as JourneyArtifact, graph);
  if (kind === "motion") return cleanMotion(artifact as MotionArtifact, graph, null);
  if (kind === "training") {
    const t = artifact as TrainingArtifact;
    return cleanTraining(t, graph) ?? { ...t, toy: { ...t.toy, nodes: [] } };
  }
  const ids = new Set(graph.blocks.map((b) => b.id));
  const a = artifact as PytorchArtifact;
  return { ...a, block_map: a.block_map.filter((m) => ids.has(m.block_id)) };
}

/** How many blocks an artifact still ties to the model. */
export function artifactBlockCount(kind: LearnKind, artifact: AnyArtifact): number {
  if (kind === "training") return new Set((artifact as TrainingArtifact).toy.nodes.map((n) => n.block_id)).size;
  return kind === "pytorch" ? (artifact as PytorchArtifact).block_map.length : (artifact as JourneyArtifact | MotionArtifact).steps.length;
}

/** Enough left to show: code mapped to a block, a journey of at least two steps, motion for one step. */
export const isUsable = (kind: LearnKind, artifact: AnyArtifact) => artifactBlockCount(kind, artifact) >= (kind === "journey" || kind === "training" ? 2 : 1);

type LearnError = { type: "error"; message: string; code?: "needs_key" | "rate_limited" | "bad_input" | "not_ready" };

export type LearnEvent =
  | { type: "status"; message: string }
  | { type: "learn"; kind: LearnKind; artifact: AnyArtifact; source: "gallery" | "sample" | "claude"; saved: boolean }
  | LearnError;

/** A block's inside view: a small graph of its own, in the same shape as the model. */
export type InsideEvent =
  | { type: "status"; message: string }
  | { type: "inside"; block_id: string; graph: Architecture; source: "gallery" | "sample" | "claude"; saved: boolean }
  | LearnError;

/** A block's inside walkthrough or its motion (POST /api/inside-learn). */
export type InsideLearnEvent =
  | { type: "status"; message: string }
  | { type: "inside_learn"; block_id: string; kind: InsideLearnKind; artifact: AnyArtifact; source: "gallery" | "sample" | "claude"; saved: boolean }
  | LearnError;

export interface AskTurn {
  role: "user" | "assistant";
  text: string;
}

export type AskEvent = { type: "delta"; text: string } | { type: "done" } | LearnError;
