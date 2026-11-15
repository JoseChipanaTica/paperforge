import { z } from "zod";
import type { Architecture } from "./graph";

/**
 * Learning artifacts: what Claude writes to help someone understand a forged paper.
 * Every step points at a block_id in the graph so the 3D model can follow along.
 * Same rule as graph.ts: no constraints structured outputs can't express.
 */

export const LEARN_KINDS = ["pytorch", "walkthrough", "example"] as const;
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

export const WalkthroughArtifact = z.object({
  intro: z.string().describe("One or two sentences: what the tour will show"),
  steps: z
    .array(
      z.object({
        block_id: z.string().describe("id of a block in <graph>"),
        title: z.string().describe("Short step title"),
        narration: z.string().describe("2–3 sentences on what happens to the data in this block and why"),
        shape_change: z.string().describe("Tensor shape in → out, e.g. '[B, n, 512] → [B, n, 512]'"),
      }),
    )
    .describe("One step per block, in data-flow order"),
  outro: z.string().describe("One or two sentences tying it together"),
});
export type WalkthroughArtifact = z.infer<typeof WalkthroughArtifact>;

export const ExampleArtifact = z.object({
  analogy: z.object({
    title: z.string().describe("Short name for the everyday analogy"),
    text: z.string().describe("The analogy in a short paragraph a non-expert understands"),
  }),
  toy_input: z.string().describe("A tiny concrete input, e.g. 'the cat sat'"),
  steps: z
    .array(
      z.object({
        block_id: z.string().describe("id of a block in <graph>"),
        what_happens: z.string().describe("Plain-language sentence on what this block does to the toy input"),
        toy_value: z.string().describe("What the toy input looks like after this block, concretely"),
      }),
    )
    .describe("One step per block, in data-flow order"),
  takeaway: z.string().describe("One sentence to remember"),
});
export type ExampleArtifact = z.infer<typeof ExampleArtifact>;

export const ARTIFACT_SCHEMA = {
  pytorch: PytorchArtifact,
  walkthrough: WalkthroughArtifact,
  example: ExampleArtifact,
} as const;

export type Artifact<K extends LearnKind> = z.infer<(typeof ARTIFACT_SCHEMA)[K]>;
export type AnyArtifact = PytorchArtifact | WalkthroughArtifact | ExampleArtifact;

/** Drop steps (or code mappings) that point at blocks the graph doesn't have. */
export function pruneArtifact(kind: LearnKind, artifact: AnyArtifact, graph: Architecture): AnyArtifact {
  const ids = new Set(graph.blocks.map((b) => b.id));
  if (kind === "pytorch") {
    const a = artifact as PytorchArtifact;
    return { ...a, block_map: a.block_map.filter((m) => ids.has(m.block_id)) };
  }
  const a = artifact as WalkthroughArtifact | ExampleArtifact;
  return { ...a, steps: a.steps.filter((s) => ids.has(s.block_id)) } as AnyArtifact;
}

/** How many blocks an artifact still ties to the model (0 means it's unusable). */
export function artifactBlockCount(kind: LearnKind, artifact: AnyArtifact): number {
  return kind === "pytorch"
    ? (artifact as PytorchArtifact).block_map.length
    : (artifact as WalkthroughArtifact | ExampleArtifact).steps.length;
}

type LearnError = { type: "error"; message: string; code?: "needs_key" | "rate_limited" | "bad_input" };

export type LearnEvent =
  | { type: "status"; message: string }
  | { type: "learn"; kind: LearnKind; artifact: AnyArtifact; source: "gallery" | "sample" | "claude"; saved: boolean }
  | LearnError;

/** A block's inside view: a small graph of its own, in the same shape as the model. */
export type InsideEvent =
  | { type: "status"; message: string }
  | { type: "inside"; block_id: string; graph: Architecture; source: "gallery" | "sample" | "claude"; saved: boolean }
  | LearnError;

export interface AskTurn {
  role: "user" | "assistant";
  text: string;
}

export type AskEvent = { type: "delta"; text: string } | { type: "done" } | LearnError;
