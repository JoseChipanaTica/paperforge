import { z } from "zod";

/**
 * The block graph: the contract between Claude (structured output), the server and the
 * 3D front end. Keep it free of constraints structured outputs can't express
 * (min/max, string lengths, recursion).
 */

export const BLOCK_TYPES = [
  "input",
  "embedding",
  "positional_encoding",
  "multi_head_attention",
  "cross_attention",
  "attention",
  "feed_forward",
  "add_norm",
  "layer_norm",
  "residual_add",
  "linear",
  "activation",
  "softmax",
  "convolution",
  "pooling",
  "recurrent",
  "normalization",
  "dropout",
  "concat",
  "output",
  "other",
] as const;

export const BlockType = z.enum(BLOCK_TYPES);
export type BlockType = z.infer<typeof BlockType>;

export const Hyperparameter = z.object({
  name: z.string().describe("Symbol or short name as written in the paper, e.g. 'd_model', 'h', 'kernel'"),
  value: z.string().describe("Value as written, e.g. '512', '8', '3x3'"),
});
export type Hyperparameter = z.infer<typeof Hyperparameter>;

export const Group = z.object({
  id: z.string().describe("snake_case id, e.g. 'encoder'"),
  label: z.string().describe("Display name, e.g. 'Encoder layer'"),
  repeat: z.number().int().describe("How many times this group is stacked (N). 1 if not repeated."),
});
export type Group = z.infer<typeof Group>;

export const Block = z.object({
  id: z.string().describe("Unique snake_case id"),
  label: z.string().describe("Short display name, e.g. 'Multi-Head Attention'"),
  type: BlockType,
  group: z.string().describe("id of the group this block belongs to, or empty string"),
  equation: z
    .string()
    .describe("Defining equation in KaTeX-compatible LaTeX (no $ delimiters), or empty string"),
  input_shape: z.string().describe("Tensor shape entering the block, e.g. '[B, n, 512]'"),
  output_shape: z.string().describe("Tensor shape leaving the block"),
  hyperparameters: z.array(Hyperparameter),
  description: z.string().describe("One plain-English sentence on what this block does"),
  source_quote: z
    .string()
    .describe(
      "One sentence copied VERBATIM from the paper text that describes this block. Copy exactly, character for character.",
    ),
});
export type Block = z.infer<typeof Block>;

export const Edge = z.object({
  from: z.string(),
  to: z.string(),
  kind: z
    .enum(["data", "residual", "cross"])
    .describe(
      "'data' for the main forward path, 'residual' for skip connections, 'cross' for links between parallel branches (e.g. encoder output into decoder cross-attention)",
    ),
});
export type Edge = z.infer<typeof Edge>;

export const Architecture = z.object({
  paper_title: z.string(),
  model_name: z.string().describe("Name of the proposed model, e.g. 'Transformer'"),
  summary: z.string().describe("One sentence a non-expert understands"),
  hyperparameters: z.array(Hyperparameter).describe("Global hyperparameters of the base model"),
  groups: z.array(Group),
  blocks: z
    .array(Block)
    .describe("In assembly order: follow the data from the inputs to the outputs"),
  edges: z.array(Edge),
});
export type Architecture = z.infer<typeof Architecture>;

/** Where a block's source quote was found in the extracted paper text. */
export interface SourceSpan {
  page: number; // 0-based page index
  start: number; // char offsets into pages[page]
  end: number;
}

export type LocatedBlock = Block & { source: SourceSpan | null };
export type LocatedArchitecture = Omit<Architecture, "blocks"> & {
  arxiv_id: string;
  blocks: LocatedBlock[];
};

export type ForgeEvent =
  | { type: "status"; stage: "fetch" | "read" | "extract" | "locate"; message: string }
  | { type: "paper"; arxiv_id: string; pages: string[] }
  | {
      type: "graph";
      graph: LocatedArchitecture;
      source: GraphSource;
      /** Present only for fresh Claude output: lets the user publish it to the gallery. */
      save_token?: string;
    }
  | { type: "error"; message: string; code?: "needs_key" | "rate_limited" | "bad_input" };

/** Where /api/forge got the architecture from. */
export type GraphSource = "gallery" | "sample" | "cache" | "claude";

/** A gallery card: paper metadata without the graph. */
export interface PaperSummary {
  arxiv_id: string;
  paper_title: string;
  model_name: string;
  summary: string;
  block_count: number;
  created_at: string; // ISO-8601
}
