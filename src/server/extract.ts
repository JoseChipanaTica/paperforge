import Anthropic from "@anthropic-ai/sdk";
import { betaZodOutputFormat } from "@anthropic-ai/sdk/helpers/beta/zod";
import { Architecture } from "../shared/graph";

const MODEL = process.env.PAPERFORGE_MODEL ?? "claude-opus-5-5";
const EFFORT = (process.env.PAPERFORGE_EFFORT ?? "medium") as "low" | "medium" | "high";

// ~1M-token context; a typical paper is 30–150k chars. Refuse rather than silently truncate.
const MAX_CHARS = 1_500_000;

const SYSTEM = `You turn deep learning papers into a block graph that a 3D viewer assembles, block by block, in front of an audience.

Extract the main model the paper proposes (the base configuration if there are several) as:
- blocks: the architectural components a reader would point to in the paper's main figure. Aim for 6–24 blocks. Include inputs and outputs. If a sub-structure is stacked N times, describe ONE copy and put its blocks in a group with repeat = N.
- edges: forward data flow between blocks. Use kind "residual" for skip connections and kind "cross" for links between parallel branches (e.g. encoder output into decoder cross-attention). The graph must be acyclic.
- equation: the block's defining equation in KaTeX LaTeX, using the paper's notation. Empty string when the paper gives none.
- input_shape / output_shape: concrete where the paper gives numbers, symbolic otherwise, e.g. "[B, n, 512]".
- hyperparameters: only values the paper states.
- source_quote: one sentence copied verbatim from the paper text below that best describes the block. The viewer highlights this exact text in the paper, so do not paraphrase, fix spacing, or merge sentences. Prefer sentences from the method section.

List blocks in assembly order: from the inputs to the outputs, following the data. When the model has parallel branches (encoder/decoder, two towers), finish one branch before the next.`;

export async function extractArchitecture(pages: string[], opts: { apiKey?: string } = {}): Promise<Architecture> {
  const paper = pages.map((p, i) => `<page number="${i + 1}">\n${p}\n</page>`).join("\n");
  if (paper.length > MAX_CHARS) {
    throw new Error(`Paper is too long to read in one pass (${paper.length.toLocaleString()} chars).`);
  }

  // Per-request client: on the hosted site each visitor brings their own key.
  const client = new Anthropic({ apiKey: opts.apiKey?.trim() || undefined });
  const response = await client.beta.messages.parse({
    model: MODEL,
    max_tokens: 16000,
    system: SYSTEM,
    output_config: { effort: EFFORT, format: betaZodOutputFormat(Architecture) },
    // On a safety decline, let the API route to a fallback model inside the same call.
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default",
    messages: [
      {
        role: "user",
        content: `<paper>\n${paper}\n</paper>\n\nExtract the architecture of the model this paper proposes.`,
      },
    ],
  });

  if (response.stop_reason === "refusal") {
    throw new Error(`Claude declined to read this paper (${response.stop_details?.category ?? "unspecified"}).`);
  }
  if (response.stop_reason === "max_tokens") {
    throw new Error("Claude ran out of output tokens before finishing the graph.");
  }
  if (!response.parsed_output) throw new Error("Claude returned a graph that did not match the schema.");
  return response.parsed_output;
}
