import Anthropic from "@anthropic-ai/sdk";
import { betaZodOutputFormat } from "@anthropic-ai/sdk/helpers/beta/zod";
import type { Architecture, Block } from "../shared/graph";
import { ARTIFACT_SCHEMA, type AnyArtifact, type AskTurn, type LearnKind } from "../shared/learn";
import { formatPaper } from "./extract";

/**
 * The tutor: Claude calls behind the learning features. Every request starts with the same
 * system prompt and the same [paper, graph] prefix, so after the first feature on a paper
 * the rest read it from the prompt cache. Keep everything before the breakpoint identical
 * across kinds, and keep effort fixed (an effort change invalidates the cache).
 */

const MODEL = process.env.PAPERFORGE_MODEL ?? "claude-opus-5-5";
const EFFORT = "medium" as const;
// Under Vercel's 300 s function limit; also opts out of the SDK's non-streaming duration guard.
const TIMEOUT_MS = 290_000;
// On a safety decline, let the API route to a fallback model inside the same call (as extract.ts).
const fallback = () => ({ betas: ["server-side-fallback-2026-07-01"], fallbacks: "default" as const });

export const TUTOR_SYSTEM = `You are Paperforge's tutor. A reader has just watched a deep learning paper's architecture assemble itself as a 3D model, block by block, and wants to understand it.

You get the paper's text and the model's block graph (<graph>, JSON). Ground everything in the paper: use its notation, hyperparameters and terminology, and say so when you go beyond what it states.

Whenever you refer to a block, use its exact "id" from <graph>; the viewer uses those ids to light up the matching block. Never invent ids.

Write for a curious engineer: plain language first, then the precise detail. Be concrete and brief.`;

export const LEARN_TASK: Record<LearnKind, string> = {
  pytorch: `Write PyTorch code that implements this architecture.
- One nn.Module per distinct kind of block (reuse a module for repeated blocks), using the paper's hyperparameters as defaults.
- Comment tensor shapes on the important lines.
- End with an if __name__ == "__main__": block that builds the model, runs one forward pass on random input and prints the output shape.
- block_map: one entry for every block in <graph>, naming the class, function or variable in your code that implements it.
- notes: every assumption you made where the paper is silent or ambiguous.`,
  walkthrough: `Write a guided walkthrough of this architecture: one step per block in <graph>, in the order data flows through the model (finish one branch before the next).
- narration: 2–3 sentences on what happens to the data in this block and why the paper does it this way.
- shape_change: the tensor shape going in and out, e.g. "[B, n, 512] → [B, n, 512]".`,
  example: `Explain this architecture to a non-expert with an everyday analogy and a tiny worked example.
- analogy: one vivid everyday analogy for the whole model.
- toy_input: a tiny concrete input of 3–5 tokens (or a tiny image, signal, etc. for non-text models).
- steps: one per block in <graph>, in data-flow order; what_happens in one plain sentence, toy_value showing concretely what the toy input looks like after that block.
- takeaway: one sentence to remember.`,
};

export function paperPrefix(pages: string[], graph: Architecture): Anthropic.Beta.BetaTextBlockParam[] {
  return [
    { type: "text", text: `<paper>\n${formatPaper(pages)}\n</paper>` },
    { type: "text", text: `<graph>\n${JSON.stringify(graph)}\n</graph>`, cache_control: { type: "ephemeral" } },
  ];
}

export function learnRequest(kind: LearnKind, pages: string[], graph: Architecture) {
  return {
    model: MODEL,
    max_tokens: 32000,
    system: TUTOR_SYSTEM,
    output_config: { effort: EFFORT, format: betaZodOutputFormat(ARTIFACT_SCHEMA[kind]) },
    ...fallback(),
    messages: [{ role: "user" as const, content: [...paperPrefix(pages, graph), { type: "text" as const, text: LEARN_TASK[kind] }] }],
  };
}

export interface AskInput {
  pages: string[];
  graph: Architecture;
  block: Block;
  question: string;
  history: AskTurn[];
}

export function askRequest({ pages, graph, block, question, history }: AskInput) {
  const focus = `The reader is asking about this block:\n${JSON.stringify({
    id: block.id,
    label: block.label,
    type: block.type,
    equation: block.equation,
    input_shape: block.input_shape,
    output_shape: block.output_shape,
  })}\nAnswer in a few short paragraphs at most. Plain text; LaTeX only inline as $...$.`;
  return {
    model: MODEL,
    max_tokens: 4000,
    system: TUTOR_SYSTEM,
    output_config: { effort: EFFORT },
    ...fallback(),
    messages: [
      { role: "user" as const, content: [...paperPrefix(pages, graph), { type: "text" as const, text: focus }] },
      ...history.map((t) => ({ role: t.role, content: t.text })),
      { role: "user" as const, content: question },
    ],
  };
}

const client = (apiKey?: string) => new Anthropic({ apiKey: apiKey?.trim() || undefined, timeout: TIMEOUT_MS });

function logUsage(what: string, usage: Anthropic.Beta.BetaUsage) {
  // Token counts only: never log the key or the content.
  console.log(
    `[tutor] ${what}: input ${usage.input_tokens}, cache write ${usage.cache_creation_input_tokens ?? 0}, cache read ${usage.cache_read_input_tokens ?? 0}, output ${usage.output_tokens}`,
  );
}

export async function generateArtifact(
  kind: LearnKind,
  { pages, graph, apiKey }: { pages: string[]; graph: Architecture; apiKey?: string },
): Promise<AnyArtifact> {
  const response = await client(apiKey).beta.messages.parse(learnRequest(kind, pages, graph));
  logUsage(kind, response.usage);
  if (response.stop_reason === "refusal") {
    throw new Error(`Claude declined to write this (${response.stop_details?.category ?? "unspecified"}).`);
  }
  if (response.stop_reason === "max_tokens") throw new Error("Claude ran out of output tokens before finishing.");
  if (!response.parsed_output) throw new Error("Claude returned something that didn't match the expected format.");
  return response.parsed_output as AnyArtifact;
}

export async function* askStream(input: AskInput & { apiKey?: string }): AsyncIterable<string> {
  const stream = client(input.apiKey).beta.messages.stream(askRequest(input));
  for await (const event of stream) {
    if (event.type === "content_block_delta" && event.delta.type === "text_delta") yield event.delta.text;
  }
  const final = await stream.finalMessage();
  logUsage("ask", final.usage);
  if (final.stop_reason === "refusal") throw new Error("Claude declined to answer that.");
}
