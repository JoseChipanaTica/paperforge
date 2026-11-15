import Anthropic from "@anthropic-ai/sdk";
import { betaZodOutputFormat } from "@anthropic-ai/sdk/helpers/beta/zod";
import { z } from "zod";
import { Architecture, type Block } from "../shared/graph";
import { ARTIFACT_SCHEMA, type AnyArtifact, type AskTurn, type LearnKind } from "../shared/learn";
import { DEFAULT_LEVEL, languageName, levelReader, type Lang, type Level } from "../shared/translate";
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

/** A block's inside view: its internal structure as a small graph of its own (same shape as the model). */
export function insideTask(block: Block): string {
  return `Look inside the block "${block.id}" (${block.label}): describe its internal structure as a small block graph of its own, the way the paper's text and equations define it.
- blocks: 3–10 internal components in data-flow order, with new snake_case ids (not ids from <graph>). Start with one block of type "input" for what enters (input_shape ${block.input_shape}) and end with one of type "output" for what leaves (output_shape ${block.output_shape}). Give every component the most specific type.
- input_shape / output_shape: concrete, using the paper's hyperparameters, so the reader sees how the tensor changes inside (e.g. split into heads and back).
- label: a short plain-text name of 1–4 words, no LaTeX or symbols such as ^ or _ (the equation field holds the math).
- groups: when a sub-structure runs N times in parallel or in sequence (e.g. one attention head), describe ONE copy in a group with repeat = N and a label of 1–3 words such as "Attention head".
- edges: forward data flow; "residual" for skip connections. Acyclic.
- equation, hyperparameters and source_quote follow the same rules as <graph>: source_quote is one sentence copied verbatim from the paper text; reuse the block's own sentence if no better one exists.
- paper_title as in <graph>; model_name: "${block.label}"; summary: one sentence on what happens inside; hyperparameters: the ones that matter inside this block.
If the paper says nothing more about its inside, give the standard structure the paper's notation implies and say so in the summary.`;
}

/**
 * One output format for every kind, inside views included. The format sits ahead of the whole
 * prompt in the cache, so a per-kind schema made every feature re-pay for the paper (measured:
 * cache read 0). The task text says which shape to write; the caller checks it against that schema.
 */
export const ANY_ARTIFACT = z.object({
  artifact: z.union([ARTIFACT_SCHEMA.pytorch, ARTIFACT_SCHEMA.walkthrough, ARTIFACT_SCHEMA.example, Architecture]),
});

export function paperPrefix(pages: string[], graph: Architecture): Anthropic.Beta.BetaTextBlockParam[] {
  return [
    { type: "text", text: `<paper>\n${formatPaper(pages)}\n</paper>` },
    { type: "text", text: `<graph>\n${JSON.stringify(graph)}\n</graph>`, cache_control: { type: "ephemeral" } },
  ];
}

export function learnRequest(kind: LearnKind, pages: string[], graph: Architecture) {
  return tutorRequest(pages, graph, `${LEARN_TASK[kind]}\n\nWrite it as the "${kind}" shape of artifact.`);
}

export function insideRequest(block: Block, pages: string[], graph: Architecture) {
  return tutorRequest(pages, graph, `${insideTask(block)}\n\nWrite it as the block-graph shape of artifact.`);
}

function tutorRequest(pages: string[], graph: Architecture, task: string) {
  return {
    model: MODEL,
    max_tokens: 32000,
    system: TUTOR_SYSTEM,
    output_config: { effort: EFFORT, format: betaZodOutputFormat(ANY_ARTIFACT) },
    ...fallback(),
    messages: [
      {
        role: "user" as const,
        content: [...paperPrefix(pages, graph), { type: "text" as const, text: task }],
      },
    ],
  };
}

export interface AskInput {
  pages: string[];
  graph: Architecture;
  block: Block;
  question: string;
  history: AskTurn[];
  /** The reader's language and level; after the cache breakpoint, so every reader shares the prefix. */
  lang?: Lang;
  level?: Level;
}

export function askRequest({ pages, graph, block, question, history, lang = "en", level = DEFAULT_LEVEL }: AskInput) {
  const focus = `The reader is asking about this block:\n${JSON.stringify({
    id: block.id,
    label: block.label,
    type: block.type,
    equation: block.equation,
    input_shape: block.input_shape,
    output_shape: block.output_shape,
  })}\nAnswer in a few short paragraphs at most. Plain text; LaTeX only inline as $...$.${
    lang === "en" ? "" : ` Answer in ${languageName(lang)}, whatever language the question is in; refer to blocks by their ids as usual.`
  }${level === DEFAULT_LEVEL ? "" : ` Write for ${levelReader(level)}.`}`;
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
  const parsed = ARTIFACT_SCHEMA[kind].safeParse(response.parsed_output?.artifact);
  if (!parsed.success) throw new Error("Claude returned something that didn't match the expected format.");
  return parsed.data as AnyArtifact;
}

/** Keep an inside view drawable: unique block ids, edges between blocks that exist, no self-loops. */
export function cleanInside(inside: Architecture): Architecture {
  const seen = new Set<string>();
  const blocks = inside.blocks.filter((b) => !seen.has(b.id) && seen.add(b.id));
  const groups = inside.groups.filter((g) => blocks.some((b) => b.group === g.id));
  const edges = inside.edges.filter((e) => seen.has(e.from) && seen.has(e.to) && e.from !== e.to);
  return { ...inside, groups, blocks: blocks.map((b) => (groups.some((g) => g.id === b.group) ? b : { ...b, group: "" })), edges };
}

export async function generateInside(
  block: Block,
  { pages, graph, apiKey }: { pages: string[]; graph: Architecture; apiKey?: string },
): Promise<Architecture> {
  const response = await client(apiKey).beta.messages.parse(insideRequest(block, pages, graph));
  logUsage(`inside ${block.id}`, response.usage);
  if (response.stop_reason === "refusal") {
    throw new Error(`Claude declined to write this (${response.stop_details?.category ?? "unspecified"}).`);
  }
  if (response.stop_reason === "max_tokens") throw new Error("Claude ran out of output tokens before finishing.");
  const parsed = Architecture.safeParse(response.parsed_output?.artifact);
  if (!parsed.success) throw new Error("Claude returned something that didn't match the expected format.");
  const inside = cleanInside(parsed.data);
  if (inside.blocks.length < 2) throw new Error(`Claude found nothing inside ${block.label} to show.`);
  return inside;
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
