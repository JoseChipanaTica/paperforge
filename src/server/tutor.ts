import Anthropic from "@anthropic-ai/sdk";
import { betaZodOutputFormat } from "@anthropic-ai/sdk/helpers/beta/zod";
import { z } from "zod";
import { Architecture, type Block } from "../shared/graph";
import { ARTIFACT_SCHEMA, type AnyArtifact, type AskTurn, type InsideLearnKind, type JourneyArtifact, type LearnKind, type View } from "../shared/learn";
import { MotionArtifact } from "../shared/motion";
import { HIGHLIGHTER, INK, SHEET, VERMILION } from "../shared/palette";
import { DEFAULT_LEVEL, languageName, levelReader, type Lang, type Level } from "../shared/translate";
import { formatPaper } from "./extract";
import { currentModel, TIMEOUT_MS } from "./model";

/**
 * The tutor: Claude calls behind the learning features. Every request starts with the same
 * system prompt and the same [paper, graph] prefix, so after the first feature on a paper
 * the rest read it from the prompt cache. Keep everything before the breakpoint identical
 * across kinds, and keep effort fixed (an effort change invalidates the cache).
 */

const EFFORT = "medium" as const;
// On a safety decline, let the API route to a fallback model inside the same call (as extract.ts).
const fallback = () => ({ betas: ["server-side-fallback-2026-07-01"], fallbacks: "default" as const });

export const TUTOR_SYSTEM = `You are Paperforge's tutor. A reader has just watched a deep learning paper's architecture assemble itself as a 3D model, block by block, and wants to understand it.

You get the paper's text and the model's block graph (<graph>, JSON). Ground everything in the paper: use its notation, hyperparameters and terminology, and say so when you go beyond what it states.

Whenever you refer to a block, use its exact "id" from <graph>; the viewer uses those ids to light up the matching block. Never invent ids.

Write for a curious engineer: plain language first, then the precise detail. Be concrete and brief.`;

/** The kinds written through the shared learning format (motion has its own, and needs the journey). */
export type TaskKind = Exclude<LearnKind, "motion" | "training">;

export const LEARN_TASK: Record<TaskKind, string> = {
  pytorch: `Write PyTorch code that implements this architecture.
- One nn.Module per distinct kind of block (reuse a module for repeated blocks), using the paper's hyperparameters as defaults.
- Comment tensor shapes on the important lines.
- End with an if __name__ == "__main__": block that builds the model, runs one forward pass on random input and prints the output shape.
- block_map: one entry for every block in <graph>, naming the class, function or variable in your code that implements it.
- notes: every assumption you made where the paper is silent or ambiguous.`,
  journey: `Follow one tiny, concrete input through this architecture, block by block, from the raw input to the final output, so a reader can watch the data change shape and meaning.
- modality: what the model takes in.
- Pick one small input that suits the paper's task and modality (3–5 tokens of text, a 4×4 image patch grid, a short audio clip, a few video frames; for multimodal, one of each), and follow that same input through every block.
- analogy: one vivid everyday analogy for the whole model. intro: what the reader is about to watch.
- input: the raw input in words (label) and as views. output: the final result in words and as views (usually "probs" for a prediction).
- steps: one per block in <graph>, in the order data flows (finish one branch before the next). title; what_happens in one plain sentence; narration in 2–3 sentences on what happens to the data here and why the paper does it this way; shape_change as "[B, n, 512] → [B, n, 512]" consistent with the block's shapes; views showing the data after the block.
- views: pick the kind that shows what this block does: "tokens" for discrete symbols, "vectors" for embeddings and hidden states, "heads" when a vector is split into h heads (split = h), "matrix" for attention weights or any item × item map, "patches" for images, "signal" for audio, "frames" for video, "probs" for output probabilities, "tensor" only when nothing else fits. Use two views in one step only when two streams coexist (e.g. encoder output beside decoder state, or text beside image).
- Values are illustrative but consistent: an item keeps its row label from step to step, and each step's numbers plausibly follow from the previous step's. At most 12 rows and 12 columns per view and at most 8 numbers per vector row; dims gives the true tensor size (e.g. "[3, 512]"), so the view is a slice of it.
- outro: one sentence to remember.`,
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
 * One output format for every learning kind, and one for inside views. The format sits ahead of the
 * whole prompt in the cache, so a per-kind schema made every feature re-pay for the paper (measured:
 * cache read 0). The task text says which shape to write; the caller checks it against that schema.
 * The journey and the block graph together compile to a grammar the API rejects as too large, so
 * inside views (many per paper) share the cache with each other, and PyTorch with the journey.
 */
export const ANY_ARTIFACT = z.object({
  artifact: z.union([ARTIFACT_SCHEMA.pytorch, ARTIFACT_SCHEMA.journey]),
});
export const INSIDE_ARTIFACT = z.object({ artifact: Architecture });
/** Motion's ops would push either shared format over the grammar limit: it gets its own. */
export const MOTION_ARTIFACT = z.object({ artifact: MotionArtifact });

export function paperPrefix(pages: string[], graph: Architecture): Anthropic.Beta.BetaTextBlockParam[] {
  return [
    { type: "text", text: `<paper>\n${formatPaper(pages)}\n</paper>` },
    { type: "text", text: `<graph>\n${JSON.stringify(graph)}\n</graph>`, cache_control: { type: "ephemeral" } },
  ];
}

export function learnRequest(kind: TaskKind, pages: string[], graph: Architecture) {
  return tutorRequest(pages, graph, `${LEARN_TASK[kind]}\n\nWrite it as the "${kind}" shape of artifact.`, ANY_ARTIFACT);
}

/** What each block does to the walkthrough's input, as a chain of ops the browser animates. */
export function motionTask(journey: JourneyArtifact): string {
  return `Animate what each block does to the walkthrough's toy input (given in <journey>), so a reader watches the block do its job: a kernel sliding over an image, attention scoring words against each other and mixing them, and so on.
- steps: one entry per step of <journey>, with the same block_id, in the same order.
- ops: 1–3 operations, played in order, that show what the paper's block computes; each op's output is the next op's input. Examples: feed-forward = linear → activation (relu) → linear; add & norm = residual_add → normalization (layer); multi-head attention = split_heads → attention → concat; a VGG conv layer = convolution → activation (relu).
- The browser recomputes every number, so give only small inputs, never results: input (the data entering the first op, ≤ 8×8, a small slice of that step's input in <journey>, with tokens naming its rows the same way), weights and extra as each op's field description says. Keep numbers small and readable (one decimal). Every matrix at most 8×8 (attention's extra: K rows then V rows, up to 8 each). Make shapes line up: linear weights are in × out, attention Q, K and V rows share a width.
- size, stride, count: as the op needs (a VGG conv3-64: size 3, stride 1, count 64); otherwise 1.
- caption: one short sentence on what this op does here, in the paper's terms.
- Use "custom" only when no op fits. Then write svg: an animated SVG, viewBox "0 0 360 220", SMIL animation only (animate, animateTransform, animateMotion, set), no script, no external references or images, under 30 KB. Colours: ink ${INK}, paper ${SHEET}, highlighter ${HIGHLIGHTER} for what's active, vermilion ${VERMILION} for results. Plain text labels.
- Otherwise svg is "".

<journey>
${JSON.stringify(journey)}
</journey>`;
}

/**
 * A walkthrough of a block's inside view: the main walkthrough's input followed through the inside
 * graph, starting from what enters the block and ending at what leaves it (when the main
 * walkthrough is known), so the two walkthroughs line up.
 */
export function insideJourneyTask(block: Block, inside: Architecture, entering: View[] | null, leaving: View[] | null): string {
  const ends =
    entering && leaving
      ? `- Start exactly from <entering> (the data reaching "${block.id}" in the model's main walkthrough) and end matching <leaving> (what leaves it), keeping the same row labels.

<entering>
${JSON.stringify(entering)}
</entering>

<leaving>
${JSON.stringify(leaving)}
</leaving>`
      : `- The model's main walkthrough isn't known: choose one small, concrete input that fits the paper and what enters "${block.id}" (input_shape ${block.input_shape}).`;
  return `Follow one input through the inside of the block "${block.id}" (${block.label}), whose internal structure is <inside> (JSON, same shape as <graph>).
- steps: one per block of <inside>, using <inside>'s ids (not <graph>'s), in data-flow order.
${ends}
${LEARN_TASK.journey}

<inside>
${JSON.stringify(inside)}
</inside>`;
}

export function insideJourneyRequest(
  pages: string[],
  graph: Architecture,
  block: Block,
  inside: Architecture,
  entering: View[] | null,
  leaving: View[] | null,
) {
  return tutorRequest(pages, graph, `${insideJourneyTask(block, inside, entering, leaving)}\n\nWrite it as the "journey" shape of artifact.`, ANY_ARTIFACT);
}

export function motionRequest(pages: string[], graph: Architecture, journey: JourneyArtifact) {
  return tutorRequest(pages, graph, motionTask(journey), MOTION_ARTIFACT);
}

export function insideRequest(block: Block, pages: string[], graph: Architecture) {
  return tutorRequest(pages, graph, insideTask(block), INSIDE_ARTIFACT);
}

function tutorRequest(
  pages: string[],
  graph: Architecture,
  task: string,
  format: typeof ANY_ARTIFACT | typeof INSIDE_ARTIFACT | typeof MOTION_ARTIFACT,
) {
  return {
    model: currentModel(),
    max_tokens: 32000,
    system: TUTOR_SYSTEM,
    output_config: { effort: EFFORT, format: betaZodOutputFormat(format as typeof ANY_ARTIFACT) },
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
    model: currentModel(),
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
  { pages, graph, apiKey, journey }: { pages: string[]; graph: Architecture; apiKey?: string; journey?: JourneyArtifact },
): Promise<AnyArtifact> {
  if (kind === "training") throw new Error("Training needs the walkthrough it trains on.");
  if (kind === "motion" && !journey) throw new Error("Motion needs the walkthrough it animates.");
  const request = kind === "motion" ? motionRequest(pages, graph, journey!) : learnRequest(kind, pages, graph);
  const response = await client(apiKey).beta.messages.parse(request);
  logUsage(kind, response.usage);
  if (response.stop_reason === "refusal") {
    throw new Error(`Claude declined to write this (${response.stop_details?.category ?? "unspecified"}).`);
  }
  if (response.stop_reason === "max_tokens") throw new Error("Claude ran out of output tokens before finishing.");
  const parsed = ARTIFACT_SCHEMA[kind].safeParse(response.parsed_output?.artifact);
  if (!parsed.success) throw new Error("Claude returned something that didn't match the expected format.");
  return parsed.data as AnyArtifact;
}

/** A block's inside walkthrough (journey) or its motion: the same artifacts, for the inside graph. */
export async function generateInsideLearn(
  kind: InsideLearnKind,
  o: {
    pages: string[];
    graph: Architecture;
    block: Block;
    inside: Architecture;
    entering?: View[] | null;
    leaving?: View[] | null;
    journey?: JourneyArtifact;
    apiKey?: string;
  },
): Promise<AnyArtifact> {
  if (kind === "motion" && !o.journey) throw new Error("Motion needs the walkthrough it animates.");
  const request =
    kind === "motion" ? motionRequest(o.pages, o.graph, o.journey!) : insideJourneyRequest(o.pages, o.graph, o.block, o.inside, o.entering ?? null, o.leaving ?? null);
  const response = await client(o.apiKey).beta.messages.parse(request);
  logUsage(`inside ${o.block.id} ${kind}`, response.usage);
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
