import Anthropic from "@anthropic-ai/sdk";
import { betaZodOutputFormat } from "@anthropic-ai/sdk/helpers/beta/zod";
import {
  TEXT_SCHEMA,
  codeSkeleton,
  languageName,
  levelReader,
  textMatches,
  type Lang,
  type Level,
  type TranslateKind,
  type TranslatedText,
} from "../shared/translate";
import { currentModel, TIMEOUT_MS } from "./model";

/** Claude call that rewrites the prose of a graph or artifact for a language and reader (see shared/translate.ts). */


export const TRANSLATE_SYSTEM = `You translate study material about a deep learning paper for Paperforge, a site that turns papers into 3D models a reader can explore.

You get a JSON object. Return the same object with every human-readable string written in the requested language for the requested reader.
- Keep every "id" and "block_id" exactly as given, and keep every list in the same order and length.
- Keep LaTeX (between $...$), tensor shapes such as [B, n, 512], code identifiers, hyperparameter symbols and numbers unchanged.
- Write naturally. Keep an English term where practitioners in that language normally use it (model names such as "Transformer", "softmax"); translate everything else.
- Block and group labels are printed next to the 3D model: only translate them, and keep them as short as the originals.
- When the reader differs from the original's (a curious engineer), rewrite the explanations for them: summaries, descriptions, narration, analogies, notes and comments. Keep each about as long as the original (the viewer gives it a fixed space), and add no facts the original doesn't support.`;

const PYTORCH_NOTE = `In "code", change only comments and docstrings. Every other character, including string literals, must stay exactly as it is.`;

export function translateRequest(kind: TranslateKind, text: TranslatedText, lang: Lang, level: Level) {
  const task = `Write this in ${languageName(lang)}, for ${levelReader(level)}.${kind === "pytorch" ? ` ${PYTORCH_NOTE}` : ""}`;
  return {
    model: currentModel(),
    max_tokens: 32000,
    system: TRANSLATE_SYSTEM,
    output_config: { effort: "low" as const, format: betaZodOutputFormat(TEXT_SCHEMA[kind] as (typeof TEXT_SCHEMA)["graph"]) },
    // On a safety decline, let the API route to a fallback model inside the same call (as extract.ts).
    betas: ["server-side-fallback-2026-07-01"],
    fallbacks: "default" as const,
    messages: [{ role: "user" as const, content: `${task}\n\n<text>\n${JSON.stringify(text)}\n</text>` }],
  };
}

export async function translateText(
  kind: TranslateKind,
  text: TranslatedText,
  lang: Lang,
  level: Level,
  apiKey?: string,
): Promise<TranslatedText> {
  const client = new Anthropic({ apiKey: apiKey?.trim() || undefined, timeout: TIMEOUT_MS });
  const response = await client.beta.messages.parse(translateRequest(kind, text, lang, level));
  if (response.stop_reason === "refusal") {
    throw new Error(`Claude declined to translate this (${response.stop_details?.category ?? "unspecified"}).`);
  }
  if (response.stop_reason === "max_tokens") throw new Error("Claude ran out of output tokens before finishing the translation.");
  const parsed = TEXT_SCHEMA[kind].safeParse(response.parsed_output);
  if (!parsed.success || !textMatches(kind, text, parsed.data)) {
    throw new Error("The translation didn't line up with the original. Try again.");
  }
  const out = parsed.data;
  // Never let a rewrite change what the code does: on any doubt, keep the original comments.
  if (kind === "pytorch" && "code" in out && "code" in text && codeSkeleton(out.code) !== codeSkeleton(text.code)) {
    return { ...out, code: text.code };
  }
  return out;
}
