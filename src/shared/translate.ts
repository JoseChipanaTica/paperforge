import { z } from "zod";
import type { Architecture } from "./graph";
import { LEARN_KINDS, type ExampleArtifact, type PytorchArtifact, type WalkthroughArtifact } from "./learn";

/**
 * Language and reading level: Claude writes everything in English for an engineer (what the
 * gallery stores and the tutor's cache is built on); a "translation" rewrites only the prose for
 * another language and/or reader. Ids, equations, shapes, hyperparameters and verbatim source
 * quotes are never sent, so nothing that links text to the model can drift.
 */

export const LANGUAGES = [
  { code: "en", name: "English", english: "English" },
  { code: "es", name: "Español", english: "Spanish" },
  { code: "pt", name: "Português", english: "Portuguese" },
  { code: "fr", name: "Français", english: "French" },
  { code: "de", name: "Deutsch", english: "German" },
  { code: "it", name: "Italiano", english: "Italian" },
  { code: "ru", name: "Русский", english: "Russian" },
  { code: "hi", name: "हिन्दी", english: "Hindi" },
  { code: "zh", name: "中文", english: "Simplified Chinese" },
  { code: "ja", name: "日本語", english: "Japanese" },
  { code: "ko", name: "한국어", english: "Korean" },
] as const;

export type Lang = (typeof LANGUAGES)[number]["code"];
export const Lang = z.enum(LANGUAGES.map((l) => l.code) as [Lang, ...Lang[]]);
export const languageName = (lang: Lang) => LANGUAGES.find((l) => l.code === lang)!.english;

export const LEVELS = [
  {
    code: "beginner",
    name: "Beginner",
    reader: "a curious beginner with no machine learning background: everyday words, every term explained in passing, short sentences",
  },
  { code: "engineer", name: "Engineer", reader: "a curious engineer: plain language first, then the precise detail" },
  {
    code: "researcher",
    name: "Researcher",
    reader: "a machine learning researcher: precise, in the paper's notation and terminology, with the design rationale and trade-offs the paper gives",
  },
] as const;

export type Level = (typeof LEVELS)[number]["code"];
export const Level = z.enum(LEVELS.map((l) => l.code) as [Level, ...Level[]]);
/** Claude's own output is English for an engineer: that combination needs no rewrite. */
export const DEFAULT_LEVEL: Level = "engineer";
export const isOriginal = (lang: Lang, level: Level) => lang === "en" && level === DEFAULT_LEVEL;
export const levelReader = (level: Level) => LEVELS.find((l) => l.code === level)!.reader;

/** "expansion" is a block's inside view: a graph of its own, keyed by the block it opens. */
export const TRANSLATE_KINDS = ["graph", "expansion", ...LEARN_KINDS] as const;
export const TranslateKind = z.enum(TRANSLATE_KINDS);
export type TranslateKind = z.infer<typeof TranslateKind>;

// What gets translated for each kind. Ids ride along so Claude (and the checks) can line up the items.
const GraphText = z.object({
  summary: z.string(),
  groups: z.array(z.object({ id: z.string(), label: z.string() })),
  blocks: z.array(z.object({ id: z.string(), label: z.string(), description: z.string() })),
});
const WalkthroughText = z.object({
  intro: z.string(),
  steps: z.array(z.object({ block_id: z.string(), title: z.string(), narration: z.string() })),
  outro: z.string(),
});
const ExampleText = z.object({
  analogy: z.object({ title: z.string(), text: z.string() }),
  toy_input: z.string(),
  steps: z.array(z.object({ block_id: z.string(), what_happens: z.string(), toy_value: z.string() })),
  takeaway: z.string(),
});
const PytorchText = z.object({
  code: z.string().describe("The code with only comments and docstrings translated"),
  notes: z.array(z.string()),
});

export const TEXT_SCHEMA = {
  graph: GraphText,
  expansion: GraphText,
  walkthrough: WalkthroughText,
  example: ExampleText,
  pytorch: PytorchText,
} as const;

export type TranslatedText = z.infer<(typeof TEXT_SCHEMA)[TranslateKind]>;
type Content = Architecture | PytorchArtifact | WalkthroughArtifact | ExampleArtifact;

/** The prose of a graph or artifact: what is sent for translation and what the cache is keyed on. */
export function pickText(kind: TranslateKind, content: Content): TranslatedText {
  switch (kind) {
    case "graph":
    case "expansion": {
      const g = content as Architecture;
      return {
        summary: g.summary,
        groups: g.groups.map(({ id, label }) => ({ id, label })),
        blocks: g.blocks.map(({ id, label, description }) => ({ id, label, description })),
      };
    }
    case "walkthrough": {
      const a = content as WalkthroughArtifact;
      return { intro: a.intro, steps: a.steps.map(({ block_id, title, narration }) => ({ block_id, title, narration })), outro: a.outro };
    }
    case "example": {
      const a = content as ExampleArtifact;
      return {
        analogy: a.analogy,
        toy_input: a.toy_input,
        steps: a.steps.map(({ block_id, what_happens, toy_value }) => ({ block_id, what_happens, toy_value })),
        takeaway: a.takeaway,
      };
    }
    case "pytorch": {
      const a = content as PytorchArtifact;
      return { code: a.code, notes: a.notes };
    }
  }
}

/** True when a translation lines up item for item with the text it came from. */
export function textMatches(kind: TranslateKind, original: TranslatedText, translated: TranslatedText): boolean {
  const ids = (items: { id?: string; block_id?: string }[]) => items.map((i) => i.id ?? i.block_id).join("\n");
  if (kind === "graph" || kind === "expansion") {
    const [o, t] = [original as z.infer<typeof GraphText>, translated as z.infer<typeof GraphText>];
    return ids(o.groups) === ids(t.groups) && ids(o.blocks) === ids(t.blocks);
  }
  if (kind === "pytorch") return (translated as z.infer<typeof PytorchText>).notes.length === (original as z.infer<typeof PytorchText>).notes.length;
  const [o, t] = [original as { steps: { block_id: string }[] }, translated as { steps: { block_id: string }[] }];
  return ids(o.steps) === ids(t.steps);
}

/** The content with its prose replaced by a translation (from pickText of the same content). */
export function applyText<C extends Content>(kind: TranslateKind, content: C, text: TranslatedText): C {
  switch (kind) {
    case "graph":
    case "expansion": {
      const g = content as Architecture;
      const t = text as z.infer<typeof GraphText>;
      const groups = new Map(t.groups.map((x) => [x.id, x]));
      const blocks = new Map(t.blocks.map((x) => [x.id, x]));
      return {
        ...g,
        summary: t.summary,
        groups: g.groups.map((x) => ({ ...x, label: groups.get(x.id)?.label ?? x.label })),
        blocks: g.blocks.map((x) => {
          const b = blocks.get(x.id);
          return b ? { ...x, label: b.label, description: b.description } : x;
        }),
      } as C;
    }
    case "walkthrough": {
      const a = content as WalkthroughArtifact;
      const t = text as z.infer<typeof WalkthroughText>;
      return {
        ...a,
        intro: t.intro,
        steps: a.steps.map((s, i) => ({ ...s, title: t.steps[i]?.title ?? s.title, narration: t.steps[i]?.narration ?? s.narration })),
        outro: t.outro,
      } as C;
    }
    case "example": {
      const a = content as ExampleArtifact;
      const t = text as z.infer<typeof ExampleText>;
      return {
        ...a,
        analogy: t.analogy,
        toy_input: t.toy_input,
        steps: a.steps.map((s, i) => ({
          ...s,
          what_happens: t.steps[i]?.what_happens ?? s.what_happens,
          toy_value: t.steps[i]?.toy_value ?? s.toy_value,
        })),
        takeaway: t.takeaway,
      } as C;
    }
    case "pytorch": {
      const t = text as z.infer<typeof PytorchText>;
      return { ...(content as PytorchArtifact), code: t.code, notes: t.notes } as C;
    }
  }
}

/**
 * Python with comments and docstrings removed and whitespace collapsed: a translated file must
 * match the original on this, so a translation can never change what the code does.
 */
export function codeSkeleton(code: string): string {
  return code
    .replace(/("""|''')[\s\S]*?\1/g, "")
    .split("\n")
    .map((line) => {
      // Cut at the first # outside a string literal.
      let quote: string | null = null;
      for (let i = 0; i < line.length; i++) {
        const c = line[i]!;
        if (quote) {
          if (c === "\\") i++;
          else if (c === quote) quote = null;
        } else if (c === '"' || c === "'") quote = c;
        else if (c === "#") return line.slice(0, i);
      }
      return line;
    })
    .join(" ")
    .replace(/\s+/g, " ")
    .trim();
}

export type TranslateResponse = { text: TranslatedText; source: "gallery" | "cache" | "claude" };
