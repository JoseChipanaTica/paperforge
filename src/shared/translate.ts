import { z } from "zod";
import type { Architecture } from "./graph";
import { LEARN_KINDS, type JourneyArtifact, type PytorchArtifact } from "./learn";
import type { MotionArtifact } from "./motion";

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
// A journey's data (rows, values, dims, shape changes) stays as is: only the words around it change.
const MomentText = z.object({ label: z.string(), captions: z.array(z.string()) });
const JourneyText = z.object({
  analogy: z.object({ title: z.string(), text: z.string() }),
  intro: z.string(),
  input: MomentText,
  steps: z.array(
    z.object({ block_id: z.string(), title: z.string(), what_happens: z.string(), narration: z.string(), captions: z.array(z.string()) }),
  ),
  output: MomentText,
  outro: z.string(),
});
// Motion: only each op's caption; every number and any custom SVG stay as they are.
const MotionText = z.object({ steps: z.array(z.object({ block_id: z.string(), captions: z.array(z.string()) })) });
const PytorchText = z.object({
  code: z.string().describe("The code with only comments and docstrings translated"),
  notes: z.array(z.string()),
});

export const TEXT_SCHEMA = {
  graph: GraphText,
  expansion: GraphText,
  journey: JourneyText,
  motion: MotionText,
  pytorch: PytorchText,
} as const;

export type TranslatedText = z.infer<(typeof TEXT_SCHEMA)[TranslateKind]>;
type Content = Architecture | PytorchArtifact | JourneyArtifact | MotionArtifact;

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
    case "journey": {
      const a = content as JourneyArtifact;
      const moment = (m: JourneyArtifact["input"]) => ({ label: m.label, captions: m.views.map((v) => v.caption) });
      return {
        analogy: a.analogy,
        intro: a.intro,
        input: moment(a.input),
        steps: a.steps.map((s) => ({
          block_id: s.block_id,
          title: s.title,
          what_happens: s.what_happens,
          narration: s.narration,
          captions: s.views.map((v) => v.caption),
        })),
        output: moment(a.output),
        outro: a.outro,
      };
    }
    case "motion": {
      const a = content as MotionArtifact;
      return { steps: a.steps.map((s) => ({ block_id: s.block_id, captions: s.ops.map((o) => o.caption) })) };
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
  if (kind === "motion") {
    const [o, t] = [original as z.infer<typeof MotionText>, translated as z.infer<typeof MotionText>];
    const counts = (m: z.infer<typeof MotionText>) => m.steps.map((s) => s.captions.length).join(",");
    return ids(o.steps) === ids(t.steps) && counts(o) === counts(t);
  }
  if (kind === "pytorch") return (translated as z.infer<typeof PytorchText>).notes.length === (original as z.infer<typeof PytorchText>).notes.length;
  const [o, t] = [original as z.infer<typeof JourneyText>, translated as z.infer<typeof JourneyText>];
  const counts = (j: z.infer<typeof JourneyText>) =>
    [j.input, ...j.steps, j.output].map((m) => m.captions.length).join(",");
  return ids(o.steps) === ids(t.steps) && counts(o) === counts(t);
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
    case "journey": {
      const a = content as JourneyArtifact;
      const t = text as z.infer<typeof JourneyText>;
      const captioned = (views: JourneyArtifact["input"]["views"], captions: string[] | undefined) =>
        views.map((v, i) => ({ ...v, caption: captions?.[i] ?? v.caption }));
      return {
        ...a,
        analogy: t.analogy,
        intro: t.intro,
        input: { label: t.input.label, views: captioned(a.input.views, t.input.captions) },
        steps: a.steps.map((s, i) => {
          const ts = t.steps[i];
          return ts
            ? { ...s, title: ts.title, what_happens: ts.what_happens, narration: ts.narration, views: captioned(s.views, ts.captions) }
            : s;
        }),
        output: { label: t.output.label, views: captioned(a.output.views, t.output.captions) },
        outro: t.outro,
      } as C;
    }
    case "motion": {
      const a = content as MotionArtifact;
      const t = text as z.infer<typeof MotionText>;
      return {
        ...a,
        steps: a.steps.map((s, i) => ({ ...s, ops: s.ops.map((o, j) => ({ ...o, caption: t.steps[i]?.captions[j] ?? o.caption })) })),
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
