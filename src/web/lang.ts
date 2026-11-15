import {
  DEFAULT_LEVEL,
  LANGUAGES,
  LEVELS,
  applyText,
  isOriginal,
  type Lang,
  type Level,
  type TranslateKind,
  type TranslateResponse,
} from "../shared/translate";
import { LearnError } from "./learn";

/** How the visitor reads: language and level pickers, where they're remembered, and the /api/translate client. */

export interface Reading {
  lang: Lang;
  level: Level;
}

const STORAGE = { lang: "paperforge.lang", level: "paperforge.level" } as const;

function stored<T extends string>(key: string, allowed: readonly { code: T }[], fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return allowed.find((x) => x.code === v)?.code ?? fallback;
  } catch {
    return fallback;
  }
}

function remember(key: string, value: string, fallback: string) {
  try {
    if (value === fallback) localStorage.removeItem(key);
    else localStorage.setItem(key, value);
  } catch {}
}

export const storedReading = (): Reading => ({
  lang: stored(STORAGE.lang, LANGUAGES, "en"),
  level: stored(STORAGE.level, LEVELS, DEFAULT_LEVEL),
});

const nativeName = (lang: Lang) => LANGUAGES.find((l) => l.code === lang)!.name;

/** What the status line says while a rewrite is on its way. */
export function readingMessage({ lang, level }: Reading): string {
  if (level === DEFAULT_LEVEL) return `Translating into ${nativeName(lang)}…`;
  const reader = level === "beginner" ? "beginners" : "researchers";
  return lang === "en" ? `Rewriting for ${reader}…` : `Rewriting for ${reader}, in ${nativeName(lang)}…`;
}

/** The language and "explain for" pickers in the top bar. */
export function initReadingPickers(lang: HTMLSelectElement, level: HTMLSelectElement, onChange: (r: Reading) => void) {
  lang.innerHTML = LANGUAGES.map((l) => `<option value="${l.code}">${l.name}</option>`).join("");
  level.innerHTML = LEVELS.map((l) => `<option value="${l.code}">${l.name}</option>`).join("");
  const r = storedReading();
  lang.value = r.lang;
  level.value = r.level;
  const changed = () => {
    remember(STORAGE.lang, lang.value, "en");
    remember(STORAGE.level, level.value, DEFAULT_LEVEL);
    onChange({ lang: lang.value as Lang, level: level.value as Level });
  };
  lang.addEventListener("change", changed);
  level.addEventListener("change", changed);
}

/** The content with its prose rewritten for this reader (English for an engineer comes back as is). */
export async function translate<C extends object>(
  id: string,
  kind: TranslateKind,
  { lang, level }: Reading,
  content: C,
  key: string,
  blockId?: string,
): Promise<C> {
  if (isOriginal(lang, level)) return content;
  const res = await fetch("/api/translate", {
    method: "POST",
    headers: { "Content-Type": "application/json", ...(key && { "x-anthropic-key": key }) },
    body: JSON.stringify({ arxiv_id: id, kind, lang, level, content, ...(blockId && { block_id: blockId }) }),
  }).catch(() => null);
  if (!res) throw new LearnError("Can't reach the Paperforge server.");
  const body = (await res.json().catch(() => ({}))) as Partial<TranslateResponse> & { error?: string; code?: string };
  if (!res.ok || !body.text) throw new LearnError(body.error ?? "Couldn't translate that.", body.code);
  return applyText(kind, content as never, body.text) as C;
}
