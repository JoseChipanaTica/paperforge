import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { ARTIFACT_SCHEMA } from "../shared/learn";
import {
  DEFAULT_LEVEL,
  Lang,
  Level,
  TEXT_SCHEMA,
  isOriginal,
  TranslateKind,
  pickText,
  type TranslateResponse,
  type TranslatedText,
} from "../shared/translate";
import { errorMessage } from "./api";
import { parseArxivId } from "./arxiv";
import { CACHE_DIR } from "./forge";
import { getStore, type Store } from "./store";
import { translateText } from "./translate";

/**
 * POST /api/translate: { arxiv_id, kind, lang, level, content, block_id? } → { text, source }. The
 * visitor sends the English graph or artifact they are looking at; the answer is its prose in `lang`
 * for a `level` reader. Each is made once: gallery → disk cache → Claude. Only text that matches
 * the published (or sample) English goes to the shared gallery, so it never holds anything a
 * visitor made up. An expansion (a block's inside view) also names its block_id.
 */

export interface TranslateDeps {
  store: Store;
  translate: typeof translateText;
  samplesDir: string;
  cacheDir: string;
}

export function defaultTranslateDeps(): TranslateDeps {
  return {
    store: getStore(),
    translate: translateText,
    samplesDir: join(import.meta.dir, "..", "..", "samples"),
    cacheDir: CACHE_DIR,
  };
}

const MAX_BODY = 262_144; // 256 KB, as /api/papers

const fileId = (id: string) => id.replace(/\//g, "_");
const json = (body: unknown, status = 200) => Response.json(body, { status });
const bad = (error: string) => json({ error, code: "bad_input" }, 400);

export const hashText = (kind: TranslateKind, text: TranslatedText) =>
  new Bun.CryptoHasher("sha256").update(`${kind}\n${JSON.stringify(text)}`).digest("hex");

const schemaFor = (kind: TranslateKind) => (kind === "graph" || kind === "expansion" ? Architecture : ARTIFACT_SCHEMA[kind]);

async function readJson(path: string): Promise<unknown> {
  const f = Bun.file(path);
  return (await f.exists()) ? f.json().catch(() => null) : null;
}

/** Is this the English everyone sees for the paper (published or a committed sample)? */
async function isPublic(d: TranslateDeps, id: string, kind: TranslateKind, hash: string, blockId: string): Promise<boolean> {
  // A block's own walkthrough or motion (block_id given) is shared like the model's.
  const ofBlock = blockId !== "" && (kind === "journey" || kind === "motion");
  const stored =
    kind === "graph" ? d.store.getPaper(id)
    : kind === "expansion" ? d.store.getInside(id, blockId)
    : ofBlock ? d.store.getInsideLearn(id, blockId, kind)
    : d.store.getArtifact(id, kind);
  const published = await stored.catch(() => null);
  const file =
    kind === "graph" ? `${fileId(id)}.json`
    : kind === "expansion" ? `${fileId(id)}.inside.${blockId}.json`
    : ofBlock ? `${fileId(id)}.inside.${blockId}.${kind}.json`
    : `${fileId(id)}.${kind}.json`;
  const sample = await readJson(join(d.samplesDir, file)); // blockId is checked to be [\w-] by the handler
  return [published, sample].some((c) => {
    const parsed = c ? schemaFor(kind).safeParse(c) : null;
    return parsed?.success && hashText(kind, pickText(kind, parsed.data)) === hash;
  });
}

export async function translateHandler(req: Request, overrides: Partial<TranslateDeps> = {}): Promise<Response> {
  const d = { ...defaultTranslateDeps(), ...overrides };
  if (Number(req.headers.get("content-length") ?? 0) > MAX_BODY) return json({ error: "Too large" }, 413);
  const raw = await req.text();
  if (raw.length > MAX_BODY) return json({ error: "Too large" }, 413);
  let body: Record<string, unknown> | null;
  try {
    body = JSON.parse(raw);
  } catch {
    return bad("Invalid JSON");
  }
  const id = parseArxivId(String(body?.arxiv_id ?? ""));
  const kind = TranslateKind.safeParse(body?.kind);
  const lang = Lang.safeParse(body?.lang);
  const level = Level.safeParse(body?.level ?? DEFAULT_LEVEL);
  const blockId = typeof body?.block_id === "string" ? body.block_id : "";
  if (!id || !kind.success || !lang.success || !level.success || isOriginal(lang.data, level.data)) {
    return bad("Expected an arXiv id, a kind, and a language or reading level to rewrite for.");
  }
  if (kind.data === "expansion" && !/^[\w-]{1,100}$/.test(blockId)) return bad("An expansion needs its block_id.");
  if (blockId && !/^[\w-]{1,100}$/.test(blockId)) return bad("That block_id isn't valid.");
  const k = kind.data;
  const l = lang.data;
  const lv = level.data;
  const content = schemaFor(k).safeParse(body?.content);
  if (!content.success) return bad("That content doesn't match the schema.");

  const text = pickText(k, content.data);
  const hash = hashText(k, text);
  const reply = (t: TranslatedText, source: TranslateResponse["source"]) => json({ text: t, source } satisfies TranslateResponse);

  const saved = await d.store.getTranslation(hash, l, lv, k).catch((err) => (console.error("translation lookup failed:", err), null));
  if (saved) return reply(saved, "gallery");
  const cachePath = join(d.cacheDir, "translations", `${hash}.${l}.${lv}.json`);
  const cached = TEXT_SCHEMA[k].safeParse(await readJson(cachePath));
  if (cached.success) return reply(cached.data, "cache");

  let translated: TranslatedText;
  try {
    translated = await d.translate(k, text, l, lv, req.headers.get("x-anthropic-key")?.trim() || undefined);
  } catch (err) {
    console.error(err);
    const e = errorMessage(err);
    return json({ error: e.message, code: e.code }, e.code === "needs_key" ? 401 : e.code === "rate_limited" ? 429 : 502);
  }
  await Bun.write(cachePath, JSON.stringify(translated));
  if (await isPublic(d, id, k, hash, blockId)) {
    try {
      // First save wins: if another visitor's landed meanwhile, everyone gets that one.
      const result = await d.store.putTranslation(hash, l, lv, k, id, translated);
      if (result === "exists") translated = (await d.store.getTranslation(hash, l, lv, k)) ?? translated;
    } catch (err) {
      console.error("translation save failed:", err);
    }
  }
  return reply(translated, "claude");
}
