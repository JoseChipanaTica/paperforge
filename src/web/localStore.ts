import { z } from "zod";
import { Architecture, type LocatedArchitecture } from "../shared/graph";
import { ARTIFACT_SCHEMA, type AnyArtifact } from "../shared/learn";
import { type Lang, type Level, TEXT_SCHEMA, type TranslateKind, type TranslatedText } from "../shared/translate";
import type { Backend, StoreName } from "./storeBackend";
import { nullBackend } from "./storeBackend";

/**
 * The browser store's rules: papers by arXiv id, artifacts by [graph hash, scope, kind], translations by
 * [text hash, language, level]. Every read is checked against the same Zod schemas as server responses (an entry
 * that fails is deleted and missed) but returns the stored value itself, so located graphs keep each block's
 * `source`. Backend failures never surface: reads miss, writes are dropped.
 */

/** JSON with object keys sorted at every depth. */
function canonical(value: unknown): string {
  return JSON.stringify(value, (_, v) =>
    v && typeof v === "object" && !Array.isArray(v)
      ? Object.fromEntries(Object.keys(v).sort().map((k) => [k, (v as Record<string, unknown>)[k]]))
      : v,
  );
}

async function sha256(text: string): Promise<string> {
  const digest = await crypto.subtle.digest("SHA-256", new TextEncoder().encode(text));
  return Array.from(new Uint8Array(digest), (b) => b.toString(16).padStart(2, "0")).join("");
}

const byteLength = (value: unknown) => new TextEncoder().encode(JSON.stringify(value)).length;

/** The graph's identity: its English content without where the quotes were found (nor the arXiv id). */
export function graphHash(g: Architecture | LocatedArchitecture): Promise<string> {
  const { arxiv_id: _id, blocks, ...rest } = g as LocatedArchitecture;
  const bare = blocks.map((b) => {
    const { source: _, ...block } = b as typeof b & { source?: unknown };
    return block;
  });
  return sha256(canonical({ ...rest, blocks: bare }));
}

export function textHash(kind: TranslateKind, text: TranslatedText): Promise<string> {
  return sha256(kind + canonical(text));
}

export type StoredPaper = {
  arxiv_id: string;
  pages: string[];
  graph: LocatedArchitecture;
  graph_hash: string;
  save_token?: string | null;
  source: "gallery" | "sample" | "cache" | "claude";
  opened_at: number;
  bytes: number;
};

/** How a write went: Task 3 reads `name` (e.g. "QuotaExceededError") to evict and retry. */
export type WriteResult = { ok: true } | { ok: false; name: string; error: unknown };

type ArtifactValue = AnyArtifact | Architecture;
type StoredArtifact = { arxiv_id: string; artifact: ArtifactValue; saved_at: number; bytes: number };
type StoredTranslation = { arxiv_ids: string[]; text: TranslatedText; saved_at: number; bytes: number };

const PaperRecord = z.object({
  arxiv_id: z.string(),
  pages: z.array(z.string()),
  graph: Architecture,
  graph_hash: z.string(),
  save_token: z.string().nullish(),
  source: z.enum(["gallery", "sample", "cache", "claude"]),
  opened_at: z.number(),
  bytes: z.number(),
});
const Envelope = z.object({ saved_at: z.number(), bytes: z.number() });
const ArtifactEnvelope = Envelope.extend({ arxiv_id: z.string() });
const TranslationEnvelope = Envelope.extend({ arxiv_ids: z.array(z.string()) });

function artifactSchema(kind: string): z.ZodType | null {
  if (kind === "inside") return Architecture;
  return Object.hasOwn(ARTIFACT_SCHEMA, kind) ? ARTIFACT_SCHEMA[kind as keyof typeof ARTIFACT_SCHEMA] : null;
}

export class LocalStore {
  readonly available: boolean;

  constructor(
    private backend: Backend,
    private now: () => number = () => Date.now(),
  ) {
    this.available = backend !== nullBackend;
  }

  // ---- safe backend access: a failure is a miss or a no-op ----

  private async read(store: StoreName, key: IDBValidKey): Promise<unknown> {
    try {
      return await this.backend.get(store, key);
    } catch {
      return undefined;
    }
  }

  /** Never throws; a failure comes back with its DOMException name ("Error" when it has none). */
  async write(store: StoreName, key: IDBValidKey, value: unknown): Promise<WriteResult> {
    try {
      await this.backend.put(store, key, value);
      return { ok: true };
    } catch (error) {
      const name = (error as { name?: unknown } | null)?.name;
      return { ok: false, name: typeof name === "string" ? name : "Error", error };
    }
  }

  private async drop(store: StoreName, key: IDBValidKey): Promise<void> {
    try {
      await this.backend.delete(store, key);
    } catch {}
  }

  private async all(store: StoreName): Promise<[IDBValidKey, unknown][]> {
    try {
      return (await this.backend.entries(store)) ?? [];
    } catch {
      return [];
    }
  }

  /** Read and validate; an entry that no longer parses is deleted. Returns the stored value as is. */
  private async readValid(store: StoreName, key: IDBValidKey, valid: (v: unknown) => boolean): Promise<unknown> {
    const value = await this.read(store, key);
    if (value === undefined || value === null) return null;
    if (valid(value)) return value;
    await this.drop(store, key);
    return null;
  }

  // ---- papers ----

  async getPaper(id: string): Promise<StoredPaper | null> {
    const paper = (await this.readValid("papers", id, (v) => PaperRecord.safeParse(v).success)) as StoredPaper | null;
    if (!paper) return null;
    const opened = { ...paper, opened_at: this.now() };
    await this.write("papers", id, opened);
    return opened;
  }

  async putPaper(p: Omit<StoredPaper, "opened_at" | "bytes">): Promise<void> {
    await this.write("papers", p.arxiv_id, { ...p, opened_at: this.now(), bytes: byteLength(p) });
  }

  // ---- artifacts (learning artifacts and inside graphs) ----

  async getArtifact(hash: string, scope: string, kind: string): Promise<AnyArtifact | Architecture | null> {
    const schema = artifactSchema(kind);
    if (!schema) return null;
    const entry = (await this.readValid(
      "artifacts",
      [hash, scope, kind],
      (v) => ArtifactEnvelope.safeParse(v).success && schema.safeParse((v as StoredArtifact).artifact).success,
    )) as StoredArtifact | null;
    return entry?.artifact ?? null;
  }

  async putArtifact(id: string, hash: string, scope: string, kind: string, value: ArtifactValue): Promise<void> {
    const entry: StoredArtifact = { arxiv_id: id, artifact: value, saved_at: this.now(), bytes: byteLength(value) };
    await this.write("artifacts", [hash, scope, kind], entry);
  }

  // ---- translations ----

  async getTranslation(textHash: string, lang: Lang, level: Level, kind: TranslateKind): Promise<TranslatedText | null> {
    const schema = TEXT_SCHEMA[kind];
    if (!schema) return null;
    const entry = (await this.readValid(
      "translations",
      [textHash, lang, level],
      (v) => TranslationEnvelope.safeParse(v).success && schema.safeParse((v as StoredTranslation).text).success,
    )) as StoredTranslation | null;
    return entry?.text ?? null;
  }

  async putTranslation(
    id: string,
    textHash: string,
    lang: Lang,
    level: Level,
    kind: TranslateKind,
    text: TranslatedText,
  ): Promise<void> {
    const key = [textHash, lang, level];
    const prior = (await this.read("translations", key)) as Partial<StoredTranslation> | undefined;
    const ids = Array.isArray(prior?.arxiv_ids) ? prior.arxiv_ids.filter((x) => typeof x === "string") : [];
    if (!ids.includes(id)) ids.push(id);
    const entry: StoredTranslation = { arxiv_ids: ids, text, saved_at: this.now(), bytes: byteLength(text) };
    await this.write("translations", key, entry);
  }

  // ---- the whole store ----

  async usage(): Promise<{ papers: number; bytes: number }> {
    const [papers, artifacts, translations] = await Promise.all([
      this.all("papers"),
      this.all("artifacts"),
      this.all("translations"),
    ]);
    const size = (v: unknown) => {
      const b = (v as { bytes?: unknown } | null)?.bytes;
      return typeof b === "number" && Number.isFinite(b) ? b : 0;
    };
    const bytes = [...papers, ...artifacts, ...translations].reduce((sum, [, v]) => sum + size(v), 0);
    return { papers: papers.length, bytes };
  }

  async clear(): Promise<void> {
    try {
      await this.backend.clear();
    } catch {}
  }
}
