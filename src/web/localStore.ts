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

/** Shared wins: a gallery or sample graph replaces the stored one only when its hash differs. */
export function sharedDecision(
  local: StoredPaper,
  shared: { graph: Architecture; source: "gallery" | "sample" } | null,
  sharedHash: string | null,
): "keep" | "replace" {
  return shared && sharedHash !== null && sharedHash !== local.graph_hash ? "replace" : "keep";
}

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
  private capBytes: number;
  private floorBytes: number;
  private open: string | null = null;

  /** The paper on screen is never evicted. */
  setOpen(id: string | null): void {
    this.open = id;
  }

  /** A put that retries once after evicting when the browser says the quota is full. */
  private async put(store: StoreName, key: IDBValidKey, value: unknown, owner: string): Promise<void> {
    const r = await this.write(store, key, value);
    if (r.ok) return void (await this.evict(this.capBytes, owner));
    if (r.name !== "QuotaExceededError") return;
    await this.evict(0, owner);
    await this.write(store, key, value);
  }

  /** When the total exceeds `trigger`, delete whole papers (least recently opened first) down to the floor. */
  private async evict(trigger: number, owner: string): Promise<void> {
    try {
      const size = (v: unknown) => {
        const b = (v as { bytes?: unknown } | null)?.bytes;
        return typeof b === "number" && Number.isFinite(b) ? b : 0;
      };
      const [papers, artifacts, translations] = await Promise.all([
        this.all("papers"),
        this.all("artifacts"),
        this.all("translations"),
      ]);
      let total = 0;
      for (const [, v] of [...papers, ...artifacts, ...translations]) total += size(v);
      if (total <= trigger) return;
      // Orphans first: artifacts and translations whose papers are gone count but nothing would ever free them.
      const ids = new Set<string>([...papers.map(([k]) => String(k)), owner]);
      if (this.open) ids.add(this.open);
      for (const [k, v] of artifacts) {
        const id = (v as { arxiv_id?: unknown } | null)?.arxiv_id;
        if (typeof id === "string" && ids.has(id)) continue;
        await this.drop("artifacts", k);
        total -= size(v);
      }
      for (const [k, v] of translations) {
        const t = v as Partial<StoredTranslation> | null;
        const have = Array.isArray(t?.arxiv_ids) ? t.arxiv_ids : [];
        const kept = have.filter((x) => ids.has(x));
        if (kept.length === have.length && have.length > 0) continue;
        if (kept.length === 0) {
          await this.drop("translations", k);
          total -= size(v);
        } else {
          t!.arxiv_ids = kept;
          await this.write("translations", k, t);
        }
      }
      const order = papers
        .filter(([k]) => k !== this.open && k !== owner)
        .sort((a, b) => ((a[1] as { opened_at?: number })?.opened_at ?? 0) - ((b[1] as { opened_at?: number })?.opened_at ?? 0));
      for (const [key, value] of order) {
        if (total <= this.floorBytes) break;
        const id = String(key);
        await this.drop("papers", key);
        total -= size(value);
        for (const [k, v] of artifacts) {
          if ((v as { arxiv_id?: unknown } | null)?.arxiv_id !== id) continue;
          await this.drop("artifacts", k);
          total -= size(v);
        }
        for (const [k, v] of translations) {
          const t = v as Partial<StoredTranslation> | null;
          if (!Array.isArray(t?.arxiv_ids) || !t.arxiv_ids.includes(id)) continue;
          const rest = t.arxiv_ids.filter((x) => x !== id);
          if (rest.length === 0) {
            await this.drop("translations", k);
            total -= size(v);
          } else {
            t.arxiv_ids = rest;
            await this.write("translations", k, t);
          }
        }
      }
    } catch {}
  }

  constructor(
    private backend: Backend,
    private now: () => number = () => Date.now(),
    opts: { capBytes?: number; floorBytes?: number } = {},
  ) {
    this.capBytes = opts.capBytes ?? 100_000_000;
    this.floorBytes = opts.floorBytes ?? 80_000_000;
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
    await this.put("papers", p.arxiv_id, { ...p, opened_at: this.now(), bytes: byteLength(p) }, p.arxiv_id);
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
    await this.put("artifacts", [hash, scope, kind], entry, id);
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
    await this.put("translations", key, entry, id);
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
