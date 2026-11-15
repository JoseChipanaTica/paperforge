import { SQL } from "bun";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { Architecture, type PaperSummary } from "../shared/graph";
import { ARTIFACT_SCHEMA, LEARN_KINDS, type AnyArtifact, type LearnKind } from "../shared/learn";
import { TEXT_SCHEMA, type Lang, type Level, type TranslateKind, type TranslatedText } from "../shared/translate";

/**
 * The public gallery: one row per arXiv id, first save wins. Postgres (Neon) when
 * DATABASE_URL is set, a local SQLite file otherwise. Queries stay in the subset both
 * dialects share; created_at is ISO-8601 text so cursors compare the same in both.
 */

export interface Store {
  getPaper(id: string): Promise<Architecture | null>;
  listPapers(opts?: { limit?: number; before?: string }): Promise<PaperSummary[]>;
  publishPaper(id: string, arch: Architecture): Promise<"created" | "exists">;
  /** Learning artifacts for published papers; first save wins, like papers. */
  getArtifact(id: string, kind: LearnKind): Promise<AnyArtifact | null>;
  putArtifact(id: string, kind: LearnKind, artifact: AnyArtifact): Promise<"created" | "exists">;
  listArtifactKinds(id: string): Promise<LearnKind[]>;
  /** Translations of published (or sample) text, keyed by a hash of the English; first save wins. */
  getTranslation(hash: string, lang: Lang, level: Level, kind: TranslateKind): Promise<TranslatedText | null>;
  putTranslation(
    hash: string,
    lang: Lang,
    level: Level,
    kind: TranslateKind,
    id: string,
    text: TranslatedText,
  ): Promise<"created" | "exists">;
  /** A block's inside view for a published paper; first save wins, like artifacts. */
  getInside(id: string, blockId: string): Promise<Architecture | null>;
  putInside(id: string, blockId: string, inside: Architecture): Promise<"created" | "exists">;
}

export function createStore(url: string): Store {
  const sqlite = url.startsWith("sqlite");
  let db = new SQL(url);

  let ready: Promise<unknown> | undefined;
  let stale = false;
  // Retry after a failure (e.g. Neon cold start) instead of caching the rejection forever.
  // Bun.SQL clients can keep a connection error, so the retry gets a fresh client too.
  const init = () => {
    if (!ready && stale) {
      void db.close().catch(() => {});
      db = new SQL(url);
      stale = false;
    }
    const json = sqlite ? "text" : "jsonb";
    return (ready ??= (async () => {
      await db.unsafe(`create table if not exists papers (
        arxiv_id text primary key,
        paper_title text not null,
        model_name text not null,
        summary text not null,
        graph ${json} not null,
        block_count integer not null,
        created_at text not null
      )`);
      await db.unsafe(`create table if not exists artifacts (
        arxiv_id text not null,
        kind text not null,
        content ${json} not null,
        created_at text not null,
        primary key (arxiv_id, kind)
      )`);
      await db.unsafe(`create table if not exists translations (
        source_hash text not null,
        lang text not null,
        level text not null,
        kind text not null,
        arxiv_id text not null,
        content ${json} not null,
        created_at text not null,
        primary key (source_hash, lang, level)
      )`);
      await db.unsafe(`create table if not exists insides (
        arxiv_id text not null,
        block_id text not null,
        content ${json} not null,
        created_at text not null,
        primary key (arxiv_id, block_id)
      )`);
    })().catch((err) => {
      ready = undefined;
      stale = true;
      throw err;
    }));
  };

  const fromJson = (v: unknown) => (typeof v === "string" ? JSON.parse(v) : v);
  const toJson = (v: unknown) => {
    const text = JSON.stringify(v);
    return sqlite ? text : db`${text}::jsonb`;
  };

  return {
    async getPaper(id) {
      await init();
      const [row] = await db`select graph from papers where arxiv_id = ${id}`;
      if (!row) return null;
      return Architecture.parse(fromJson(row.graph));
    },

    async listPapers({ limit = 24, before } = {}) {
      await init();
      const n = Math.min(Math.max(Math.trunc(limit) || 24, 1), 50);
      const rows: PaperSummary[] = await db`
        select arxiv_id, paper_title, model_name, summary, block_count, created_at from papers
        ${before ? db`where created_at < ${before}` : db``}
        order by created_at desc limit ${n}`;
      return rows.map((r) => ({ ...r, block_count: Number(r.block_count) }));
    },

    async publishPaper(id, arch) {
      const a = Architecture.parse(arch);
      await init();
      const result = await db`
        insert into papers (arxiv_id, paper_title, model_name, summary, graph, block_count, created_at)
        values (${id}, ${a.paper_title}, ${a.model_name}, ${a.summary},
                ${toJson(a)}, ${a.blocks.length}, ${new Date().toISOString()})
        on conflict (arxiv_id) do nothing`;
      return result.count > 0 ? "created" : "exists";
    },

    async getArtifact(id, kind) {
      await init();
      const [row] = await db`select content from artifacts where arxiv_id = ${id} and kind = ${kind}`;
      return row ? (ARTIFACT_SCHEMA[kind].parse(fromJson(row.content)) as AnyArtifact) : null;
    },

    async putArtifact(id, kind, artifact) {
      const a = ARTIFACT_SCHEMA[kind].parse(artifact);
      await init();
      const result = await db`
        insert into artifacts (arxiv_id, kind, content, created_at)
        values (${id}, ${kind}, ${toJson(a)}, ${new Date().toISOString()})
        on conflict (arxiv_id, kind) do nothing`;
      return result.count > 0 ? "created" : "exists";
    },

    async listArtifactKinds(id) {
      await init();
      const rows: { kind: string }[] = await db`select kind from artifacts where arxiv_id = ${id}`;
      return LEARN_KINDS.filter((k) => rows.some((r) => r.kind === k));
    },

    async getTranslation(hash, lang, level, kind) {
      await init();
      const [row] = await db`
        select content from translations where source_hash = ${hash} and lang = ${lang} and level = ${level}`;
      return row ? TEXT_SCHEMA[kind].parse(fromJson(row.content)) : null;
    },

    async putTranslation(hash, lang, level, kind, id, text) {
      const t = TEXT_SCHEMA[kind].parse(text);
      await init();
      const result = await db`
        insert into translations (source_hash, lang, level, kind, arxiv_id, content, created_at)
        values (${hash}, ${lang}, ${level}, ${kind}, ${id}, ${toJson(t)}, ${new Date().toISOString()})
        on conflict (source_hash, lang, level) do nothing`;
      return result.count > 0 ? "created" : "exists";
    },

    async getInside(id, blockId) {
      await init();
      const [row] = await db`select content from insides where arxiv_id = ${id} and block_id = ${blockId}`;
      return row ? Architecture.parse(fromJson(row.content)) : null;
    },

    async putInside(id, blockId, inside) {
      const a = Architecture.parse(inside);
      await init();
      const result = await db`
        insert into insides (arxiv_id, block_id, content, created_at)
        values (${id}, ${blockId}, ${toJson(a)}, ${new Date().toISOString()})
        on conflict (arxiv_id, block_id) do nothing`;
      return result.count > 0 ? "created" : "exists";
    },
  };
}

/** A store whose every call rejects, so callers' "gallery unavailable" paths take over. */
function unavailableStore(reason: string): Store {
  const fail = () => Promise.reject(new Error(reason));
  return {
    getPaper: fail,
    listPapers: fail,
    publishPaper: fail,
    getArtifact: fail,
    putArtifact: fail,
    listArtifactKinds: fail,
    getTranslation: fail,
    putTranslation: fail,
    getInside: fail,
    putInside: fail,
  };
}

export function storeFromEnv(env: Record<string, string | undefined>): Store {
  if (env.DATABASE_URL) return createStore(env.DATABASE_URL);
  // The deployed project dir is read-only; without a database the gallery is simply off.
  if (env.VERCEL) return unavailableStore("No DATABASE_URL set: the gallery is disabled.");
  const dir = join(import.meta.dir, "..", "..", "data");
  mkdirSync(dir, { recursive: true });
  return createStore(`sqlite://${join(dir, "paperforge.db")}`);
}

let store: Store | undefined;

export function getStore(): Store {
  return (store ??= storeFromEnv(process.env));
}
