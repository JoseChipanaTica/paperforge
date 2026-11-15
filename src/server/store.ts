import { SQL } from "bun";
import { mkdirSync } from "node:fs";
import { join } from "node:path";
import { Architecture, type PaperSummary } from "../shared/graph";

/**
 * The public gallery: one row per arXiv id, first save wins. Postgres (Neon) when
 * DATABASE_URL is set, a local SQLite file otherwise. Queries stay in the subset both
 * dialects share; created_at is ISO-8601 text so cursors compare the same in both.
 */

export interface Store {
  getPaper(id: string): Promise<Architecture | null>;
  listPapers(opts?: { limit?: number; before?: string }): Promise<PaperSummary[]>;
  publishPaper(id: string, arch: Architecture): Promise<"created" | "exists">;
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
    return (ready ??= db.unsafe(`create table if not exists papers (
      arxiv_id text primary key,
      paper_title text not null,
      model_name text not null,
      summary text not null,
      graph ${sqlite ? "text" : "jsonb"} not null,
      block_count integer not null,
      created_at text not null
    )`).catch((err) => {
      ready = undefined;
      stale = true;
      throw err;
    }));
  };

  return {
    async getPaper(id) {
      await init();
      const [row] = await db`select graph from papers where arxiv_id = ${id}`;
      if (!row) return null;
      return Architecture.parse(typeof row.graph === "string" ? JSON.parse(row.graph) : row.graph);
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
      const json = JSON.stringify(a);
      const result = await db`
        insert into papers (arxiv_id, paper_title, model_name, summary, graph, block_count, created_at)
        values (${id}, ${a.paper_title}, ${a.model_name}, ${a.summary},
                ${sqlite ? json : db`${json}::jsonb`}, ${a.blocks.length}, ${new Date().toISOString()})
        on conflict (arxiv_id) do nothing`;
      return result.count > 0 ? "created" : "exists";
    },
  };
}

/** A store whose every call rejects, so callers' "gallery unavailable" paths take over. */
function unavailableStore(reason: string): Store {
  const fail = () => Promise.reject(new Error(reason));
  return { getPaper: fail, listPapers: fail, publishPaper: fail };
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
