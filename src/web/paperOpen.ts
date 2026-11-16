import type { Architecture, ForgeEvent, LocatedArchitecture } from "../shared/graph";
import { locateQuote } from "../shared/locate";
import type { StoredPaper } from "./localStore";

/**
 * Opening a paper from the browser store (pure): replay a stored paper unless the visitor forges again, and let a
 * shared (gallery or sample) graph replace it afterwards when it differs.
 */

/** A stored paper replays with no request (no key, no network); "forge again" or a miss goes to /api/forge. */
export function openPlan(hit: StoredPaper | null, refresh: boolean): "replay" | "network" {
  return hit && !refresh ? "replay" : "network";
}

/** The events /api/forge would have sent for this paper. */
export function storedEvents(p: StoredPaper): ForgeEvent[] {
  return [
    { type: "paper", arxiv_id: p.arxiv_id, pages: p.pages },
    { type: "graph", graph: p.graph, source: p.source, ...(p.save_token ? { save_token: p.save_token } : {}) },
  ];
}

/** The stored paper with the shared graph in place: its quotes located in the stored pages, nothing to publish. */
export function replaceWithShared(
  p: StoredPaper,
  shared: { graph: Architecture; source: "gallery" | "sample" },
  hash: string,
): StoredPaper {
  const graph: LocatedArchitecture = {
    ...shared.graph,
    arxiv_id: p.arxiv_id,
    blocks: shared.graph.blocks.map((b) => ({ ...b, source: locateQuote(p.pages, b.source_quote) })),
  };
  return { ...p, graph, graph_hash: hash, save_token: null, source: shared.source };
}

/** The background check lands only on the paper it was made for, in the same run, and never during a refresh. */
export function sharedStillApplies(
  asked: { id: string; run: unknown },
  now: { id: string | null; run: unknown; refreshing: boolean },
): boolean {
  return !now.refreshing && now.id === asked.id && now.run === asked.run;
}
