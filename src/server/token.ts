import { randomUUID, timingSafeEqual } from "node:crypto";
import type { LocatedArchitecture } from "../shared/graph";

/**
 * Save tokens: an HMAC over the graph exactly as /api/forge emitted it, so the gallery
 * only accepts graphs this server generated (arxiv_id included, so a graph can't be
 * published under another paper).
 */

let localSecret: string | undefined;

/** Throws when tokens can't be trusted: deployed without PAPERFORGE_SECRET, each instance would sign differently. */
export function getSecret(): string {
  if (process.env.PAPERFORGE_SECRET) return process.env.PAPERFORGE_SECRET;
  if (process.env.NODE_ENV === "production" || process.env.VERCEL) {
    throw new Error("PAPERFORGE_SECRET must be set when deployed.");
  }
  return (localSecret ??= randomUUID()); // local dev: tokens only need to outlive this process
}

const hmac = (graph: unknown) =>
  new Bun.CryptoHasher("sha256", getSecret()).update(JSON.stringify(graph)).digest("hex");

export function signGraph(graph: LocatedArchitecture): string {
  return hmac(graph);
}

export function verifyGraph(graph: unknown, token: string): boolean {
  const expected = Buffer.from(hmac(graph), "hex");
  const given = Buffer.from(token, "hex");
  return given.length === expected.length && timingSafeEqual(given, expected);
}
