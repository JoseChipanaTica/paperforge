import type { JourneyArtifact, View } from "../shared/learn";
import type { ArtifactSource } from "./localStore";

/**
 * The browser store in front of a request: a stored value is used with no request (`local`); a miss fetches and
 * stores what came back. The read failing is a miss; the write is never awaited and its failure never surfaces.
 */
export async function withStored<T>(
  get: () => Promise<T | null>,
  fetch: () => Promise<T>,
  put: (v: T) => Promise<void>,
): Promise<{ value: T; local: boolean }> {
  const hit = await get().catch(() => null);
  if (hit !== null && hit !== undefined) return { value: hit, local: true };
  const value = await fetch();
  try {
    put(value).catch(() => {});
  } catch {}
  return { value, local: false };
}

/** The origin to store for what the server sent: anything but gallery or sample is the visitor's own. */
export function artifactSource(source: unknown): ArtifactSource {
  return source === "gallery" || source === "sample" ? source : "claude";
}

/**
 * Ask the server first (it's free) when it lists the kind and the browser holds no copy of the visitor's own: a shared
 * stored copy may have been replaced since. The visitor's own copy is always used with no request.
 */
export function preferServer(listedByServer: boolean, stored: ArtifactSource | null): boolean {
  return listedByServer && stored !== "claude";
}

/**
 * withStored for artifacts, which keep their origin. A shared stored copy of a kind the server lists is refreshed from
 * `shared` (the server's saved or shipped copy only, never a Claude call); that failing (offline) uses the stored copy.
 * A miss goes to `fetch`. What comes back is stored with its origin; the write is never awaited.
 */
export async function withStoredArtifact<T>(o: {
  get: () => Promise<{ artifact: T; source: ArtifactSource } | null>;
  listed: boolean;
  shared: () => Promise<{ value: T; source: string }>;
  fetch: () => Promise<{ value: T; source: string }>;
  put: (v: T, source: ArtifactSource) => Promise<void>;
}): Promise<{ value: T; local: boolean }> {
  const hit = await o.get().catch(() => null);
  const store = (r: { value: T; source: string }) => {
    try {
      o.put(r.value, artifactSource(r.source)).catch(() => {});
    } catch {}
    return { value: r.value, local: false };
  };
  if (hit) {
    if (!preferServer(o.listed, hit.source)) return { value: hit.artifact, local: true };
    const fresh = await o.shared().catch(() => null);
    return fresh ? store(fresh) : { value: hit.artifact, local: true };
  }
  return store(await o.fetch());
}

type Ends = { entering: View[] | null; leaving: View[] | null };

/**
 * A stored block walkthrough still fits the model's walkthrough on screen: it starts from the same example, i.e. the
 * item labels entering the block in the main walkthrough are those of some view of its input. (Only the start is
 * checked: Claude copies what enters but only matches what leaves loosely.) No main walkthrough: nothing to check.
 */
export function insideJourneyFits(journey: JourneyArtifact, { entering }: Ends): boolean {
  const labelled = entering?.find((v) => v.rows.length > 0);
  if (!labelled) return true;
  const want = JSON.stringify(labelled.rows);
  return journey.input.views.some((v) => JSON.stringify(v.rows) === want);
}
