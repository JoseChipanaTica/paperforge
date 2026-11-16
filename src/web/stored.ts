import type { JourneyArtifact, View } from "../shared/learn";

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
