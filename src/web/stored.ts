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
