/**
 * The Claude models a visitor can pick for their own key. The server accepts only ids from this list.
 * `shares`: what this model makes may be saved for everyone (gallery, shared cache, publishing); results from
 * the others stay with the visitor who asked, so the gallery keeps the best read of each paper.
 */
export const MODELS = [
  { id: "claude-opus-5-5", name: "Claude Opus 5.5", shares: true, note: "Recommended: the best read of a paper for the cost." },
  { id: "claude-fable-5-1", name: "Claude Fable 5.1", shares: true, note: "Most capable. Slower, about 2.5× the cost of Opus." },
  { id: "claude-sonnet-5-5", name: "Claude Sonnet 5.5", shares: false, note: "Faster, about half the cost of Opus. What it writes stays with you: only Opus and Fable add to the gallery." },
  { id: "claude-haiku-5-5", name: "Claude Haiku 5.5", shares: false, note: "Fastest and cheapest. What it writes stays with you: only Opus and Fable add to the gallery." },
] as const;

export type ModelId = (typeof MODELS)[number]["id"];
export const DEFAULT_MODEL: ModelId = "claude-opus-5-5";

export const isModel = (id: unknown): id is ModelId => MODELS.some((m) => m.id === id);

/** "claude-opus-5-5" → "Claude Opus 5.5"; an id outside the list is formatted the same way, or shown as is. */
export function modelName(id: string): string {
  const known = MODELS.find((m) => m.id === id);
  if (known) return known.name;
  const m = /^claude-([a-z]+)-(\d+)(?:-(\d+))?(?:-\d{8})?$/.exec(id);
  if (!m) return id;
  const [, family, major, minor] = m;
  return `Claude ${family![0]!.toUpperCase()}${family!.slice(1)} ${major}${minor ? `.${minor}` : ""}`;
}
