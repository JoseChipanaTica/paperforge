import { DEFAULT_MODEL, isModel, type ModelId } from "../shared/models";

/** The visitor's Claude model, remembered in this browser next to their key. */
const STORAGE = "paperforge.model";

export function chosenModel(): ModelId {
  try {
    const id = localStorage.getItem(STORAGE);
    return isModel(id) ? id : DEFAULT_MODEL;
  } catch {
    return DEFAULT_MODEL;
  }
}

export function chooseModel(id: ModelId) {
  try {
    if (id === DEFAULT_MODEL) localStorage.removeItem(STORAGE);
    else localStorage.setItem(STORAGE, id);
  } catch {}
}

/** Headers for a request that may call Claude: the visitor's model and, when set, their key. */
export const claudeHeaders = (key: string) => ({
  "Content-Type": "application/json",
  "x-anthropic-model": chosenModel(),
  ...(key && { "x-anthropic-key": key }),
});
