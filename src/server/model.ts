import { AsyncLocalStorage } from "node:async_hooks";
import { DEFAULT_MODEL, isModel, MODELS } from "../shared/models";

/** The model when a call has no visitor's pick (the CLI scripts, tests); `PAPERFORGE_MODEL` overrides it. */
const FALLBACK = process.env.PAPERFORGE_MODEL ?? DEFAULT_MODEL;
const picked = new AsyncLocalStorage<string>();

/** The model for the current request: the visitor's pick, else the fallback. */
export const currentModel = () => picked.getStore() ?? FALLBACK;

/** Whether this request's results may be saved for everyone; a model outside MODELS (`PAPERFORGE_MODEL`) is the operator's call. */
export const sharesResults = () => MODELS.find((m) => m.id === currentModel())?.shares ?? true;

/** Runs `run` with the model the visitor picked (`x-anthropic-model`); an id not in MODELS is ignored. */
export function withModel<T>(req: Request, run: () => T): T {
  const id = req.headers.get("x-anthropic-model")?.trim();
  return isModel(id) ? picked.run(id, run) : run();
}

/** Under the function's 900 s limit (vercel.json); also opts out of the SDK's non-streaming duration guard. */
export const TIMEOUT_MS = 870_000;
