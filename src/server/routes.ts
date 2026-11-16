import type { Server } from "bun";
import { errorMessage, listHandler, publishHandler } from "./api";
import { forge } from "./forge";
import { withModel } from "./model";
import { askHandler, insideHandler, insideLearnHandler, kindsHandler, learnHandler } from "./learn-api";
import { ndjson } from "./stream";
import { translateHandler } from "./translate-api";
import type { ForgeEvent } from "../shared/graph";

/**
 * A route that calls Claude: it runs with the visitor's model, and may wait for minutes. Bun drops a
 * connection idle for `idleTimeout` (255 s at most); the function's own limit (vercel.json) bounds these.
 */
const long =
  (handler: (req: Request) => Response | Promise<Response>) =>
  (req: Request, server: Server<unknown>) => {
    server.timeout(req, 0);
    return withModel(req, () => handler(req));
  };

/** Every /api route. `index.ts` (local, with the page) and `server.ts` (Vercel, API only) both serve these. */
export const apiRoutes = {
  "/api/forge": {
    POST: long(async (req) => {
      const { url, refresh } = (await req.json()) as { url?: string; refresh?: boolean };
      // The visitor's own key; never logged or stored.
      const apiKey = req.headers.get("x-anthropic-key") ?? undefined;
      return ndjson<ForgeEvent>(async (emit) => {
        try {
          await forge(url ?? "", emit, { refresh: !!refresh, apiKey });
        } catch (err) {
          console.error(err);
          emit({ type: "error", ...errorMessage(err) });
        }
      });
    }),
  },
  "/api/papers": { GET: (req: Request) => listHandler(req), POST: (req: Request) => publishHandler(req) },
  "/api/learn": { GET: (req: Request) => kindsHandler(req), POST: long(learnHandler) },
  "/api/ask": { POST: long(askHandler) },
  "/api/inside": { POST: long(insideHandler) },
  "/api/inside-learn": { POST: long(insideLearnHandler) },
  "/api/translate": { POST: long(translateHandler) },
};
