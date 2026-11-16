import { errorMessage, listHandler, publishHandler } from "./api";
import { forge } from "./forge";
import { askHandler, insideHandler, insideLearnHandler, kindsHandler, learnHandler } from "./learn-api";
import { translateHandler } from "./translate-api";
import type { ForgeEvent } from "../shared/graph";

/** Every /api route. `index.ts` (local, with the page) and `server.ts` (Vercel, API only) both serve these. */
export const apiRoutes = {
  "/api/forge": {
    POST: async (req: Request) => {
      const { url, refresh } = (await req.json()) as { url?: string; refresh?: boolean };
      // The visitor's own key; never logged or stored.
      const apiKey = req.headers.get("x-anthropic-key") ?? undefined;
      const encoder = new TextEncoder();
      const stream = new ReadableStream({
        async start(controller) {
          const emit = (e: ForgeEvent) => controller.enqueue(encoder.encode(JSON.stringify(e) + "\n"));
          try {
            await forge(url ?? "", emit, { refresh: !!refresh, apiKey });
          } catch (err) {
            console.error(err);
            emit({ type: "error", ...errorMessage(err) });
          }
          controller.close();
        },
      });
      return new Response(stream, { headers: { "Content-Type": "application/x-ndjson" } });
    },
  },
  "/api/papers": { GET: (req: Request) => listHandler(req), POST: (req: Request) => publishHandler(req) },
  "/api/learn": { GET: (req: Request) => kindsHandler(req), POST: (req: Request) => learnHandler(req) },
  "/api/ask": { POST: (req: Request) => askHandler(req) },
  "/api/inside": { POST: (req: Request) => insideHandler(req) },
  "/api/inside-learn": { POST: (req: Request) => insideLearnHandler(req) },
  "/api/translate": { POST: (req: Request) => translateHandler(req) },
};
