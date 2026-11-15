import index from "./src/web/index.html";
import { errorMessage, listHandler, publishHandler } from "./src/server/api";
import { forge } from "./src/server/forge";
import type { ForgeEvent } from "./src/shared/graph";

const server = Bun.serve({
  port: Number(process.env.PORT ?? 3000),
  idleTimeout: 255, // Claude can take a couple of minutes on a long paper
  routes: {
    "/": index,
    "/api/forge": {
      POST: async (req) => {
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
    "/api/papers": { GET: (req) => listHandler(req), POST: (req) => publishHandler(req) },
  },
  development: process.env.NODE_ENV !== "production" && { hmr: true, console: true },
});

console.log(`Paperforge running at ${server.url}`);
