import Anthropic from "@anthropic-ai/sdk";
import index from "./src/web/index.html";
import { forge } from "./src/server/forge";
import type { ForgeEvent } from "./src/shared/graph";

function errorMessage(err: unknown): string {
  if (err instanceof Anthropic.AuthenticationError) {
    return "No valid Anthropic API key. Set ANTHROPIC_API_KEY in .env to forge new papers.";
  }
  if (err instanceof Anthropic.RateLimitError) return "Claude is rate limited right now. Try again in a minute.";
  if (err instanceof Anthropic.APIError) return `Claude API error ${err.status}: ${err.message}`;
  if (err instanceof Error && /api key|apiKey|authToken/i.test(err.message)) {
    return "No Anthropic API key found. Set ANTHROPIC_API_KEY in .env to forge new papers (the Transformer sample works without one).";
  }
  return err instanceof Error ? err.message : String(err);
}

const server = Bun.serve({
  port: Number(process.env.PORT ?? 3000),
  idleTimeout: 255, // Claude can take a couple of minutes on a long paper
  routes: {
    "/": index,
    "/api/forge": {
      POST: async (req) => {
        const { url, refresh } = (await req.json()) as { url?: string; refresh?: boolean };
        const encoder = new TextEncoder();
        const stream = new ReadableStream({
          async start(controller) {
            const emit = (e: ForgeEvent) => controller.enqueue(encoder.encode(JSON.stringify(e) + "\n"));
            try {
              await forge(url ?? "", emit, { refresh: !!refresh });
            } catch (err) {
              console.error(err);
              emit({ type: "error", message: errorMessage(err) });
            }
            controller.close();
          },
        });
        return new Response(stream, { headers: { "Content-Type": "application/x-ndjson" } });
      },
    },
  },
  development: process.env.NODE_ENV !== "production" && { hmr: true, console: true },
});

console.log(`Paperforge running at ${server.url}`);
