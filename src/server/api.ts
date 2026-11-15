import Anthropic from "@anthropic-ai/sdk";
import { Architecture, type ForgeEvent } from "../shared/graph";
import { getStore, type Store } from "./store";
import { verifyGraph } from "./token";

const MAX_BODY = 262_144; // 256 KB: a large graph is ~40 KB

const json = (body: unknown, status = 200) => Response.json(body, { status });
const unavailable = () => json({ error: "Gallery unavailable" }, 503);

export async function listHandler(req: Request, store: Store = getStore()): Promise<Response> {
  const params = new URL(req.url).searchParams;
  try {
    const papers = await store.listPapers({
      limit: Number(params.get("limit") ?? 24),
      before: params.get("before") ?? undefined,
    });
    return json(papers);
  } catch (err) {
    console.error("gallery list failed:", err);
    return unavailable();
  }
}

export async function publishHandler(req: Request, store: Store = getStore()): Promise<Response> {
  if (Number(req.headers.get("content-length") ?? 0) > MAX_BODY) return json({ error: "Too large" }, 413);
  const text = await req.text();
  if (text.length > MAX_BODY) return json({ error: "Too large" }, 413);

  let body: { graph?: { arxiv_id?: unknown }; save_token?: unknown };
  try {
    body = JSON.parse(text);
  } catch {
    return json({ error: "Invalid JSON" }, 400);
  }
  const { graph, save_token } = body ?? {};
  if (!graph || typeof graph.arxiv_id !== "string" || typeof save_token !== "string") {
    return json({ error: "Expected { graph, save_token }" }, 400);
  }
  // Only graphs this server generated (and signed in the forge stream) can be published.
  if (!verifyGraph(graph, save_token)) return json({ error: "This graph wasn't forged here" }, 403);
  const parsed = Architecture.safeParse(graph);
  if (!parsed.success) return json({ error: "Graph doesn't match the schema" }, 400);

  try {
    const result = await store.publishPaper(graph.arxiv_id, parsed.data);
    return result === "created"
      ? json({ arxiv_id: graph.arxiv_id }, 201)
      : json({ arxiv_id: graph.arxiv_id, error: "Already in the gallery" }, 409);
  } catch (err) {
    console.error("gallery publish failed:", err);
    return unavailable();
  }
}

type ErrorEvent = Omit<Extract<ForgeEvent, { type: "error" }>, "type">;

export function errorMessage(err: unknown): ErrorEvent {
  if (err instanceof Anthropic.AuthenticationError) {
    return { code: "needs_key", message: "That Anthropic API key was rejected. Check it and try again." };
  }
  if (err instanceof Anthropic.RateLimitError) {
    return { code: "rate_limited", message: "Claude is rate limited right now. Try again in a minute." };
  }
  if (err instanceof Anthropic.APIError) return { message: `Claude API error ${err.status}: ${err.message}` };
  const message = err instanceof Error ? err.message : String(err);
  if (/api key|apiKey|authToken/i.test(message)) {
    return { code: "needs_key", message: "Add your Anthropic API key to forge new papers (published papers work without one)." };
  }
  if (/doesn't look like an arXiv/.test(message)) return { code: "bad_input", message };
  return { message };
}
