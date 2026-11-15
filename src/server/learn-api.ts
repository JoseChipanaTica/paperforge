import { join } from "node:path";
import { Architecture } from "../shared/graph";
import {
  ARTIFACT_SCHEMA,
  LEARN_KINDS,
  LearnKind,
  artifactBlockCount,
  pruneArtifact,
  type AnyArtifact,
  type AskEvent,
  type AskTurn,
  type LearnEvent,
} from "../shared/learn";
import { errorMessage } from "./api";
import { parseArxivId } from "./arxiv";
import { loadPaperPages } from "./forge";
import { getStore, type Store } from "./store";
import { askStream, generateArtifact } from "./tutor";

/** Routes for the learning features: /api/learn (generate or fetch), /api/ask, GET /api/learn (kinds). */

export interface LearnDeps {
  store: Store;
  generate: typeof generateArtifact;
  ask: typeof askStream;
  loadPages: (id: string) => Promise<string[]>;
  samplesDir: string;
}

export function defaultLearnDeps(): LearnDeps {
  return {
    store: getStore(),
    generate: generateArtifact,
    ask: askStream,
    loadPages: (id) => loadPaperPages(id),
    samplesDir: join(import.meta.dir, "..", "..", "samples"),
  };
}

const MAX_QUESTION = 1000;
const MAX_TURNS = 10;
const MAX_TURN_TEXT = 4000;

const fileId = (id: string) => id.replace(/\//g, "_");
const keyOf = (req: Request) => req.headers.get("x-anthropic-key")?.trim() || undefined;
const bad = (error: string) => Response.json({ error }, { status: 400 });

function ndjson<E>(run: (emit: (e: E) => void) => Promise<void>): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      await run((e) => controller.enqueue(encoder.encode(JSON.stringify(e) + "\n")));
      controller.close();
    },
  });
  return new Response(stream, { headers: { "Content-Type": "application/x-ndjson" } });
}

/** The published graph, or null when the paper isn't in the gallery (or the gallery is down). */
const publishedGraph = (store: Store, id: string) =>
  store.getPaper(id).catch((err) => (console.error("gallery lookup failed:", err), null));

async function readSample(dir: string, id: string, kind: LearnKind): Promise<AnyArtifact | null> {
  const f = Bun.file(join(dir, `${fileId(id)}.${kind}.json`));
  return (await f.exists()) ? (ARTIFACT_SCHEMA[kind].parse(await f.json()) as AnyArtifact) : null;
}

async function readBody(req: Request): Promise<Record<string, unknown> | null> {
  try {
    const body = await req.json();
    return body && typeof body === "object" ? (body as Record<string, unknown>) : null;
  } catch {
    return null;
  }
}

export async function learnHandler(req: Request, overrides: Partial<LearnDeps> = {}): Promise<Response> {
  const d = { ...defaultLearnDeps(), ...overrides };
  const body = await readBody(req);
  const apiKey = keyOf(req);

  return ndjson<LearnEvent>(async (emit) => {
    const fail = (message: string, code?: "bad_input") => emit({ type: "error", message, ...(code && { code }) });
    const id = parseArxivId(String(body?.arxiv_id ?? ""));
    const kind = LearnKind.safeParse(body?.kind);
    if (!id || !kind.success) return fail("Expected an arXiv id and a kind (pytorch, walkthrough or example).", "bad_input");
    const k = kind.data;

    try {
      const published = await publishedGraph(d.store, id);
      if (published) {
        const saved = await d.store.getArtifact(id, k).catch(() => null);
        if (saved) return emit({ type: "learn", kind: k, artifact: saved, source: "gallery", saved: true });
      }
      const sample = await readSample(d.samplesDir, id, k);
      if (sample) return emit({ type: "learn", kind: k, artifact: sample, source: "sample", saved: false });

      // A published paper always uses its stored graph; otherwise the visitor's own forge.
      const parsed = Architecture.safeParse(body?.graph);
      const graph = published ?? (parsed.success ? parsed.data : null);
      if (!graph) return fail("Forge the paper first, then try again.", "bad_input");

      emit({ type: "status", message: STATUS[k] });
      const pages = await d.loadPages(id);
      const artifact = pruneArtifact(k, await d.generate(k, { pages, graph, apiKey }), graph);
      if (artifactBlockCount(k, artifact) === 0) return fail("Claude couldn't map this to the model's blocks. Try again.");

      if (!published) return emit({ type: "learn", kind: k, artifact, source: "claude", saved: false });
      try {
        // First save wins: if another visitor's landed meanwhile, show everyone that one.
        const result = await d.store.putArtifact(id, k, artifact);
        const stored = result === "exists" ? await d.store.getArtifact(id, k) : artifact;
        emit({ type: "learn", kind: k, artifact: stored ?? artifact, source: "claude", saved: true });
      } catch (err) {
        console.error("artifact save failed:", err);
        emit({ type: "learn", kind: k, artifact, source: "claude", saved: false });
      }
    } catch (err) {
      console.error(err);
      emit({ type: "error", ...errorMessage(err) });
    }
  });
}

const STATUS: Record<LearnKind, string> = {
  pytorch: "Claude is writing the PyTorch code…",
  walkthrough: "Claude is preparing the walkthrough…",
  example: "Claude is working out an easy example…",
};

function parseHistory(raw: unknown): AskTurn[] | null {
  if (raw === undefined) return [];
  if (!Array.isArray(raw) || raw.length > MAX_TURNS) return null;
  const turns: AskTurn[] = [];
  for (const t of raw) {
    if (!t || (t.role !== "user" && t.role !== "assistant") || typeof t.text !== "string") return null;
    if (t.text.length > MAX_TURN_TEXT) return null;
    turns.push({ role: t.role, text: t.text });
  }
  return turns;
}

export async function askHandler(req: Request, overrides: Partial<LearnDeps> = {}): Promise<Response> {
  const d = { ...defaultLearnDeps(), ...overrides };
  const body = await readBody(req);
  const id = parseArxivId(String(body?.arxiv_id ?? ""));
  const question = typeof body?.question === "string" ? body.question.trim() : "";
  const history = parseHistory(body?.history);
  if (!id) return bad("Expected an arXiv id.");
  if (!question || question.length > MAX_QUESTION) return bad(`Ask a question of up to ${MAX_QUESTION} characters.`);
  if (!history) return bad(`History must be at most ${MAX_TURNS} turns of up to ${MAX_TURN_TEXT} characters.`);

  const published = await publishedGraph(d.store, id);
  const parsed = Architecture.safeParse(body?.graph);
  const graph = published ?? (parsed.success ? parsed.data : null);
  if (!graph) return bad("Forge the paper first, then ask.");
  const block = graph.blocks.find((b) => b.id === body?.block_id);
  if (!block) return bad("That block isn't in this model.");
  const apiKey = keyOf(req);

  return ndjson<AskEvent>(async (emit) => {
    try {
      const pages = await d.loadPages(id);
      for await (const text of d.ask({ pages, graph, block, question, history, apiKey })) emit({ type: "delta", text });
      emit({ type: "done" });
    } catch (err) {
      console.error(err);
      emit({ type: "error", ...errorMessage(err) });
    }
  });
}

export async function kindsHandler(req: Request, overrides: Partial<LearnDeps> = {}): Promise<Response> {
  const d = { ...defaultLearnDeps(), ...overrides };
  const id = parseArxivId(new URL(req.url).searchParams.get("arxiv_id") ?? "");
  if (!id) return bad("Expected ?arxiv_id=");
  const saved = (await publishedGraph(d.store, id)) ? await d.store.listArtifactKinds(id).catch(() => []) : [];
  const available = await Promise.all(
    LEARN_KINDS.map(async (k) => saved.includes(k) || (await Bun.file(join(d.samplesDir, `${fileId(id)}.${k}.json`)).exists())),
  );
  return Response.json(LEARN_KINDS.filter((_, i) => available[i]));
}
