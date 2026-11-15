import { join } from "node:path";
import { Architecture } from "../shared/graph";
import {
  ARTIFACT_SCHEMA,
  LEARN_KINDS,
  LearnKind,
  artifactBlockCount,
  isUsable,
  pruneArtifact,
  JourneyArtifact,
  type AnyArtifact,
  type AskEvent,
  type AskTurn,
  type InsideEvent,
  type LearnEvent,
} from "../shared/learn";
import { cleanMotion, motionFits, type MotionArtifact } from "../shared/motion";
import { DEFAULT_LEVEL, Lang, Level } from "../shared/translate";
import { errorMessage } from "./api";
import { parseArxivId } from "./arxiv";
import { loadPaperPages } from "./forge";
import { getStore, type Store } from "./store";
import { askStream, generateArtifact, generateInside } from "./tutor";

/** Routes for the learning features: /api/learn (generate or fetch), /api/ask, GET /api/learn (kinds), /api/inside. */

export interface LearnDeps {
  store: Store;
  generate: typeof generateArtifact;
  ask: typeof askStream;
  inside: typeof generateInside;
  loadPages: (id: string) => Promise<string[]>;
  samplesDir: string;
}

export function defaultLearnDeps(): LearnDeps {
  return {
    store: getStore(),
    generate: generateArtifact,
    ask: askStream,
    inside: generateInside,
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

/**
 * A committed sample, if it fits the paper's published graph. Samples are written for one
 * graph; if the gallery holds a different one (re-forged and published), a sample whose
 * blocks don't all exist would leave tour steps and code lines pointing at nothing.
 */
async function readSample(dir: string, id: string, kind: LearnKind, published: Architecture | null): Promise<AnyArtifact | null> {
  const f = Bun.file(join(dir, `${fileId(id)}.${kind}.json`));
  if (!(await f.exists())) return null;
  const sample = ARTIFACT_SCHEMA[kind].parse(await f.json()) as AnyArtifact;
  if (!published) return sample;
  const fits = artifactBlockCount(kind, pruneArtifact(kind, sample, published)) === artifactBlockCount(kind, sample);
  return fits ? sample : null;
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
    if (!id || !kind.success) return fail("Expected an arXiv id and a kind (pytorch, journey or motion).", "bad_input");
    const k = kind.data;

    try {
      const published = await publishedGraph(d.store, id);
      // Motion animates a walkthrough: the one everyone sees for this paper, else the visitor's own.
      const found = k === "motion" ? await journeyFor(d, id, published, body?.journey) : null;
      const journey = found?.journey ?? null;
      if (k === "motion" && !journey) return fail("Open the walkthrough first.", "bad_input");
      const fits = (a: AnyArtifact) => k !== "motion" || motionFits(a as MotionArtifact, journey!);
      // A saved artifact that doesn't fit (e.g. motion for 1 of 18 steps) is replaced, not served forever.
      let unusable = false;
      if (published) {
        const saved = await d.store.getArtifact(id, k).catch(() => null);
        if (saved && fits(saved)) return emit({ type: "learn", kind: k, artifact: saved, source: "gallery", saved: true });
        unusable = saved !== null;
      }
      const sample = await readSample(d.samplesDir, id, k, published);
      if (sample && fits(sample)) return emit({ type: "learn", kind: k, artifact: sample, source: "sample", saved: false });

      // A published paper always uses its stored graph; otherwise the visitor's own forge.
      const parsed = Architecture.safeParse(body?.graph);
      const graph = published ?? (parsed.success ? parsed.data : null);
      if (!graph) return fail("Forge the paper first, then try again.", "bad_input");

      emit({ type: "status", message: STATUS[k] });
      const pages = await d.loadPages(id);
      const generated = await d.generate(k, { pages, graph, apiKey, ...(journey && { journey }) });
      const artifact = k === "motion" ? cleanMotion(generated as MotionArtifact, graph, journey) : pruneArtifact(k, generated, graph);
      if (!isUsable(k, artifact) || !fits(artifact)) return fail(UNUSABLE[k]);

      // Only shared inputs make shared results: motion from a visitor's own walkthrough isn't saved.
      if (!published || found?.posted) return emit({ type: "learn", kind: k, artifact, source: "claude", saved: false });
      try {
        // First save wins: if another visitor's landed meanwhile, show everyone that one.
        const result = await d.store.putArtifact(id, k, artifact, { replace: unusable });
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
  journey: "Claude is following an input through the model…",
  motion: "Claude is animating each block…",
};

const UNUSABLE: Record<LearnKind, string> = {
  pytorch: "Claude couldn't map this to the model's blocks. Try again.",
  journey: "Couldn't build the walkthrough. Try again.",
  motion: "Couldn't animate this model.",
};

/**
 * The walkthrough motion follows: published (gallery), then the committed sample, then the one the
 * visitor posted (`posted`: its motion is never saved for everyone).
 */
async function journeyFor(
  d: LearnDeps,
  id: string,
  published: Architecture | null,
  body: unknown,
): Promise<{ journey: JourneyArtifact; posted: boolean } | null> {
  if (published) {
    const saved = await d.store.getArtifact(id, "journey").catch(() => null);
    if (saved) return { journey: saved as JourneyArtifact, posted: false };
  }
  const sample = await readSample(d.samplesDir, id, "journey", published);
  if (sample) return { journey: sample as JourneyArtifact, posted: false };
  const parsed = JourneyArtifact.safeParse(body);
  return parsed.success ? { journey: parsed.data, posted: true } : null;
}

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
  const lang = Lang.catch("en").parse(body?.lang ?? "en");
  const level = Level.catch(DEFAULT_LEVEL).parse(body?.level ?? DEFAULT_LEVEL);
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
      for await (const text of d.ask({ pages, graph, block, question, history, lang, level, apiKey })) emit({ type: "delta", text });
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
  const published = await publishedGraph(d.store, id);
  const saved = published ? await d.store.listArtifactKinds(id).catch(() => []) : [];
  const available = await Promise.all(
    LEARN_KINDS.map(async (k) => saved.includes(k) || (await readSample(d.samplesDir, id, k, published)) !== null),
  );
  return Response.json(LEARN_KINDS.filter((_, i) => available[i]));
}

/**
 * POST /api/inside { arxiv_id, block_id, graph }: a block's internal structure as a graph of its
 * own. Same order as /api/learn: gallery (published papers) → committed sample → Claude, and only
 * a published paper's inside views are saved (first save wins). Quotes are located in the browser,
 * which already has the paper's text.
 */
export async function insideHandler(req: Request, overrides: Partial<LearnDeps> = {}): Promise<Response> {
  const d = { ...defaultLearnDeps(), ...overrides };
  const body = await readBody(req);
  const apiKey = keyOf(req);

  return ndjson<InsideEvent>(async (emit) => {
    const fail = (message: string) => emit({ type: "error", message, code: "bad_input" });
    const id = parseArxivId(String(body?.arxiv_id ?? ""));
    const blockId = typeof body?.block_id === "string" ? body.block_id : "";
    if (!id || !/^[\w-]{1,100}$/.test(blockId)) return fail("Expected an arXiv id and a block_id.");

    try {
      const published = await publishedGraph(d.store, id);
      const parsed = Architecture.safeParse(body?.graph);
      const graph = published ?? (parsed.success ? parsed.data : null);
      if (!graph) return fail("Forge the paper first, then try again.");
      const block = graph.blocks.find((b) => b.id === blockId);
      if (!block) return fail("That block isn't in this model.");
      const done = (inside: Architecture, source: "gallery" | "sample" | "claude", saved: boolean) =>
        emit({ type: "inside", block_id: blockId, graph: inside, source, saved });

      if (published) {
        const saved = await d.store.getInside(id, blockId).catch(() => null);
        if (saved) return done(saved, "gallery", true);
      }
      const sample = Bun.file(join(d.samplesDir, `${fileId(id)}.inside.${blockId}.json`));
      if (await sample.exists()) return done(Architecture.parse(await sample.json()), "sample", false);

      emit({ type: "status", message: `Claude is looking inside ${block.label}…` });
      const pages = await d.loadPages(id);
      const inside = await d.inside(block, { pages, graph, apiKey });
      if (!published) return done(inside, "claude", false);
      try {
        // First save wins: if another visitor's landed meanwhile, show everyone that one.
        const result = await d.store.putInside(id, blockId, inside);
        const stored = result === "exists" ? await d.store.getInside(id, blockId) : inside;
        done(stored ?? inside, "claude", true);
      } catch (err) {
        console.error("inside view save failed:", err);
        done(inside, "claude", false);
      }
    } catch (err) {
      console.error(err);
      emit({ type: "error", ...errorMessage(err) });
    }
  });
}
