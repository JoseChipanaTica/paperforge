import { beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import type { AnyArtifact, AskEvent, LearnEvent, LearnKind } from "../shared/learn";
import { askHandler, kindsHandler, learnHandler, type LearnDeps } from "./learn-api";
import { createStore, type Store } from "./store";

const SAMPLES = join(import.meta.dir, "../../samples");
const graph = Architecture.parse(await Bun.file(join(SAMPLES, "1706.03762.json")).json());
const walkthrough = (await Bun.file(join(SAMPLES, "1706.03762.walkthrough.json")).json()) as AnyArtifact;
const example = (await Bun.file(join(SAMPLES, "1706.03762.example.json")).json()) as AnyArtifact;
const ID = "2401.00001"; // no committed sample for this id

let calls: { kind: LearnKind; graph: Architecture; apiKey?: string }[];
let next: AnyArtifact;
let samplesDir: string;

function deps(store: Store, extra: Partial<LearnDeps> = {}): Partial<LearnDeps> {
  return {
    store,
    samplesDir,
    loadPages: async () => ["page"],
    generate: async (kind, { graph, apiKey }) => {
      calls.push({ kind, graph, apiKey });
      return next;
    },
    ask: async function* () {
      yield "Hel";
      yield "lo";
    },
    ...extra,
  };
}

const post = (path: string, body: unknown, key?: string) =>
  new Request(`http://x${path}`, {
    method: "POST",
    headers: key ? { "x-anthropic-key": key } : {},
    body: JSON.stringify(body),
  });

async function events<T>(res: Response): Promise<T[]> {
  return (await res.text()).split("\n").filter(Boolean).map((l) => JSON.parse(l) as T);
}

async function learn(body: unknown, d: Partial<LearnDeps>, key?: string) {
  const all = await events<LearnEvent>(await learnHandler(post("/api/learn", body, key), d));
  return all.find((e) => e.type === "learn" || e.type === "error")!;
}

const broken: Store = {
  getPaper: () => Promise.reject(new Error("db down")),
  listPapers: () => Promise.reject(new Error("db down")),
  publishPaper: () => Promise.reject(new Error("db down")),
  getArtifact: () => Promise.reject(new Error("db down")),
  putArtifact: () => Promise.reject(new Error("db down")),
  listArtifactKinds: () => Promise.reject(new Error("db down")),
};

beforeEach(() => {
  calls = [];
  next = walkthrough;
  samplesDir = mkdtempSync(join(tmpdir(), "paperforge-samples-"));
});

describe("POST /api/learn", () => {
  test("a saved artifact for a published paper loads with no key and no Claude call", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    await store.putArtifact(ID, "walkthrough", walkthrough);
    const e = await learn({ arxiv_id: ID, kind: "walkthrough" }, deps(store));
    expect(e).toMatchObject({ type: "learn", source: "gallery", saved: true, artifact: walkthrough });
    expect(calls).toEqual([]);
  });

  test("committed samples are served, including old-style slash ids", async () => {
    await Bun.write(join(samplesDir, "cs_0112017.example.json"), JSON.stringify(example));
    const e = await learn({ arxiv_id: "cs/0112017", kind: "example", graph }, deps(createStore("sqlite://:memory:")));
    expect(e).toMatchObject({ type: "learn", source: "sample", saved: false });
    expect(calls).toEqual([]);
  });

  test("an unpublished paper is generated from the posted graph and not saved", async () => {
    const store = createStore("sqlite://:memory:");
    const e = await learn({ arxiv_id: ID, kind: "walkthrough", graph }, deps(store), "  sk-test ");
    expect(e).toMatchObject({ type: "learn", source: "claude", saved: false });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.graph).toEqual(graph);
    expect(calls[0]!.apiKey).toBe("sk-test");
    expect(await store.getArtifact(ID, "walkthrough")).toBeNull();
  });

  test("a published paper uses its stored graph, not the posted one, and saves the result", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    const tampered = { ...graph, model_name: "Something Else" };
    const e = await learn({ arxiv_id: ID, kind: "walkthrough", graph: tampered }, deps(store));
    expect(e).toMatchObject({ type: "learn", source: "claude", saved: true });
    expect(calls[0]!.graph.model_name).toBe(graph.model_name);
    expect(await store.getArtifact(ID, "walkthrough")).toEqual(walkthrough);
  });

  test("when another visitor saved first, everyone gets the stored one", async () => {
    const real = createStore("sqlite://:memory:");
    await real.publishPaper(ID, graph);
    const theirs = { ...(walkthrough as object), intro: "Saved first by someone else." } as AnyArtifact;
    let misses = 1;
    const racing: Store = {
      ...real,
      // The first lookup misses; the other visitor's save lands while we generate.
      getArtifact: async (id, kind) => {
        if (misses-- > 0) {
          await real.putArtifact(id, kind, theirs);
          return null;
        }
        return real.getArtifact(id, kind);
      },
    };
    const e = await learn({ arxiv_id: ID, kind: "walkthrough" }, deps(racing));
    expect(e).toMatchObject({ type: "learn", saved: true, artifact: theirs });
  });

  test("an artifact that maps to none of the model's blocks is an error and isn't saved", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    next = { ...(walkthrough as object), steps: [{ block_id: "nope", title: "t", narration: "n", shape_change: "s" }] } as AnyArtifact;
    const e = await learn({ arxiv_id: ID, kind: "walkthrough" }, deps(store));
    expect(e.type).toBe("error");
    expect(await store.getArtifact(ID, "walkthrough")).toBeNull();
  });

  test("a broken database still generates, unsaved", async () => {
    const e = await learn({ arxiv_id: ID, kind: "walkthrough", graph }, deps(broken));
    expect(e).toMatchObject({ type: "learn", source: "claude", saved: false });
  });

  test("a missing key surfaces as needs_key", async () => {
    const e = await learn(
      { arxiv_id: ID, kind: "walkthrough", graph },
      deps(createStore("sqlite://:memory:"), {
        generate: async () => {
          throw new Error("Could not resolve authentication method. Expected either apiKey or authToken to be set.");
        },
      }),
    );
    expect(e).toMatchObject({ type: "error", code: "needs_key" });
  });

  test("an unknown kind or missing graph is bad_input", async () => {
    const store = createStore("sqlite://:memory:");
    expect(await learn({ arxiv_id: ID, kind: "poem", graph }, deps(store))).toMatchObject({ code: "bad_input" });
    expect(await learn({ arxiv_id: ID, kind: "example" }, deps(store))).toMatchObject({ code: "bad_input" });
  });
});

describe("POST /api/ask", () => {
  const ask = (body: object, store = createStore("sqlite://:memory:")) =>
    askHandler(post("/api/ask", { arxiv_id: ID, block_id: "encoder_ffn", question: "Why 2048?", history: [], graph, ...body }), deps(store));

  test("streams deltas, then done", async () => {
    const res = await ask({});
    expect(res.status).toBe(200);
    expect(await events<AskEvent>(res)).toEqual([{ type: "delta", text: "Hel" }, { type: "delta", text: "lo" }, { type: "done" }]);
  });

  test("rejects oversized or malformed questions before calling Claude", async () => {
    expect((await ask({ question: "x".repeat(1001) })).status).toBe(400);
    expect((await ask({ history: Array.from({ length: 11 }, () => ({ role: "user", text: "hi" })) })).status).toBe(400);
    expect((await ask({ history: [{ role: "user", text: "x".repeat(4001) }] })).status).toBe(400);
    expect((await ask({ block_id: "nope" })).status).toBe(400);
  });
});

describe("GET /api/learn", () => {
  test("lists saved and sample kinds, slash ids included", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper("cs/0112017", graph);
    await store.putArtifact("cs/0112017", "pytorch", (await Bun.file(join(SAMPLES, "1706.03762.pytorch.json")).json()) as AnyArtifact);
    await Bun.write(join(samplesDir, "cs_0112017.walkthrough.json"), JSON.stringify(walkthrough));
    const res = await kindsHandler(new Request("http://x/api/learn?arxiv_id=cs/0112017"), deps(store));
    expect(await res.json()).toEqual(["pytorch", "walkthrough"]);
  });
});
