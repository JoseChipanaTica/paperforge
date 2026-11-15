import { beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import type { AnyArtifact, AskEvent, InsideEvent, InsideLearnEvent, JourneyArtifact, LearnEvent, LearnKind } from "../shared/learn";
import { askHandler, insideHandler, insideLearnHandler, kindsHandler, learnHandler, type LearnDeps } from "./learn-api";
import { createStore, type Store } from "./store";

const SAMPLES = join(import.meta.dir, "../../samples");
const graph = Architecture.parse(await Bun.file(join(SAMPLES, "1706.03762.json")).json());
const journey = (await Bun.file(join(SAMPLES, "1706.03762.journey.json")).json()) as JourneyArtifact;
const ID = "2401.00001"; // no committed sample for this id
const innerGraph = Architecture.parse(await Bun.file(join(SAMPLES, "1706.03762.inside.encoder_self_attention.json")).json());

let calls: { kind: LearnKind; graph: Architecture; apiKey?: string; journey?: JourneyArtifact }[];
let next: AnyArtifact;
let samplesDir: string;

function deps(store: Store, extra: Partial<LearnDeps> = {}): Partial<LearnDeps> {
  return {
    store,
    samplesDir,
    loadPages: async () => ["page"],
    generate: async (kind, { graph, apiKey, journey }) => {
      calls.push({ kind, graph, apiKey, journey });
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
  getTranslation: () => Promise.reject(new Error("db down")),
  putTranslation: () => Promise.reject(new Error("db down")),
  getInside: () => Promise.reject(new Error("db down")),
  putInside: () => Promise.reject(new Error("db down")),
  getInsideLearn: () => Promise.reject(new Error("db down")),
  putInsideLearn: () => Promise.reject(new Error("db down")),
};

beforeEach(() => {
  calls = [];
  next = journey;
  samplesDir = mkdtempSync(join(tmpdir(), "paperforge-samples-"));
});

describe("POST /api/learn", () => {
  test("a saved artifact for a published paper loads with no key and no Claude call", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    await store.putArtifact(ID, "journey", journey);
    const e = await learn({ arxiv_id: ID, kind: "journey" }, deps(store));
    expect(e).toMatchObject({ type: "learn", source: "gallery", saved: true, artifact: journey });
    expect(calls).toEqual([]);
  });

  test("committed samples are served, including old-style slash ids", async () => {
    await Bun.write(join(samplesDir, "cs_0112017.journey.json"), JSON.stringify(journey));
    const e = await learn({ arxiv_id: "cs/0112017", kind: "journey", graph }, deps(createStore("sqlite://:memory:")));
    expect(e).toMatchObject({ type: "learn", source: "sample", saved: false });
    expect(calls).toEqual([]);
  });

  test("an unpublished paper is generated from the posted graph and not saved", async () => {
    const store = createStore("sqlite://:memory:");
    const e = await learn({ arxiv_id: ID, kind: "journey", graph }, deps(store), "  sk-test ");
    expect(e).toMatchObject({ type: "learn", source: "claude", saved: false });
    expect(calls).toHaveLength(1);
    expect(calls[0]!.graph).toEqual(graph);
    expect(calls[0]!.apiKey).toBe("sk-test");
    expect(await store.getArtifact(ID, "journey")).toBeNull();
  });

  test("a published paper uses its stored graph, not the posted one, and saves the result", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    const tampered = { ...graph, model_name: "Something Else" };
    const e = await learn({ arxiv_id: ID, kind: "journey", graph: tampered }, deps(store));
    expect(e).toMatchObject({ type: "learn", source: "claude", saved: true });
    expect(calls[0]!.graph.model_name).toBe(graph.model_name);
    expect(await store.getArtifact(ID, "journey")).toEqual(journey);
  });

  test("when another visitor saved first, everyone gets the stored one", async () => {
    const real = createStore("sqlite://:memory:");
    await real.publishPaper(ID, graph);
    const theirs = { ...(journey as object), intro: "Saved first by someone else." } as AnyArtifact;
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
    const e = await learn({ arxiv_id: ID, kind: "journey" }, deps(racing));
    expect(e).toMatchObject({ type: "learn", saved: true, artifact: theirs });
  });

  test("an artifact that maps to none of the model's blocks is an error and isn't saved", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    next = { ...journey, steps: journey.steps.map((s) => ({ ...s, block_id: "nope" })) };
    const e = await learn({ arxiv_id: ID, kind: "journey" }, deps(store));
    expect(e.type).toBe("error");
    expect(await store.getArtifact(ID, "journey")).toBeNull();
  });

  test("a journey with fewer than 2 usable steps fails and isn't saved", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    next = { ...journey, steps: [journey.steps[0]!, { ...journey.steps[1]!, block_id: "nope" }] };
    const e = await learn({ arxiv_id: ID, kind: "journey" }, deps(store));
    expect(e).toMatchObject({ type: "error", message: "Couldn't build the walkthrough. Try again." });
    expect(await store.getArtifact(ID, "journey")).toBeNull();
  });

  test("a broken database still generates, unsaved", async () => {
    const e = await learn({ arxiv_id: ID, kind: "journey", graph }, deps(broken));
    expect(e).toMatchObject({ type: "learn", source: "claude", saved: false });
  });

  test("a missing key surfaces as needs_key", async () => {
    const e = await learn(
      { arxiv_id: ID, kind: "journey", graph },
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
    expect(await learn({ arxiv_id: ID, kind: "example" }, deps(store))).toMatchObject({ code: "bad_input" }); // retired kind
    expect(await learn({ arxiv_id: ID, kind: "journey" }, deps(store))).toMatchObject({ code: "bad_input" });
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
    await Bun.write(join(samplesDir, "cs_0112017.journey.json"), JSON.stringify(journey));
    const res = await kindsHandler(new Request("http://x/api/learn?arxiv_id=cs/0112017"), deps(store));
    expect(await res.json()).toEqual(["pytorch", "journey"]);
  });
});

describe("samples against a published graph", () => {
  test("a sample that doesn't fit the published graph is not served or listed", async () => {
    const store = createStore("sqlite://:memory:");
    // Someone re-forged and published this paper with different block ids.
    const other = { ...graph, blocks: graph.blocks.map((b) => ({ ...b, id: `x_${b.id}` })), edges: [] };
    await store.publishPaper(ID, other);
    await Bun.write(join(samplesDir, `${ID}.journey.json`), JSON.stringify(journey));
    next = { ...journey, steps: journey.steps.map((st) => ({ ...st, block_id: `x_${st.block_id}` })) };
    const e = await learn({ arxiv_id: ID, kind: "journey" }, deps(store));
    expect(e).toMatchObject({ type: "learn", source: "claude" });
    const kinds = await kindsHandler(new Request(`http://x/api/learn?arxiv_id=${ID}`), deps(store));
    expect(await kinds.json()).toEqual(["journey"]); // now saved from Claude, not the sample
  });

  test("a sample that fits the published graph is still served", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    await Bun.write(join(samplesDir, `${ID}.journey.json`), JSON.stringify(journey));
    const e = await learn({ arxiv_id: ID, kind: "journey" }, deps(store));
    expect(e).toMatchObject({ type: "learn", source: "sample" });
  });
});

describe("POST /api/inside", () => {
  const BLOCK = "encoder_self_attention";
  const inner = { ...graph, model_name: "Multi-Head Attention", blocks: graph.blocks.slice(0, 3), edges: [] };
  let insideCalls: string[];
  const withInside = (store: Store) =>
    deps(store, {
      inside: async (block) => {
        insideCalls.push(block.id);
        return inner;
      },
    });
  const inside = async (body: unknown, d: Partial<LearnDeps>) =>
    (await events<InsideEvent>(await insideHandler(post("/api/inside", body), d))).find((e) => e.type !== "status")!;

  beforeEach(() => (insideCalls = []));

  test("an unpublished paper's inside view comes from Claude and isn't saved", async () => {
    const store = createStore("sqlite://:memory:");
    const e = await inside({ arxiv_id: ID, block_id: BLOCK, graph }, withInside(store));
    expect(e).toMatchObject({ type: "inside", block_id: BLOCK, source: "claude", saved: false, graph: inner });
    expect(await store.getInside(ID, BLOCK)).toBeNull();
  });

  test("a published paper's inside view is saved once, then served with no Claude call", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    expect(await inside({ arxiv_id: ID, block_id: BLOCK }, withInside(store))).toMatchObject({ source: "claude", saved: true });
    expect(await inside({ arxiv_id: ID, block_id: BLOCK }, withInside(store))).toMatchObject({ source: "gallery", graph: inner });
    expect(insideCalls).toEqual([BLOCK]);
  });

  test("a committed sample is served without a key", async () => {
    await Bun.write(join(samplesDir, `${ID}.inside.${BLOCK}.json`), JSON.stringify(inner));
    const e = await inside({ arxiv_id: ID, block_id: BLOCK, graph }, withInside(createStore("sqlite://:memory:")));
    expect(e).toMatchObject({ source: "sample", graph: inner });
    expect(insideCalls).toEqual([]);
  });

  test("unknown or malformed block ids are bad_input", async () => {
    const d = withInside(createStore("sqlite://:memory:"));
    for (const block_id of ["no_such_block", "../../etc", ""]) {
      expect(await inside({ arxiv_id: ID, block_id, graph }, d)).toMatchObject({ type: "error", code: "bad_input" });
    }
    expect(insideCalls).toEqual([]);
  });
});

describe("POST /api/learn, motion", () => {
  const ids = journey.steps.map((s) => s.block_id);
  const linear = { kind: "linear", caption: "Mix", tokens: [], input: [[1, 2]], weights: [[1], [1]], extra: [], fn: "", size: 1, stride: 1, count: 1, svg: "" };
  const motion = { steps: ids.map((block_id) => ({ block_id, ops: [linear] })) } as AnyArtifact;

  test("motion without a journey is bad_input", async () => {
    const e = await learn({ arxiv_id: ID, kind: "motion", graph }, deps(createStore("sqlite://:memory:")));
    expect(e).toMatchObject({ type: "error", code: "bad_input", message: "Open the walkthrough first." });
  });

  test("an unpublished paper's motion is generated from the posted journey", async () => {
    next = motion;
    const e = await learn({ arxiv_id: ID, kind: "motion", graph, journey }, deps(createStore("sqlite://:memory:")));
    expect(e).toMatchObject({ type: "learn", source: "claude", saved: false });
    expect(calls[0]!.journey!.steps.length).toBe(journey.steps.length);
  });

  test("a published paper's motion follows its stored journey, not the posted one", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    await store.putArtifact(ID, "journey", journey);
    next = motion;
    const posted = { ...journey, intro: "Posted by a visitor." };
    await learn({ arxiv_id: ID, kind: "motion", journey: posted }, deps(store));
    expect(calls[0]!.journey!.intro).toBe(journey.intro);
  });

  test("a published paper's motion built from a visitor's posted journey is shown but never saved", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    next = motion;
    const e = await learn({ arxiv_id: ID, kind: "motion", journey }, deps(store));
    expect(e).toMatchObject({ type: "learn", source: "claude", saved: false });
    expect(await store.getArtifact(ID, "motion")).toBeNull();
  });

  test("a motion sample that doesn't fit the published journey isn't served", async () => {
    const store = createStore("sqlite://:memory:");
    const renamed = { ...graph, blocks: graph.blocks.map((b) => ({ ...b, id: `x_${b.id}` })), edges: [] };
    await store.publishPaper(ID, renamed);
    const theirs = { ...journey, steps: journey.steps.map((s) => ({ ...s, block_id: `x_${s.block_id}` })) };
    await store.putArtifact(ID, "journey", theirs);
    await Bun.write(join(samplesDir, `${ID}.motion.json`), JSON.stringify(motion));
    next = { steps: ids.map((id) => ({ block_id: `x_${id}`, ops: [linear] })) } as AnyArtifact;
    const e = await learn({ arxiv_id: ID, kind: "motion" }, deps(store));
    expect(e).toMatchObject({ type: "learn", source: "claude" });
  });

  test("motion covering too little of the walkthrough fails and isn't saved", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    await store.putArtifact(ID, "journey", journey);
    next = { steps: [{ block_id: ids[0]!, ops: [linear] }] } as AnyArtifact;
    const e = await learn({ arxiv_id: ID, kind: "motion" }, deps(store));
    expect(e).toMatchObject({ type: "error", message: "Couldn't animate this model." });
    expect(await store.getArtifact(ID, "motion")).toBeNull();
  });

  test("a stored motion that covers too little is regenerated and replaced (VGG16's 1-of-18)", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    await store.putArtifact(ID, "journey", journey);
    await store.putArtifact(ID, "motion", { steps: [{ block_id: ids[0]!, ops: [linear] }] } as AnyArtifact);
    next = motion;
    const e = await learn({ arxiv_id: ID, kind: "motion" }, deps(store));
    expect(e).toMatchObject({ type: "learn", source: "claude", saved: true });
    expect(((await store.getArtifact(ID, "motion")) as { steps: unknown[] }).steps.length).toBe(ids.length);
  });

  test("zero usable steps fails and isn't saved", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    await store.putArtifact(ID, "journey", journey);
    next = { steps: [{ block_id: "nope", ops: [linear] }] } as AnyArtifact;
    const e = await learn({ arxiv_id: ID, kind: "motion" }, deps(store));
    expect(e).toMatchObject({ type: "error", message: "Couldn't animate this model." });
    expect(await store.getArtifact(ID, "motion")).toBeNull();
  });
});

describe("POST /api/inside-learn", () => {
  const B = "encoder_self_attention";
  const innerIds = innerGraph.blocks.map((b) => b.id);
  // A journey through the inside graph, built from the main sample's step shape.
  const innerJourney: JourneyArtifact = { ...journey, steps: innerIds.map((id, i) => ({ ...journey.steps[i % journey.steps.length]!, block_id: id })) };
  const linear = { kind: "linear", caption: "Mix", tokens: [], input: [[1, 2]], weights: [[1], [1]], extra: [], fn: "", size: 1, stride: 1, count: 1, svg: "" };
  const innerMotion = { steps: innerIds.map((block_id) => ({ block_id, ops: [linear] })) } as AnyArtifact;
  let made: { kind: string; inside: Architecture; journey?: JourneyArtifact }[];
  let produce: AnyArtifact;
  const withLearn = (store: Store) =>
    deps(store, {
      insideLearn: async (kind, o) => {
        made.push({ kind, inside: o.inside, journey: o.journey });
        return produce;
      },
    });
  const ask = async (body: unknown, d: Partial<LearnDeps>) =>
    (await events<InsideLearnEvent>(await insideLearnHandler(post("/api/inside-learn", body), d))).find((e) => e.type !== "status")!;

  beforeEach(() => {
    made = [];
    produce = innerJourney;
  });

  test("serves the committed sample with no key and no Claude call", async () => {
    await Bun.write(join(samplesDir, `${ID}.inside.${B}.json`), JSON.stringify(innerGraph));
    await Bun.write(join(samplesDir, `${ID}.inside.${B}.journey.json`), JSON.stringify(innerJourney));
    const e = await ask({ arxiv_id: ID, block_id: B, kind: "journey", graph }, withLearn(createStore("sqlite://:memory:")));
    expect(e).toMatchObject({ type: "inside_learn", block_id: B, kind: "journey", source: "sample", saved: false });
    expect(made).toEqual([]);
  });

  test("needs the inside graph, and motion needs the walkthrough", async () => {
    const store = createStore("sqlite://:memory:");
    expect(await ask({ arxiv_id: ID, block_id: B, kind: "journey", graph }, withLearn(store))).toMatchObject({ code: "bad_input", message: "Open the block's inside view first." });
    expect(await ask({ arxiv_id: ID, block_id: B, kind: "motion", graph, inside: innerGraph }, withLearn(store))).toMatchObject({ code: "bad_input", message: "Open the walkthrough first." });
  });

  test("built from a posted inside graph: shown, never saved, even for a published paper", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    const e = await ask({ arxiv_id: ID, block_id: B, kind: "journey", inside: innerGraph }, withLearn(store));
    expect(e).toMatchObject({ type: "inside_learn", source: "claude", saved: false });
    expect(made[0]!.inside.blocks.map((b) => b.id)).toEqual(innerIds);
    expect(await store.getInsideLearn(ID, B, "journey")).toBeNull();
  });

  test("a published paper's inside walkthrough and motion are saved, then served from the gallery", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    await store.putInside(ID, B, innerGraph);
    expect(await ask({ arxiv_id: ID, block_id: B, kind: "journey" }, withLearn(store))).toMatchObject({ source: "claude", saved: true });
    produce = innerMotion;
    expect(await ask({ arxiv_id: ID, block_id: B, kind: "motion" }, withLearn(store))).toMatchObject({ source: "claude", saved: true });
    expect(made[1]!.journey!.steps.map((s) => s.block_id)).toEqual(innerIds);
    expect(await ask({ arxiv_id: ID, block_id: B, kind: "journey" }, withLearn(store))).toMatchObject({ source: "gallery" });
    expect(made.length).toBe(2);
  });

  test("a stored journey that doesn't fit the inside graph is regenerated and replaced", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, graph);
    await store.putInside(ID, B, innerGraph);
    await store.putInsideLearn(ID, B, "journey", journey); // the main model's steps, not the inside's
    const e = await ask({ arxiv_id: ID, block_id: B, kind: "journey" }, withLearn(store));
    expect(e).toMatchObject({ source: "claude", saved: true });
    expect(((await store.getInsideLearn(ID, B, "journey")) as JourneyArtifact).steps.map((s) => s.block_id)).toEqual(innerIds);
  });

  test("a journey with fewer than 2 usable steps fails", async () => {
    produce = { ...innerJourney, steps: [innerJourney.steps[0]!] };
    const e = await ask({ arxiv_id: ID, block_id: B, kind: "journey", graph, inside: innerGraph }, withLearn(createStore("sqlite://:memory:")));
    expect(e).toMatchObject({ type: "error", message: "Couldn't build the walkthrough. Try again." });
  });
});
