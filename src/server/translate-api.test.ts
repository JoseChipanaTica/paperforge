import { beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { pickText, type Lang, type Level, type TranslateKind, type TranslatedText } from "../shared/translate";
import { createStore, type Store } from "./store";
import { hashText, translateHandler, type TranslateDeps } from "./translate-api";

const SAMPLES = join(import.meta.dir, "../../samples");
const SAMPLE_ID = "1706.03762";
const graph = Architecture.parse(await Bun.file(join(SAMPLES, "1706.03762.json")).json());
const journey = await Bun.file(join(SAMPLES, "1706.03762.journey.json")).json();

let calls: { kind: TranslateKind; lang: Lang; level: Level; apiKey?: string }[];
let store: Store;
let cacheDir: string;

const spanish = (text: TranslatedText): TranslatedText => ({ ...(text as object), summary: "Un modelo solo de atención." }) as TranslatedText;

function deps(extra: Partial<TranslateDeps> = {}): Partial<TranslateDeps> {
  return {
    store,
    cacheDir,
    samplesDir: SAMPLES,
    translate: async (kind, text, lang, level, apiKey) => {
      calls.push({ kind, lang, level, apiKey });
      return kind === "graph" ? spanish(text) : text;
    },
    ...extra,
  };
}

const translate = (body: unknown, d = deps(), key?: string) =>
  translateHandler(
    new Request("http://x/api/translate", { method: "POST", headers: key ? { "x-anthropic-key": key } : {}, body: JSON.stringify(body) }),
    d,
  );

beforeEach(() => {
  calls = [];
  store = createStore("sqlite://:memory:");
  cacheDir = mkdtempSync(join(tmpdir(), "paperforge-cache-"));
});

describe("POST /api/translate", () => {
  test("translates with the visitor's key, then serves the same translation without Claude", async () => {
    const body = { arxiv_id: "2401.00001", kind: "graph", lang: "es", content: graph };
    const first = await translate(body, deps(), "sk-test");
    expect(first.status).toBe(200);
    expect(await first.json()).toMatchObject({ source: "claude", text: { summary: "Un modelo solo de atención." } });
    expect(calls).toEqual([{ kind: "graph", lang: "es", level: "engineer", apiKey: "sk-test" }]);

    const again = await translate(body);
    expect(await again.json()).toMatchObject({ source: "cache", text: { summary: "Un modelo solo de atención." } });
    expect(calls).toHaveLength(1);
  });

  test("each language is its own translation", async () => {
    await translate({ arxiv_id: "2401.00001", kind: "graph", lang: "es", content: graph });
    await translate({ arxiv_id: "2401.00001", kind: "graph", lang: "fr", content: graph });
    expect(calls.map((c) => c.lang)).toEqual(["es", "fr"]);
  });

  test("a translation of the sample graph goes to the gallery, so other servers reuse it", async () => {
    await translate({ arxiv_id: SAMPLE_ID, kind: "graph", lang: "es", content: graph });
    expect(await store.getTranslation(hashText("graph", pickText("graph", graph)), "es", "engineer", "graph")).not.toBeNull();

    cacheDir = mkdtempSync(join(tmpdir(), "paperforge-cache-")); // another instance: empty disk cache
    const res = await translate({ arxiv_id: SAMPLE_ID, kind: "graph", lang: "es", content: graph });
    expect(await res.json()).toMatchObject({ source: "gallery" });
    expect(calls).toHaveLength(1);
  });

  test("a published artifact's translation goes to the gallery", async () => {
    const id = "2401.00001";
    await store.publishPaper(id, graph);
    await store.putArtifact(id, "journey", journey);
    await translate({ arxiv_id: id, kind: "journey", lang: "ja", content: journey });
    expect(await store.getTranslation(hashText("journey", pickText("journey", journey)), "ja", "engineer", "journey")).not.toBeNull();
  });

  test("text that isn't published or a sample stays out of the gallery", async () => {
    const edited = { ...graph, summary: "Something a visitor made up." };
    const res = await translate({ arxiv_id: SAMPLE_ID, kind: "graph", lang: "es", content: edited });
    expect(res.status).toBe(200);
    expect(await store.getTranslation(hashText("graph", pickText("graph", edited)), "es", "engineer", "graph")).toBeNull();
  });

  test("still translates when the gallery is down", async () => {
    const fail = () => Promise.reject(new Error("db down"));
    store = { ...createStore("sqlite://:memory:"), getPaper: fail, getArtifact: fail, getTranslation: fail, putTranslation: fail };
    const res = await translate({ arxiv_id: SAMPLE_ID, kind: "graph", lang: "es", content: graph });
    expect(await res.json()).toMatchObject({ source: "claude" });
  });

  test("each reading level is its own rewrite, English included", async () => {
    const body = { arxiv_id: "2401.00001", kind: "graph", content: graph };
    for (const [lang, level] of [["en", "beginner"], ["en", "researcher"], ["es", "beginner"], ["es", "engineer"]]) {
      expect((await translate({ ...body, lang, level })).status).toBe(200);
    }
    await translate({ ...body, lang: "en", level: "beginner" }); // cached
    expect(calls.map((c) => `${c.lang}/${c.level}`)).toEqual(["en/beginner", "en/researcher", "es/beginner", "es/engineer"]);
  });

  test("a published inside view's translation goes to the gallery", async () => {
    const id = "2401.00001";
    const inside = { ...graph, model_name: "Multi-Head Attention" };
    await store.putInside(id, "encoder_self_attention", inside);
    const body = { arxiv_id: id, kind: "expansion", block_id: "encoder_self_attention", lang: "es", content: inside };
    expect((await translate(body)).status).toBe(200);
    expect(await store.getTranslation(hashText("expansion", pickText("expansion", inside)), "es", "engineer", "expansion")).not.toBeNull();
    expect((await translate({ ...body, block_id: "../x" })).status).toBe(400);
  });

  test("rejects English, unknown languages, unknown kinds and content that isn't a graph", async () => {
    for (const body of [
      { arxiv_id: SAMPLE_ID, kind: "graph", lang: "en", content: graph },
      { arxiv_id: SAMPLE_ID, kind: "graph", lang: "en", level: "engineer", content: graph },
      { arxiv_id: SAMPLE_ID, kind: "graph", lang: "es", level: "expert", content: graph },
      { arxiv_id: SAMPLE_ID, kind: "graph", lang: "xx", content: graph },
      { arxiv_id: SAMPLE_ID, kind: "poem", lang: "es", content: graph },
      { arxiv_id: SAMPLE_ID, kind: "graph", lang: "es", content: { blocks: [] } },
      { arxiv_id: "nope", kind: "graph", lang: "es", content: graph },
    ]) {
      expect((await translate(body)).status).toBe(400);
    }
    expect(calls).toHaveLength(0);
  });

  test("asks for a key when Claude has none", async () => {
    const res = await translate(
      { arxiv_id: SAMPLE_ID, kind: "graph", lang: "es", content: graph },
      deps({ translate: () => Promise.reject(new Error("Could not resolve authentication method. Expected either apiKey or authToken")) }),
    );
    expect(res.status).toBe(401);
    expect(await res.json()).toMatchObject({ code: "needs_key" });
  });
});
