import { afterEach, describe, expect, test } from "bun:test";
import { join } from "node:path";
import type { Architecture } from "../shared/graph";
import { applyText, pickText, type TranslatedText } from "../shared/translate";
import { translate, type TranslationStore } from "./lang";
import { textHash } from "./localStore";
import { withStored } from "./stored";

const graph = (await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json()) as Architecture;
const tick = () => new Promise((r) => setTimeout(r, 0));

describe("withStored", () => {
  test("a hit is used with no fetch and is not saved again", async () => {
    let fetched = 0;
    let put = 0;
    const r = await withStored(
      async () => "stored",
      async () => (fetched++, "fresh"),
      async () => void put++,
    );
    expect(r).toEqual({ value: "stored", local: true });
    await tick();
    expect(fetched).toBe(0);
    expect(put).toBe(0);
  });

  test("a miss fetches, then stores what came back", async () => {
    const puts: string[] = [];
    const r = await withStored(
      async () => null,
      async () => "fresh",
      async (v) => void puts.push(v),
    );
    expect(r).toEqual({ value: "fresh", local: false });
    await tick();
    expect(puts).toEqual(["fresh"]);
  });

  test("a failing read counts as a miss", async () => {
    const r = await withStored(
      () => Promise.reject(new Error("idb")),
      async () => "fresh",
      async () => {},
    );
    expect(r).toEqual({ value: "fresh", local: false });
  });

  test("a failing put still returns the fetched value", async () => {
    const rejecting = await withStored(
      async () => null,
      async () => "fresh",
      () => Promise.reject(new Error("quota")),
    );
    expect(rejecting.value).toBe("fresh");
    const throwing = await withStored(
      async () => null,
      async () => "fresh",
      () => {
        throw new Error("sync");
      },
    );
    expect(throwing.value).toBe("fresh");
  });

  test("the result never waits on the put", async () => {
    let release!: () => void;
    const r = await withStored(
      async () => null,
      async () => "fresh",
      () => new Promise<void>((res) => (release = res)),
    );
    expect(r.value).toBe("fresh");
    release();
  });

  test("a failing fetch puts nothing", async () => {
    let put = 0;
    const p = withStored(
      async () => null,
      () => Promise.reject(new Error("offline")),
      async () => void put++,
    );
    await expect(p).rejects.toThrow("offline");
    await tick();
    expect(put).toBe(0);
  });
});

describe("translate with the browser store", () => {
  const text = pickText("graph", graph) as { summary: string };
  const french = { ...text, summary: "Un résumé." } as TranslatedText;
  const realFetch = globalThis.fetch;
  let calls = 0;
  afterEach(() => {
    globalThis.fetch = realFetch;
    calls = 0;
  });
  const stubFetch = () => {
    globalThis.fetch = (async () => {
      calls++;
      return new Response(JSON.stringify({ text: french, source: "claude" }), { status: 200 });
    }) as unknown as typeof fetch;
  };
  const memory = () => {
    const m = new Map<string, TranslatedText>();
    const store: TranslationStore = {
      get: async (h, lang, level) => m.get(`${h}|${lang}|${level}`) ?? null,
      put: async (h, lang, level, _kind, t) => void m.set(`${h}|${lang}|${level}`, t),
    };
    return { m, store };
  };

  test("a stored rewrite is used with no request", async () => {
    stubFetch();
    const { m, store } = memory();
    m.set(`${await textHash("graph", text as TranslatedText)}|fr|engineer`, french);
    let requested = false;
    const out = await translate("1706.03762", "graph", { lang: "fr", level: "engineer" }, graph, "", undefined, {
      store,
      onRequest: () => (requested = true),
    });
    expect(calls).toBe(0);
    expect(requested).toBe(false);
    expect(out).toEqual(applyText("graph", graph, french));
  });

  test("a miss requests the rewrite, then stores it", async () => {
    stubFetch();
    const { m, store } = memory();
    let requested = false;
    const out = await translate("1706.03762", "graph", { lang: "fr", level: "engineer" }, graph, "", undefined, {
      store,
      onRequest: () => (requested = true),
    });
    expect(calls).toBe(1);
    expect(requested).toBe(true);
    expect(out.summary).toBe("Un résumé.");
    await tick();
    expect(m.get(`${await textHash("graph", text as TranslatedText)}|fr|engineer`)).toEqual(french);
  });

  test("English for an engineer never touches the store", async () => {
    stubFetch();
    let touched = false;
    const store: TranslationStore = {
      get: async () => ((touched = true), null),
      put: async () => void (touched = true),
    };
    const out = await translate("1706.03762", "graph", { lang: "en", level: "engineer" }, graph, "", undefined, { store });
    expect(out).toBe(graph);
    expect(touched).toBe(false);
    expect(calls).toBe(0);
  });
});
