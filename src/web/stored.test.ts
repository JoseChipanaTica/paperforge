import { afterEach, describe, expect, test } from "bun:test";
import { join } from "node:path";
import type { Architecture } from "../shared/graph";
import { applyText, pickText, type TranslatedText } from "../shared/translate";
import { translate, type TranslationStore } from "./lang";
import { textHash } from "./localStore";
import { artifactSource, insideJourneyFits, preferServer, withStored, withStoredArtifact } from "./stored";
import { blockEnds, type JourneyArtifact } from "../shared/learn";

const graph = (await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json()) as Architecture;
const main = (await Bun.file(join(import.meta.dir, "../../samples/1706.03762.journey.json")).json()) as JourneyArtifact;
const ffn = (await Bun.file(join(import.meta.dir, "../../samples/1706.03762.inside.encoder_ffn.journey.json")).json()) as JourneyArtifact;
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

describe("preferServer", () => {
  test("a kind the server lists is asked of it (free) unless the stored copy is the visitor's own", () => {
    expect(preferServer(true, null)).toBe(true);
    expect(preferServer(true, "gallery")).toBe(true);
    expect(preferServer(true, "sample")).toBe(true);
    expect(preferServer(true, "claude")).toBe(false);
    expect(preferServer(false, "gallery")).toBe(false);
    expect(preferServer(false, null)).toBe(false);
  });

  test("artifactSource: anything but gallery or sample is the visitor's own", () => {
    expect(artifactSource("gallery")).toBe("gallery");
    expect(artifactSource("sample")).toBe("sample");
    expect(artifactSource("claude")).toBe("claude");
    expect(artifactSource("local")).toBe("claude");
    expect(artifactSource(undefined)).toBe("claude");
  });
});

describe("withStoredArtifact", () => {
  type Put = [string, string];
  const run = (o: {
    stored: { artifact: string; source: "gallery" | "sample" | "claude" } | null;
    listed: boolean;
    shared?: () => Promise<{ value: string; source: string }>;
  }) => {
    const calls = { shared: 0, fetch: 0, puts: [] as Put[] };
    const p = withStoredArtifact({
      get: async () => o.stored,
      listed: o.listed,
      shared: () => (calls.shared++, (o.shared ?? (async () => ({ value: "server", source: "gallery" })))()),
      fetch: async () => (calls.fetch++, { value: "fetched", source: "claude" }),
      put: async (v, source) => void calls.puts.push([v, source]),
    });
    return { p, calls };
  };

  test("the visitor's own stored copy is used with no request", async () => {
    const { p, calls } = run({ stored: { artifact: "mine", source: "claude" }, listed: true });
    expect(await p).toEqual({ value: "mine", local: true });
    await tick();
    expect(calls).toEqual({ shared: 0, fetch: 0, puts: [] });
  });

  test("a shared stored copy is refreshed from the server when it lists the kind, and overwritten", async () => {
    const { p, calls } = run({ stored: { artifact: "old", source: "gallery" }, listed: true });
    expect(await p).toEqual({ value: "server", local: false });
    await tick();
    expect(calls.shared).toBe(1);
    expect(calls.fetch).toBe(0);
    expect(calls.puts).toEqual([["server", "gallery"]]);
  });

  test("offline or failing, the shared stored copy is used", async () => {
    const { p, calls } = run({
      stored: { artifact: "old", source: "sample" },
      listed: true,
      shared: () => Promise.reject(new Error("offline")),
    });
    expect(await p).toEqual({ value: "old", local: true });
    await tick();
    expect(calls.fetch).toBe(0);
    expect(calls.puts).toEqual([]);
  });

  test("a shared stored copy the server doesn't list is used as is", async () => {
    const { p, calls } = run({ stored: { artifact: "old", source: "gallery" }, listed: false });
    expect(await p).toEqual({ value: "old", local: true });
    expect(calls.shared).toBe(0);
  });

  test("a miss fetches and stores with the source that came back", async () => {
    const { p, calls } = run({ stored: null, listed: true });
    expect(await p).toEqual({ value: "fetched", local: false });
    await tick();
    expect(calls.shared).toBe(0);
    expect(calls.puts).toEqual([["fetched", "claude"]]);
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

describe("insideJourneyFits", () => {
  const ends = blockEnds(main, "encoder_ffn");

  test("a block walkthrough that starts from the main example fits", () => {
    expect(insideJourneyFits(ffn, ends)).toBe(true);
  });

  test("with no main walkthrough on screen there is nothing to check against", () => {
    expect(insideJourneyFits(ffn, { entering: null, leaving: null })).toBe(true);
  });

  test("one that starts from another example doesn't", () => {
    const other = {
      ...ffn,
      input: { ...ffn.input, views: ffn.input.views.map((v) => ({ ...v, rows: v.rows.map((r) => r + "!") })) },
    };
    expect(insideJourneyFits(other, ends)).toBe(false);
  });
});
