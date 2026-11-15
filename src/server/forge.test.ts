import { beforeEach, describe, expect, test } from "bun:test";
import { mkdtempSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Architecture, type ForgeEvent } from "../shared/graph";
import { forge } from "./forge";
import { createStore, type Store } from "./store";
import { verifyGraph } from "./token";

const sample = Architecture.parse(await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json());
const ID = "2401.00001"; // no sample ships for this id

let calls: (string | undefined)[];
let cacheDir: string;
const extract = async (_pages: string[], opts?: { apiKey?: string }) => {
  calls.push(opts?.apiKey);
  return sample;
};
const loadPages = async () => ["page"];

async function run(opts: { store: Store; apiKey?: string; refresh?: boolean }) {
  const events: ForgeEvent[] = [];
  await forge(ID, (e) => events.push(e), { ...opts, extract, loadPages, cacheDir });
  return events.find((e) => e.type === "graph") as Extract<ForgeEvent, { type: "graph" }>;
}

beforeEach(() => {
  calls = [];
  cacheDir = mkdtempSync(join(tmpdir(), "paperforge-test-"));
});

describe("forge lookup order", () => {
  test("a published paper loads from the gallery with no key and no Claude call", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, sample);
    const e = await run({ store });
    expect(e.source).toBe("gallery");
    expect(e.save_token).toBeUndefined();
    expect(calls).toEqual([]);
  });

  test("a broken database falls through to Claude, with a valid save token", async () => {
    const broken: Store = {
      getPaper: () => Promise.reject(new Error("db down")),
      listPapers: () => Promise.reject(new Error("db down")),
      publishPaper: () => Promise.reject(new Error("db down")),
    };
    const e = await run({ store: broken });
    expect(e.source).toBe("claude");
    expect(verifyGraph(e.graph, e.save_token!)).toBe(true);
  });

  test("refresh skips the gallery", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper(ID, sample);
    const e = await run({ store, refresh: true });
    expect(e.source).toBe("claude");
    expect(calls.length).toBe(1);
  });

  test("a second forge of the same paper hits the disk cache", async () => {
    const store = createStore("sqlite://:memory:");
    await run({ store });
    const e = await run({ store });
    expect(e.source).toBe("cache");
    // Still this server's own output: reloading before publishing must not lose the option.
    expect(verifyGraph(e.graph, e.save_token!)).toBe(true);
  });
});

describe("signing secret", () => {
  test("missing in production fails before Claude runs (no paid call wasted)", async () => {
    const saved = { ...process.env };
    process.env.NODE_ENV = "production";
    delete process.env.PAPERFORGE_SECRET;
    try {
      await expect(run({ store: createStore("sqlite://:memory:") })).rejects.toThrow(/PAPERFORGE_SECRET/);
      expect(calls).toEqual([]);
    } finally {
      Object.assign(process.env, saved);
    }
  });
});

describe("api key", () => {
  test("is trimmed before reaching Claude", async () => {
    await run({ store: createStore("sqlite://:memory:"), apiKey: "  sk-test  " });
    expect(calls).toEqual(["sk-test"]);
  });

  test("an empty key means none (fall back to the server env)", async () => {
    await run({ store: createStore("sqlite://:memory:"), apiKey: "   " });
    expect(calls).toEqual([undefined]);
  });
});
