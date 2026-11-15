import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { createStore, storeFromEnv } from "./store";

const sample = Architecture.parse(await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json());
const memory = () => createStore("sqlite://:memory:");

describe("store", () => {
  test("publish then get round-trips the graph", async () => {
    const store = memory();
    expect(await store.publishPaper("1706.03762", sample)).toBe("created");
    expect(await store.getPaper("1706.03762")).toEqual(sample);
  });

  test("first save wins", async () => {
    const store = memory();
    await store.publishPaper("1706.03762", sample);
    expect(await store.publishPaper("1706.03762", { ...sample, model_name: "Imposter" })).toBe("exists");
    expect((await store.getPaper("1706.03762"))?.model_name).toBe(sample.model_name);
  });

  test("unknown paper is null", async () => {
    expect(await memory().getPaper("nope")).toBeNull();
  });

  test("lists newest first without graphs, with a created_at cursor", async () => {
    const store = memory();
    for (const id of ["1111.00001", "1111.00002", "1111.00003"]) {
      await store.publishPaper(id, sample);
      await Bun.sleep(2);
    }
    const list = await store.listPapers();
    expect(list.map((p) => p.arxiv_id)).toEqual(["1111.00003", "1111.00002", "1111.00001"]);
    expect(list[0]!.block_count).toBe(sample.blocks.length);
    expect(list[0]!.model_name).toBe(sample.model_name);
    expect("graph" in list[0]!).toBe(false);
    expect(await store.listPapers({ limit: 1, before: list[0]!.created_at })).toEqual([list[1]!]);
  });

  test("old-style ids with a slash", async () => {
    const store = memory();
    await store.publishPaper("cs/0112017", sample);
    expect(await store.getPaper("cs/0112017")).not.toBeNull();
  });

  test("rejects a graph that doesn't match the schema", async () => {
    await expect(memory().publishPaper("x", { ...sample, blocks: "bad" } as never)).rejects.toThrow();
  });
});

describe("store from env", () => {
  test("on Vercel without DATABASE_URL, the store rejects instead of throwing (EROFS)", async () => {
    const store = storeFromEnv({ VERCEL: "1" });
    await expect(store.getPaper("1706.03762")).rejects.toThrow(/DATABASE_URL/);
    await expect(store.listPapers()).rejects.toThrow(/DATABASE_URL/);
  });

  test("a failed table create is retried on the next call", async () => {
    const dir = join(tmpdir(), `paperforge-retry-${crypto.randomUUID()}`);
    const store = createStore(`sqlite://${join(dir, "g.db")}`);
    await expect(store.listPapers()).rejects.toThrow();
    mkdirSync(dir, { recursive: true });
    expect(await store.listPapers()).toEqual([]);
  });
});
