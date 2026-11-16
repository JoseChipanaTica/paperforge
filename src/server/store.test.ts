import { SQL } from "bun";
import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { pickText } from "../shared/translate";
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

describe("artifacts", () => {
  const load = (kind: string) => Bun.file(join(import.meta.dir, `../../samples/1706.03762.${kind}.json`)).json();

  test("put then get round-trips; first save wins", async () => {
    const store = memory();
    const w = await load("journey");
    expect(await store.putArtifact("1706.03762", "journey", w)).toBe("created");
    expect(await store.getArtifact("1706.03762", "journey")).toEqual(w);
    expect(await store.putArtifact("1706.03762", "journey", { ...w, intro: "Imposter" })).toBe("exists");
    expect(await store.getArtifact("1706.03762", "journey")).toEqual(w);
  });

  test("replace overwrites a saved artifact (only for one found unusable)", async () => {
    const store = memory();
    const w = await load("journey");
    await store.putArtifact("1706.03762", "journey", w);
    const other = { ...w, intro: "Better" };
    expect(await store.putArtifact("1706.03762", "journey", other, { replace: true })).toBe("created");
    expect(((await store.getArtifact("1706.03762", "journey")) as { intro: string }).intro).toBe("Better");
  });

  test("missing artifact is null", async () => {
    expect(await memory().getArtifact("1706.03762", "journey")).toBeNull();
  });

  test("lists the kinds saved for a paper", async () => {
    const store = memory();
    await store.putArtifact("1706.03762", "journey", await load("journey"));
    await store.putArtifact("1706.03762", "pytorch", await load("pytorch"));
    await store.putArtifact("1512.03385", "journey", await load("journey"));
    expect((await store.listArtifactKinds("1706.03762")).sort()).toEqual(["journey", "pytorch"]);
  });

  test("rejects an artifact that doesn't match its kind's schema", async () => {
    await expect(memory().putArtifact("x", "journey", { nope: 1 } as never)).rejects.toThrow();
  });
});

describe("inside_learn", () => {
  const load = (kind: string) => Bun.file(join(import.meta.dir, `../../samples/1706.03762.${kind}.json`)).json();

  test("round-trips per block and kind; first save wins; replace overwrites", async () => {
    const store = memory();
    const j = await load("journey");
    expect(await store.getInsideLearn("1706.03762", "encoder_ffn", "journey")).toBeNull();
    expect(await store.putInsideLearn("1706.03762", "encoder_ffn", "journey", j)).toBe("created");
    expect(await store.getInsideLearn("1706.03762", "encoder_ffn", "journey")).toEqual(j);
    expect(await store.getInsideLearn("1706.03762", "encoder_self_attention", "journey")).toBeNull();
    expect(await store.getInsideLearn("1706.03762", "encoder_ffn", "motion")).toBeNull();
    expect(await store.putInsideLearn("1706.03762", "encoder_ffn", "journey", { ...j, intro: "Imposter" })).toBe("exists");
    expect(await store.getInsideLearn("1706.03762", "encoder_ffn", "journey")).toEqual(j);
    await store.putInsideLearn("1706.03762", "encoder_ffn", "journey", { ...j, intro: "Better" }, { replace: true });
    expect(((await store.getInsideLearn("1706.03762", "encoder_ffn", "journey")) as { intro: string }).intro).toBe("Better");
  });
});

describe("translations table from before reading levels", () => {
  test("gains a level column (old rows become engineer) and the wider primary key", async () => {
    const dir = join(tmpdir(), `paperforge-levels-${crypto.randomUUID()}`);
    mkdirSync(dir, { recursive: true });
    const url = `sqlite://${join(dir, "g.db")}`;
    const old = new SQL(url);
    await old.unsafe(`create table translations (
      source_hash text not null, lang text not null, kind text not null, arxiv_id text not null,
      content text not null, created_at text not null, primary key (source_hash, lang))`);
    const text = pickText("graph", sample);
    await old`insert into translations values ('h1', 'es', 'graph', '1706.03762', ${JSON.stringify(text)}, '2026-11-01')`;
    await old.close();

    const store = createStore(url);
    expect(await store.getTranslation("h1", "es", "engineer", "graph")).toEqual(text);
    expect(await store.putTranslation("h1", "es", "beginner", "graph", "1706.03762", text)).toBe("created");
    expect(await store.getTranslation("h1", "es", "beginner", "graph")).toEqual(text);
  });
});
