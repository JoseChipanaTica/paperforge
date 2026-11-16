import Anthropic from "@anthropic-ai/sdk";
import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { errorMessage, graphHandler, listHandler, publishHandler } from "./api";
import { locate } from "./forge";
import { createStore, type Store } from "./store";
import { signGraph } from "./token";

const sample = Architecture.parse(await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json());
const graph = locate("1706.03762", sample, ["page"]);
const token = signGraph(graph);

const publish = (body: unknown, store: Store) =>
  publishHandler(
    new Request("http://x/api/papers", { method: "POST", body: typeof body === "string" ? body : JSON.stringify(body) }),
    store,
  );

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

describe("POST /api/papers", () => {
  test("publishes once, then 409", async () => {
    const store = createStore("sqlite://:memory:");
    const first = await publish({ graph, save_token: token }, store);
    expect(first.status).toBe(201);
    expect(await first.json()).toEqual({ arxiv_id: "1706.03762" });
    const again = await publish({ graph, save_token: token }, store);
    expect(again.status).toBe(409);
    expect(await again.json()).toEqual({ arxiv_id: "1706.03762", error: "Already in the gallery" });
  });

  test("rejects a tampered graph", async () => {
    const res = await publish({ graph: { ...graph, model_name: "Fake" }, save_token: token }, createStore("sqlite://:memory:"));
    expect(res.status).toBe(403);
  });

  test("400 without a token or with bad JSON", async () => {
    const store = createStore("sqlite://:memory:");
    expect((await publish({ graph }, store)).status).toBe(400);
    expect((await publish("{not json", store)).status).toBe(400);
  });

  test("413 for an oversized body", async () => {
    const res = await publish({ graph, save_token: token, pad: "x".repeat(300_000) }, createStore("sqlite://:memory:"));
    expect(res.status).toBe(413);
  });

  test("503 when the database is down", async () => {
    const res = await publish({ graph, save_token: token }, broken);
    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({ error: "Gallery unavailable" });
  });
});

describe("GET /api/papers", () => {
  test("lists published papers", async () => {
    const store = createStore("sqlite://:memory:");
    await publish({ graph, save_token: token }, store);
    const res = await listHandler(new Request("http://x/api/papers?limit=5"), store);
    expect(res.status).toBe(200);
    const list = (await res.json()) as { arxiv_id: string }[];
    expect(list.map((p) => p.arxiv_id)).toEqual(["1706.03762"]);
  });

  test("503 when the database is down", async () => {
    const res = await listHandler(new Request("http://x/api/papers"), broken);
    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({ error: "Gallery unavailable" });
  });
});

describe("errorMessage", () => {
  test("missing key → needs_key", () => {
    const err = new Error('Could not resolve authentication method. Expected either apiKey or authToken to be set.');
    expect(errorMessage(err).code).toBe("needs_key");
  });

  test("rejected key → needs_key", () => {
    const err = new Anthropic.AuthenticationError(401, { type: "error" }, "invalid x-api-key", new Headers());
    expect(errorMessage(err).code).toBe("needs_key");
  });

  test("bad link → bad_input", () => {
    expect(errorMessage(new Error(`That doesn't look like an arXiv link or id: "x"`)).code).toBe("bad_input");
  });
});

describe("GET /api/graph", () => {
  const get = (q: string, store: Store) => graphHandler(new Request(`http://x/api/graph${q}`), store);

  test("returns the published graph first", async () => {
    const store = createStore("sqlite://:memory:");
    await store.publishPaper("1706.03762", { ...sample, model_name: "Published" });
    const res = await get("?arxiv_id=1706.03762", store);
    expect(res.status).toBe(200);
    expect(res.headers.get("cache-control")).toBe("no-store");
    const body = (await res.json()) as { graph: Architecture; source: string };
    expect(body.source).toBe("gallery");
    expect(body.graph.model_name).toBe("Published");
  });

  test("falls back to the committed sample", async () => {
    const res = await get("?arxiv_id=1706.03762", createStore("sqlite://:memory:"));
    expect(res.status).toBe(200);
    expect(((await res.json()) as { source: string }).source).toBe("sample");
  });

  test("404 when neither exists", async () => {
    const res = await get("?arxiv_id=2001.00001", createStore("sqlite://:memory:"));
    expect(res.status).toBe(404);
    expect(res.headers.get("cache-control")).toBe("no-store");
  });

  test("400 on a malformed id", async () => {
    const store = createStore("sqlite://:memory:");
    expect((await get("?arxiv_id=nope", store)).status).toBe(400);
    expect((await get("", store)).status).toBe(400);
  });

  test("a broken gallery still serves the sample", async () => {
    const res = await get("?arxiv_id=1706.03762", broken);
    expect(res.status).toBe(200);
    expect(((await res.json()) as { source: string }).source).toBe("sample");
  });
});
