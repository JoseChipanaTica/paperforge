import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { locate } from "./forge";
import { signGraph, verifyGraph } from "./token";

const sample = Architecture.parse(await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json());
const graph = () => locate("1706.03762", sample, ["page"]);

describe("save token", () => {
  test("verifies a graph it signed", () => {
    const g = graph();
    expect(verifyGraph(g, signGraph(g))).toBe(true);
  });

  test("survives a JSON round trip (what the browser posts back)", () => {
    const g = graph();
    expect(verifyGraph(JSON.parse(JSON.stringify(g)), signGraph(g))).toBe(true);
  });

  test("rejects a tampered block", () => {
    const g = graph();
    const token = signGraph(g);
    g.blocks[0]!.label = "Totally Real Block";
    expect(verifyGraph(g, token)).toBe(false);
  });

  test("rejects the graph re-published under another paper id", () => {
    const g = graph();
    const token = signGraph(g);
    expect(verifyGraph({ ...g, arxiv_id: "1512.03385" }, token)).toBe(false);
  });

  test("rejects malformed tokens without throwing", () => {
    const g = graph();
    expect(verifyGraph(g, "deadbeef")).toBe(false);
    expect(verifyGraph(g, "")).toBe(false);
  });
});
