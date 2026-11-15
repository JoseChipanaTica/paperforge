import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { askRequest, learnRequest } from "./tutor";

const graph = Architecture.parse(await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json());
const pages = ["Page one text.", "Page two text."];
const block = graph.blocks.find((b) => b.id === "encoder_self_attention")!;

const prefix = (content: unknown) => JSON.stringify((content as unknown[]).slice(0, 2));
const firstContent = (req: { messages: { content: unknown }[] }) => req.messages[0]!.content;

describe("cached paper prefix", () => {
  const reqs = (["pytorch", "walkthrough", "example"] as const).map((k) => learnRequest(k, pages, graph));
  const ask = askRequest({ pages, graph, block, question: "Why divide by sqrt(d_k)?", history: [] });

  test("is byte-identical for every learning kind", () => {
    const first = prefix(firstContent(reqs[0]!));
    for (const r of reqs) expect(prefix(firstContent(r))).toBe(first);
  });

  test("sends the same output format for every kind (a format change invalidates the cache)", () => {
    const formats = reqs.map((r) => JSON.stringify(r.output_config.format.schema));
    expect(new Set(formats).size).toBe(1);
  });

  test("ends in a cache breakpoint on the graph block", () => {
    const blocks = firstContent(reqs[0]!) as { text: string; cache_control?: unknown }[];
    expect(blocks[0]!.text).toContain('<page number="1">');
    expect(blocks[1]!.text).toContain("<graph>");
    expect(blocks[1]!.cache_control).toEqual({ type: "ephemeral" });
  });

  test("ask shares the prefix, system prompt and effort", () => {
    expect(prefix(firstContent(ask))).toBe(prefix(firstContent(reqs[0]!)));
    for (const r of reqs) {
      expect(r.system).toBe(ask.system);
      expect(r.output_config.effort).toBe("medium");
    }
    expect(ask.output_config.effort).toBe("medium");
  });
});

describe("askRequest", () => {
  test("puts the paper + focused block first, then history, then the question", () => {
    const req = askRequest({
      pages,
      graph,
      block,
      question: "And the masking?",
      history: [
        { role: "user", text: "What is a head?" },
        { role: "assistant", text: "One of 8 parallel attention functions." },
      ],
    });
    expect(req.messages.map((m) => m.role)).toEqual(["user", "user", "assistant", "user"]);
    expect(JSON.stringify(req.messages[0]!.content)).toContain("encoder_self_attention");
    expect(req.messages.at(-1)!.content).toBe("And the masking?");
  });
});
