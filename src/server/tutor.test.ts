import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { askRequest, cleanInside, insideJourneyRequest, insideRequest, learnRequest, motionRequest } from "./tutor";
import type { JourneyArtifact } from "../shared/learn";

const graph = Architecture.parse(await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json());
const pages = ["Page one text.", "Page two text."];
const block = graph.blocks.find((b) => b.id === "encoder_self_attention")!;

const prefix = (content: unknown) => JSON.stringify((content as unknown[]).slice(0, 2));
const firstContent = (req: { messages: { content: unknown }[] }) => req.messages[0]!.content;

describe("cached paper prefix", () => {
  const learn = (["pytorch", "journey"] as const).map((k) => learnRequest(k, pages, graph));
  const inside = insideRequest(block, pages, graph);
  const reqs = [...learn, inside];
  const ask = askRequest({ pages, graph, block, question: "Why divide by sqrt(d_k)?", history: [] });

  test("is byte-identical for every learning kind", () => {
    const first = prefix(firstContent(reqs[0]!));
    for (const r of reqs) expect(prefix(firstContent(r))).toBe(first);
  });

  test("learning kinds share one output format (a format change invalidates the cache)", () => {
    const formats = learn.map((r) => JSON.stringify(r.output_config.format.schema));
    expect(new Set(formats).size).toBe(1);
  });

  test("inside views have their own format: the journey and the block graph don't fit one grammar", () => {
    expect(JSON.stringify(inside.output_config.format.schema)).not.toBe(JSON.stringify(learn[0]!.output_config.format.schema));
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

  test("asks for the reader's language after the cached prefix", () => {
    const es = askRequest({ pages, graph, block, question: "¿Por qué?", history: [], lang: "es" });
    expect(prefix(firstContent(es))).toBe(prefix(firstContent(learnRequest("pytorch", pages, graph))));
    expect(JSON.stringify(es.messages[0]!.content)).toContain("Answer in Spanish");
    expect(JSON.stringify(askRequest({ pages, graph, block, question: "Why?", history: [] }).messages[0]!.content)).not.toContain(
      "whatever language",
    );
  });

  test("asks for the reader's level after the cached prefix", () => {
    const req = askRequest({ pages, graph, block, question: "Why?", history: [], level: "beginner" });
    expect(prefix(firstContent(req))).toBe(prefix(firstContent(learnRequest("pytorch", pages, graph))));
    expect(JSON.stringify(req.messages[0]!.content)).toContain("Write for a curious beginner");
  });
});

describe("cleanInside", () => {
  test("drops duplicate blocks, dangling edges, self-loops and empty groups", () => {
    const [a, b] = graph.blocks;
    const messy = {
      ...graph,
      groups: [{ id: "heads", label: "Head", repeat: 8 }, { id: "unused", label: "x", repeat: 1 }],
      blocks: [{ ...a!, group: "heads" }, b!, { ...b!, label: "dup" }, { ...b!, id: "lonely", group: "gone" }],
      edges: [
        { from: a!.id, to: b!.id, kind: "data" as const },
        { from: b!.id, to: b!.id, kind: "data" as const },
        { from: a!.id, to: "nowhere", kind: "data" as const },
      ],
    };
    const clean = cleanInside(messy);
    expect(clean.blocks.map((x) => x.id)).toEqual([a!.id, b!.id, "lonely"]);
    expect(clean.blocks[2]!.group).toBe("");
    expect(clean.groups.map((g) => g.id)).toEqual(["heads"]);
    expect(clean.edges).toHaveLength(1);
  });
});

describe("motionRequest", () => {
  test("shares the system prompt and paper prefix, with its own format and the journey after the breakpoint", async () => {
    const journey = (await Bun.file(join(import.meta.dir, "../../samples/1706.03762.journey.json")).json()) as JourneyArtifact;
    const motion = motionRequest(pages, graph, journey);
    const learn = learnRequest("journey", pages, graph);
    expect(motion.system).toBe(learn.system);
    expect(prefix(firstContent(motion))).toBe(prefix(firstContent(learn)));
    expect(JSON.stringify(motion.output_config.format.schema)).not.toBe(JSON.stringify(learn.output_config.format.schema));
    expect(motion.output_config.effort).toBe(learn.output_config.effort);
    const task = (firstContent(motion) as { text: string }[])[2]!.text;
    expect(task).toContain("<journey>");
    for (const s of journey.steps) expect(task).toContain(s.block_id);
  });
});

describe("insideJourneyRequest", () => {
  const SAMPLES = join(import.meta.dir, "../../samples");
  const loadInside = async () => Architecture.parse(await Bun.file(join(SAMPLES, "1706.03762.inside.encoder_self_attention.json")).json());

  test("shares the system prompt, paper prefix and format with the main journey", async () => {
    const inside = await loadInside();
    const req = insideJourneyRequest(pages, graph, block, inside, null, null);
    const main = learnRequest("journey", pages, graph);
    expect(req.system).toBe(main.system);
    expect(prefix(firstContent(req))).toBe(prefix(firstContent(main)));
    expect(JSON.stringify(req.output_config.format.schema)).toBe(JSON.stringify(main.output_config.format.schema));
  });

  test("names every inside block and carries what enters and leaves the block", async () => {
    const inside = await loadInside();
    const journey = (await Bun.file(join(SAMPLES, "1706.03762.journey.json")).json()) as JourneyArtifact;
    const at = journey.steps.findIndex((s) => s.block_id === block.id);
    const req = insideJourneyRequest(pages, graph, block, inside, journey.steps[at - 1]!.views, journey.steps[at]!.views);
    const task = (firstContent(req) as { text: string }[])[2]!.text;
    for (const b of inside.blocks) expect(task).toContain(b.id);
    expect(task).toContain("<entering>");
    expect(task).toContain(journey.steps[at]!.views[0]!.caption);
    expect(task).toContain('"journey" shape');
  });

  test("without the main walkthrough it lets Claude choose a small input", async () => {
    const task = (firstContent(insideJourneyRequest(pages, graph, block, await loadInside(), null, null)) as { text: string }[])[2]!.text;
    expect(task).not.toContain("<entering>");
    expect(task).toContain("choose");
  });
});
