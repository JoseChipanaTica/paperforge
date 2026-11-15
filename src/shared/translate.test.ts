import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "./graph";
import { ARTIFACT_SCHEMA, LEARN_KINDS, type JourneyArtifact, type PytorchArtifact } from "./learn";
import { TEXT_SCHEMA, applyText, codeSkeleton, pickText, textMatches, type TranslatedText } from "./translate";

const SAMPLES = join(import.meta.dir, "../../samples");
const graph = Architecture.parse(await Bun.file(join(SAMPLES, "1706.03762.json")).json());
const sample = async (kind: (typeof LEARN_KINDS)[number]) =>
  ARTIFACT_SCHEMA[kind].parse(await Bun.file(join(SAMPLES, `1706.03762.${kind}.json`)).json());

/** A fake translation: every string value except ids gets a prefix. */
function fake(value: unknown, key = ""): any {
  if (typeof value === "string") return key === "id" || key === "block_id" ? value : `ES ${value}`;
  if (Array.isArray(value)) return value.map((v) => fake(v));
  if (value && typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, fake(v, k)]));
  return value;
}

describe("graph text", () => {
  const text = pickText("graph", graph);
  const translated = applyText("graph", graph, fake(text));

  test("is only the prose, and fits its schema", () => {
    expect(TEXT_SCHEMA.graph.safeParse(text).success).toBe(true);
    expect(JSON.stringify(text)).not.toContain(graph.blocks[0]!.source_quote);
  });

  test("replaces labels, descriptions and the summary, nothing else", () => {
    expect(translated.summary).toBe(`ES ${graph.summary}`);
    translated.blocks.forEach((b, i) => {
      const o = graph.blocks[i]!;
      expect(b).toEqual({ ...o, label: `ES ${o.label}`, description: `ES ${o.description}` });
    });
    expect(translated.groups.map((g) => g.label)).toEqual(graph.groups.map((g) => `ES ${g.label}`));
    expect(translated.edges).toEqual(graph.edges);
    expect(translated.paper_title).toBe(graph.paper_title);
  });

  test("keeps fields the schema doesn't know (located source spans)", () => {
    const located = { ...graph, arxiv_id: "1706.03762", blocks: graph.blocks.map((b) => ({ ...b, source: null })) };
    expect(applyText("graph", located, fake(text)).blocks.every((b) => b.source === null)).toBe(true);
  });

  test("a translation with different blocks doesn't match", () => {
    const t = fake(text);
    expect(textMatches("graph", text, t)).toBe(true);
    expect(textMatches("graph", text, { ...t, blocks: t.blocks.slice(1) })).toBe(false);
  });
});

describe("artifact text", () => {
  for (const kind of LEARN_KINDS) {
    test(`${kind}: round-trips and keeps block links`, async () => {
      const a = await sample(kind);
      const text = pickText(kind, a);
      expect(TEXT_SCHEMA[kind].safeParse(text).success).toBe(true);
      const t: TranslatedText = fake(text);
      expect(textMatches(kind, text, t)).toBe(true);
      const out = applyText(kind, a, t) as any;
      expect(ARTIFACT_SCHEMA[kind].safeParse(out).success).toBe(true);
      if (kind === "pytorch") expect(out.block_map).toEqual((a as PytorchArtifact).block_map);
      else expect(out.steps.map((s: any) => s.block_id)).toEqual((a as any).steps.map((s: any) => s.block_id));
    });
  }

  test("journey: only prose is sent, and the data never changes", async () => {
    const a = (await sample("journey")) as JourneyArtifact;
    const text = pickText("journey", a);
    const sent = JSON.stringify(text);
    for (const field of ['"values"', '"rows"', '"cols"', '"dims"', '"split"', '"shape_change"', '"modality"']) expect(sent).not.toContain(field);
    const out = applyText("journey", a, fake(text));
    const data = (j: JourneyArtifact) =>
      [j.input, ...j.steps, j.output].map((m) => [
        "shape_change" in m ? m.shape_change : "",
        m.views.map(({ kind, rows, cols, values, dims, split }) => ({ kind, rows, cols, values, dims, split })),
      ]);
    expect(data(out)).toEqual(data(a));
    expect(out.modality).toBe(a.modality);
    expect(out.steps[0]!.views[0]!.caption).toBe(`ES ${a.steps[0]!.views[0]!.caption}`);
    expect(out.input.label).toBe(`ES ${a.input.label}`);
  });

  test("journey: a different number of captions doesn't match", async () => {
    const text = pickText("journey", await sample("journey")) as any;
    const t = fake(text);
    t.steps[0].captions = [...t.steps[0].captions, "extra"];
    expect(textMatches("journey", text, t)).toBe(false);
  });

  test("steps out of order don't match", async () => {
    const text = pickText("journey", await sample("journey")) as any;
    expect(textMatches("journey", text, { ...text, steps: [...text.steps].reverse() })).toBe(false);
  });
});

describe("codeSkeleton", () => {
  const code = `class Attn(nn.Module):\n    """Scaled dot-product attention."""\n    def forward(self, x):  # x: [B, n, d]\n        return x  # unchanged\n\nprint("# of heads:", 8)`;

  test("ignores comments and docstrings", () => {
    const es = code.replace("Scaled dot-product attention.", "Atención escalada.").replace("# unchanged", "# sin cambios");
    expect(codeSkeleton(es)).toBe(codeSkeleton(code));
  });

  test("notices changed code, including string literals", () => {
    expect(codeSkeleton(code.replace("return x", "return x * 2"))).not.toBe(codeSkeleton(code));
    expect(codeSkeleton(code.replace('"# of heads:"', '"# de cabezas:"'))).not.toBe(codeSkeleton(code));
  });
});
