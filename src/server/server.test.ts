import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { parseArxivId } from "./arxiv";
import { locateQuote } from "./locate";

describe("parseArxivId", () => {
  test.each([
    ["https://arxiv.org/abs/1706.03762", "1706.03762"],
    ["https://arxiv.org/abs/1706.03762v7", "1706.03762"],
    ["https://arxiv.org/pdf/1706.03762", "1706.03762"],
    ["https://arxiv.org/pdf/1706.03762v5.pdf", "1706.03762"],
    ["arxiv.org/abs/2401.12345?context=cs", "2401.12345"],
    ["arXiv:1512.03385", "1512.03385"],
    ["1512.03385", "1512.03385"],
    ["  https://www.alphaxiv.org/abs/2010.11929  ", "2010.11929"],
    ["https://arxiv.org/abs/cs/0112017v1", "cs/0112017"],
  ])("%s → %s", (input, id) => {
    expect(parseArxivId(input)).toBe(id);
  });

  test("rejects non-arXiv input", () => {
    expect(parseArxivId("https://example.com/paper")).toBeNull();
    expect(parseArxivId("")).toBeNull();
  });
});

describe("locateQuote", () => {
  const pages = [
    "Intro text.",
    "That is, the output of each sub-layer is\nLayerNorm( x + Sublayer( x )) , where it is position-\nwise and d model = 512 .",
  ];

  test("matches across pdf.js spacing and line-break hyphens", () => {
    const span = locateQuote(pages, "the output of each sub-layer is LayerNorm(x + Sublayer(x)), where it is position-wise");
    expect(span?.page).toBe(1);
    expect(pages[1]!.slice(span!.start, span!.end)).toStartWith("the output");
    expect(pages[1]!.slice(span!.start, span!.end)).toEndWith("wise");
  });

  test("falls back to head/tail anchors when the middle is garbled", () => {
    const quote = "That is, the output of each sub-layer is LayerNorm[[garbled equation here]] where it is position-wise and d_model = 512.";
    expect(locateQuote(pages, quote)?.page).toBe(1);
  });

  test("returns null when the quote is not in the paper", () => {
    expect(locateQuote(pages, "Convolutions are all you need for image recognition at scale.")).toBeNull();
  });
});

const sample = Architecture.parse(
  await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json(),
);
// Real extracted text, present once the PDF has been cached by a forge run.
const pagesFile = Bun.file(join(import.meta.dir, "../../data/cache/1706.03762.pages.json"));
const havePages = await pagesFile.exists();

describe("samples", () => {
  test("edges reference existing blocks and groups exist", () => {
    const ids = new Set(sample.blocks.map((b) => b.id));
    expect(ids.size).toBe(sample.blocks.length);
    for (const e of sample.edges) {
      expect(ids.has(e.from)).toBe(true);
      expect(ids.has(e.to)).toBe(true);
    }
    const groups = new Set(sample.groups.map((g) => g.id));
    for (const b of sample.blocks) if (b.group) expect(groups.has(b.group)).toBe(true);
  });

  test.skipIf(!havePages)("every source quote is found in the paper", async () => {
    const pages = (await pagesFile.json()) as string[];
    const missing = sample.blocks.filter((b) => !locateQuote(pages, b.source_quote)).map((b) => b.id);
    expect(missing).toEqual([]);
  });
});
