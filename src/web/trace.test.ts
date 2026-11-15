import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import { aliasesOf, dimTokens, shapeHtml, traceDim } from "./trace";

const graph = Architecture.parse(await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json());

describe("dimTokens", () => {
  test("reads symbolic and numeric dimensions, including joined shapes", () => {
    expect(dimTokens("[B, m, 512] + [B, n, 512]")).toEqual(["B", "m", "512", "B", "n", "512"]);
    expect(dimTokens("[B, n, d_model]")).toEqual(["B", "n", "d_model"]);
  });
});

describe("shapeHtml", () => {
  test("turns each dimension into a button and keeps the punctuation", () => {
    const html = shapeHtml("[B, n, 512]", "512");
    expect(html.match(/<button/g)).toHaveLength(3);
    expect(html).toContain('data-dim="512" aria-pressed="true"');
    expect(html.replace(/<[^>]+>/g, "")).toBe("[B, n, 512]");
  });
});

describe("traceDim on the Transformer", () => {
  test("512 is d_model, and splits into 8 heads of 64", () => {
    expect(aliasesOf(graph, "512").has("d_model")).toBe(true);
    const t = traceDim(graph, "512");
    expect(t.aliases).toContain("d_model");
    expect(t.splits).toContain("h × d_k = 8 × 64");
  });

  test("a dimension never splits into one hyperparameter twice", () => {
    expect(traceDim(graph, "64").splits).toEqual([]);
  });

  test("512 is created by the embeddings, flows through the stacks, and is consumed by the final linear", () => {
    const t = traceDim(graph, "512");
    expect(t.blocks.get("input_embedding")).toBe("created");
    expect(t.blocks.get("encoder_self_attention")).toBe("through");
    expect(t.blocks.get("final_linear")).toBe("consumed");
    expect(t.blocks.has("inputs")).toBe(false);
  });

  test("V (vocabulary) appears only at the output", () => {
    const t = traceDim(graph, "V");
    expect([...t.blocks]).toEqual([
      ["final_linear", "created"],
      ["softmax", "through"],
    ]);
  });

  test("n stays in the encoder until cross-attention reads it", () => {
    const t = traceDim(graph, "n");
    expect(t.blocks.get("inputs")).toBe("through");
    expect(t.blocks.get("decoder_cross_attention")).toBe("consumed");
    expect(t.blocks.has("decoder_masked_attention")).toBe(false);
  });

  test("a hyperparameter that never appears in a shape lights the blocks that use it", () => {
    const t = traceDim(graph, "d_ff");
    expect(t.blocks.get("encoder_ffn")).toBe("param");
    expect(t.blocks.get("decoder_ffn")).toBe("param");
  });
});
