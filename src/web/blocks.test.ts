import { expect, test } from "bun:test";
import { BLOCK_TYPES } from "../shared/graph";
import { HIGHLIGHTER, INK, TYPE_COLOR, VERMILION } from "./blocks";

// The printed palette from docs/superpowers/specs/2026-11-15-ui-overhaul-design.md.
const SPEC: Record<(typeof BLOCK_TYPES)[number], string> = {
  attention: "#c2410c",
  multi_head_attention: "#c2410c",
  cross_attention: "#d9733b",
  feed_forward: "#1e3a5f",
  add_norm: "#e0b04c",
  layer_norm: "#e0b04c",
  normalization: "#e0b04c",
  embedding: "#e8dcc4",
  positional_encoding: "#7c8f6a",
  linear: "#4b5d73",
  softmax: "#b8860b",
  convolution: "#2f6f6a",
  pooling: "#8a6f4e",
  recurrent: "#6d4c6f",
  activation: "#a3533c",
  dropout: "#b9ae98",
  residual_add: "#9c8f78",
  concat: "#9c8f78",
  input: "#cfc5b0",
  output: "#cfc5b0",
  other: "#8f8a80",
};

test("every block type uses the printed palette", () => {
  for (const t of BLOCK_TYPES) expect(`${t}: ${TYPE_COLOR[t]}`).toBe(`${t}: ${SPEC[t]}`);
});

test("ink, highlighter and vermilion match the design tokens", () => {
  expect([INK, HIGHLIGHTER, VERMILION]).toEqual(["#1e1d1a", "#f6d77a", "#c2410c"]);
});
