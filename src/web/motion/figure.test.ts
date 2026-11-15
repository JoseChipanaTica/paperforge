import { describe, expect, test } from "bun:test";
import type { MotionArtifact, Op } from "../../shared/motion";
import { figureState } from "./figure";

const op: Op = { kind: "softmax", caption: "c", tokens: [], input: [[1, 2]], weights: [], extra: [], fn: "", size: 1, stride: 1, count: 1, svg: "" };
const motion: MotionArtifact = { steps: [{ block_id: "a", ops: [op] }] };

describe("figureState", () => {
  test("loading until motion arrives", () => {
    expect(figureState(null, "loading", "a")).toEqual({ kind: "loading" });
  });

  test("a step motion covers gets its ops", () => {
    expect(figureState(motion, "ready", "a")).toEqual({ kind: "ops", ops: [op] });
  });

  test("a step motion lacks shows nothing", () => {
    expect(figureState(motion, "ready", "b")).toEqual({ kind: "none" });
  });

  test("errors and a missing key pass through", () => {
    expect(figureState(null, "error", "a")).toEqual({ kind: "error" });
    expect(figureState(null, "needs_key", "a")).toEqual({ kind: "needs_key" });
  });
});
