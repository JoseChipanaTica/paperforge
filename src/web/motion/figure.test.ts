import { describe, expect, test } from "bun:test";
import type { MotionArtifact, Op } from "../../shared/motion";
import { FigureSlot, figureState } from "./figure";

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

describe("FigureSlot", () => {
  const settled = async (p: Promise<void>) => {
    let done = false;
    void p.then(() => (done = true));
    for (let i = 0; i < 5; i++) await Promise.resolve();
    return done;
  };

  test("what Play waits on is set at once and covers the whole run, awaits included", async () => {
    const slot = new FigureSlot();
    let finish!: () => void;
    slot.set(async () => {
      await Promise.resolve(); // e.g. localizing the captions
      await new Promise<void>((r) => (finish = r)); // the animation playing
    });
    expect(await settled(slot.played)).toBe(false);
    finish();
    expect(await settled(slot.played)).toBe(true);
  });

  test("a run that fails still lets Play move on", async () => {
    const slot = new FigureSlot();
    slot.set(async () => {
      throw new Error("boom");
    });
    expect(await settled(slot.played)).toBe(true);
  });

  test("no figure means nothing to wait for", async () => {
    const slot = new FigureSlot();
    slot.set(null);
    expect(await settled(slot.played)).toBe(true);
  });
});
