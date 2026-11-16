import { describe, expect, test } from "bun:test";
import { initialPanel, panelReducer, snapSheet, trainingPrefetchable, walkthroughShown, type PanelAction, type PanelState } from "./panelState";

const run = (...actions: PanelAction[]): PanelState => actions.reduce(panelReducer, initialPanel());

describe("panelReducer", () => {
  test("ready kinds get a dot; the rest stay idle", () => {
    const s = run({ type: "kinds", ready: ["pytorch"] });
    expect(s.status).toEqual({ pytorch: "ready", journey: "idle", motion: "idle", training: "idle" });
    expect(s.active).toBe("overview");
  });

  test("a generating tab keeps its status across refreshes and tab switches", () => {
    const s = run(
      { type: "generate", kind: "journey" },
      { type: "kinds", ready: [] },
      { type: "select", tab: "pytorch" },
    );
    expect(s.status.journey).toBe("generating");
    expect(s.active).toBe("pytorch");
  });

  test("kinds never downgrades a loaded tab", () => {
    expect(run({ type: "loaded", kind: "journey" }, { type: "kinds", ready: [] }).status.journey).toBe("loaded");
  });

  test("failures become needs_key or error; retrying clears the error", () => {
    const s = run(
      { type: "failed", kind: "journey", message: "x", needsKey: true },
      { type: "failed", kind: "pytorch", message: "boom", needsKey: false },
    );
    expect(s.status.journey).toBe("needs_key");
    expect(s.status.pytorch).toBe("error");
    expect(s.error.pytorch).toBe("boom");
    expect(panelReducer(s, { type: "generate", kind: "pytorch" }).error.pytorch).toBeUndefined();
  });

  test("a new paper resets every tab; Overview or Paper stay selected", () => {
    const busy = run({ type: "loaded", kind: "pytorch" }, { type: "generate", kind: "journey" }, { type: "select", tab: "journey" });
    const reset = panelReducer(busy, { type: "reset" });
    expect(reset.status).toEqual({ pytorch: "idle", journey: "idle", motion: "idle", training: "idle" });
    expect(reset.active).toBe("overview");
    expect(panelReducer(run({ type: "select", tab: "paper" }), { type: "reset" }).active).toBe("paper");
  });
});

describe("snapSheet", () => {
  const snaps = [64, 380, 760];
  test("a slow release goes to the nearest snap", () => {
    expect(snapSheet(300, 0, snaps)).toBe(380);
    expect(snapSheet(800, 0, snaps)).toBe(760);
    expect(snapSheet(150, 0, snaps)).toBe(64);
  });

  test("a flick goes to the next snap in its direction", () => {
    expect(snapSheet(300, 900, snaps)).toBe(380);
    expect(snapSheet(400, 900, snaps)).toBe(760);
    expect(snapSheet(400, -900, snaps)).toBe(380);
    expect(snapSheet(100, -900, snaps)).toBe(64);
  });
});

describe("walkthroughShown", () => {
  test("inside a block, the walkthrough is that block's own (a language change included)", () => {
    expect(walkthroughShown("encoder_ffn", true)).toEqual({ kind: "inside", blockId: "encoder_ffn" });
    expect(walkthroughShown("encoder_ffn", false)).toEqual({ kind: "inside", blockId: "encoder_ffn" });
  });

  test("on the whole model: the model's walkthrough if loaded, else its start state (never a leftover inside list)", () => {
    expect(walkthroughShown(null, true)).toEqual({ kind: "main" });
    expect(walkthroughShown(null, false)).toEqual({ kind: "state" });
  });
});

describe("trainingPrefetchable", () => {
  const after = (...actions: PanelAction[]) => actions.reduce(panelReducer, initialPanel());

  test("only a training step that already exists is fetched before the visitor asks", () => {
    expect(trainingPrefetchable(initialPanel())).toBe(false);
    expect(trainingPrefetchable(after({ type: "kinds", ready: ["journey", "motion"] }))).toBe(false);
    expect(trainingPrefetchable(after({ type: "kinds", ready: ["journey", "training"] }))).toBe(true);
    expect(trainingPrefetchable(after({ type: "kinds", ready: ["training"] }, { type: "reset" }))).toBe(false);
  });
});
