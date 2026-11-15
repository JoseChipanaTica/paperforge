import { describe, expect, test } from "bun:test";
import { initialPanel, panelReducer, snapSheet, type PanelAction, type PanelState } from "./panelState";

const run = (...actions: PanelAction[]): PanelState => actions.reduce(panelReducer, initialPanel());

describe("panelReducer", () => {
  test("ready kinds get a dot; the rest stay idle", () => {
    const s = run({ type: "kinds", ready: ["pytorch"] });
    expect(s.status).toEqual({ pytorch: "ready", walkthrough: "idle", example: "idle" });
    expect(s.active).toBe("overview");
  });

  test("a generating tab keeps its status across refreshes and tab switches", () => {
    const s = run(
      { type: "generate", kind: "walkthrough" },
      { type: "kinds", ready: [] },
      { type: "select", tab: "pytorch" },
    );
    expect(s.status.walkthrough).toBe("generating");
    expect(s.active).toBe("pytorch");
  });

  test("kinds never downgrades a loaded tab", () => {
    expect(run({ type: "loaded", kind: "example" }, { type: "kinds", ready: [] }).status.example).toBe("loaded");
  });

  test("failures become needs_key or error; retrying clears the error", () => {
    const s = run(
      { type: "failed", kind: "walkthrough", message: "x", needsKey: true },
      { type: "failed", kind: "example", message: "boom", needsKey: false },
    );
    expect(s.status.walkthrough).toBe("needs_key");
    expect(s.status.example).toBe("error");
    expect(s.error.example).toBe("boom");
    expect(panelReducer(s, { type: "generate", kind: "example" }).error.example).toBeUndefined();
  });

  test("a new paper resets every tab; Overview or Paper stay selected", () => {
    const busy = run({ type: "loaded", kind: "pytorch" }, { type: "generate", kind: "example" }, { type: "select", tab: "walkthrough" });
    const reset = panelReducer(busy, { type: "reset" });
    expect(reset.status).toEqual({ pytorch: "idle", walkthrough: "idle", example: "idle" });
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
