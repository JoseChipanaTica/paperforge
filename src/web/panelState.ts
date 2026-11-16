import { LEARN_KINDS, type LearnKind } from "../shared/learn";

/** The study panel's state, kept pure so tab behaviour is testable without a DOM. */

export const TABS = ["overview", "paper", "journey", "pytorch"] as const;
export type TabId = (typeof TABS)[number];
/** Learning kinds with a tab of their own (motion lives inside the walkthrough). */
export type TabKind = Extract<LearnKind, TabId>;
export const isTabKind = (t: TabId): t is TabKind => t === "journey" || t === "pytorch";

/**
 * What the Walkthrough tab shows: inside a block, that block's own walkthrough; on the whole
 * model, the model's (or its start state when not loaded yet, never a leftover inside list).
 */
export function walkthroughShown(insideBlock: string | null, mainLoaded: boolean): { kind: "inside"; blockId: string } | { kind: "main" } | { kind: "state" } {
  if (insideBlock) return { kind: "inside", blockId: insideBlock };
  return mainLoaded ? { kind: "main" } : { kind: "state" };
}

export type KindStatus = "idle" | "ready" | "generating" | "loaded" | "needs_key" | "error";

export interface PanelState {
  active: TabId;
  status: Record<LearnKind, KindStatus>;
  error: Partial<Record<LearnKind, string>>;
}

export type PanelAction =
  | { type: "select"; tab: TabId }
  | { type: "kinds"; ready: LearnKind[] }
  | { type: "generate"; kind: LearnKind }
  | { type: "loaded"; kind: LearnKind }
  | { type: "unready"; kind: LearnKind }
  | { type: "failed"; kind: LearnKind; message: string; needsKey: boolean }
  | { type: "reset" };

const allIdle = () => Object.fromEntries(LEARN_KINDS.map((k) => [k, "idle"])) as Record<LearnKind, KindStatus>;

export function initialPanel(): PanelState {
  return { active: "overview", status: allIdle(), error: {} };
}

export function panelReducer(s: PanelState, a: PanelAction): PanelState {
  switch (a.type) {
    case "select":
      return { ...s, active: a.tab };
    case "kinds": {
      // Only idle tabs learn they're ready; never downgrade work in progress or done.
      const status = { ...s.status };
      for (const k of a.ready) if (status[k] === "idle") status[k] = "ready";
      return { ...s, status };
    }
    case "generate": {
      const { [a.kind]: _cleared, ...error } = s.error;
      return { ...s, status: { ...s.status, [a.kind]: "generating" }, error };
    }
    case "loaded":
      return { ...s, status: { ...s.status, [a.kind]: "loaded" } };
    case "unready":
      // It looked ready (stored in this browser) but wasn't there after all: back to "not written yet".
      return s.status[a.kind] === "ready" ? { ...s, status: { ...s.status, [a.kind]: "idle" } } : s;
    case "failed":
      return {
        ...s,
        status: { ...s.status, [a.kind]: a.needsKey ? "needs_key" : "error" },
        error: a.needsKey ? s.error : { ...s.error, [a.kind]: a.message },
      };
    case "reset":
      return { ...initialPanel(), active: s.active === "paper" ? "paper" : "overview" };
  }
}

/**
 * Fetch the training step before the visitor asks only when it already exists (gallery or sample, per GET /api/learn's
 * kinds): reaching the walkthrough's end must never start a Claude call on its own.
 */
export const trainingPrefetchable = (s: PanelState) => s.status.training === "ready";

/**
 * A learning tab whose artifact isn't in the browser store after all: write it with Claude only when the visitor
 * clicked for it or the server listed it as existing (free). A tab that opened on its own because the browser seemed to
 * hold it shows its "not written yet" state instead, so nothing is paid for unasked.
 */
export const missAction = ({ explicit, serverReady }: { explicit: boolean; serverReady: boolean }): "generate" | "unwritten" =>
  explicit || serverReady ? "generate" : "unwritten";

const FLICK = 500; // px/s

/** Where a dragged bottom sheet settles: the nearest snap, or the next one in a flick's direction. */
export function snapSheet(height: number, velocity: number, snaps: number[]): number {
  if (velocity > FLICK) return snaps.find((s) => s >= height) ?? snaps.at(-1)!;
  if (velocity < -FLICK) return [...snaps].reverse().find((s) => s <= height) ?? snaps[0]!;
  return snaps.reduce((best, s) => (Math.abs(s - height) < Math.abs(best - height) ? s : best));
}
