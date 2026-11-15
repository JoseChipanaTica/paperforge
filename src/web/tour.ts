import type { JourneyArtifact, View } from "../shared/learn";
import { Run } from "./anim";
import type { ForgeScene } from "./scene";

/** The guided tour: one step per block, following one input from the raw data to the output. */

export interface TourStep {
  block_id: string;
  title: string; // empty → the block's label
  what: string; // one plain sentence
  body: string; // the narration
  shape: string; // "[B, 3, 512] → [B, 3, 512]"
  views: View[]; // the data after this block
}

export interface TourScript {
  analogy: { title: string; text: string };
  intro: { title: string; body: string; label: string; views: View[] };
  steps: TourStep[];
  output: { label: string; views: View[] };
  outro: string;
}

export function tourSteps(a: JourneyArtifact): TourScript {
  return {
    analogy: a.analogy,
    intro: { title: "Walkthrough", body: a.intro, label: a.input.label, views: a.input.views },
    steps: a.steps.map((s) => ({
      block_id: s.block_id,
      title: s.title,
      what: s.what_happens,
      body: s.narration,
      shape: s.shape_change,
      views: s.views,
    })),
    output: a.output,
    outro: a.outro,
  };
}

const PLAY_MS = 4000;

const reducedMotion = () => typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;

/**
 * Drives the camera and highlight through a tour script: -1 is the intro, steps.length the
 * outro. The step list UI (stepsView) renders the words and calls goTo / next / back.
 */
export class TourDriver {
  script: TourScript | null = null;
  index = -1;
  onIndex: (i: number) => void = () => {};
  onPlaying: (playing: boolean) => void = () => {};
  private run = new Run();
  private timer: ReturnType<typeof setTimeout> | undefined;
  // Where a stopped tour was, so Next / Back / Play pick it up again after Esc.
  private resume: { script: TourScript; index: number } | null = null;

  constructor(private scene: ForgeScene) {}

  get active() {
    return this.script !== null;
  }

  get playing() {
    return this.timer !== undefined;
  }

  goTo(script: TourScript, index: number) {
    this.script = script;
    this.index = Math.max(-1, Math.min(script.steps.length, index));
    const step = script.steps[this.index];
    this.run.cancel();
    this.run = new Run();
    if (reducedMotion()) this.run.skip(); // cut instead of flying
    this.scene.highlight(step?.block_id ?? null);
    void this.scene.focusOn(step?.block_id ?? null, this.run);
    this.onIndex(this.index);
  }

  /** The tour to step through: the running one, or the one Esc stopped. */
  private current(): { script: TourScript; index: number } | null {
    return this.script ? { script: this.script, index: this.index } : this.resume;
  }

  next() {
    const c = this.current();
    if (!c || c.index >= c.script.steps.length) return this.pause();
    this.goTo(c.script, c.index + 1);
  }

  back() {
    const c = this.current();
    if (c && c.index > -1) this.goTo(c.script, c.index - 1);
  }

  togglePlay() {
    if (this.playing) return this.pause();
    const c = this.current();
    if (!c) return;
    this.script = c.script;
    this.index = c.index;
    const tick = () => {
      this.next();
      if (this.script && this.index < this.script.steps.length) this.timer = setTimeout(tick, PLAY_MS);
      else this.pause();
    };
    this.timer = setTimeout(tick, 0);
    this.onPlaying(true);
  }

  pause() {
    if (this.timer === undefined) return;
    clearTimeout(this.timer);
    this.timer = undefined;
    this.onPlaying(false);
  }

  /** A new paper: stop, and don't resume the old paper's tour. */
  forget() {
    this.stop();
    this.resume = null;
  }

  /** Leave the tour: clear the highlight and bring the whole model back into view. */
  stop() {
    this.pause();
    this.run.cancel();
    if (!this.script) return;
    this.resume = { script: this.script, index: this.index };
    this.script = null;
    this.index = -1;
    this.scene.highlight(null);
    this.run = new Run();
    if (reducedMotion()) this.run.skip();
    void this.scene.focusOn(null, this.run);
  }
}
