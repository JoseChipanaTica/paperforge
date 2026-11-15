import type { LocatedArchitecture } from "../shared/graph";
import type { ExampleArtifact, WalkthroughArtifact } from "../shared/learn";
import { Run } from "./anim";
import type { ForgeScene } from "./scene";
import { escapeHtml } from "./scene";

/** The guided tour shared by the walkthrough and the easy example: one caption per block. */

export interface TourStep {
  block_id: string;
  title: string; // empty → the block's label
  body: string;
  detail: string;
}

export interface TourScript {
  intro: { title: string; body: string; detail: string };
  steps: TourStep[];
  outro: string;
}

export function tourSteps(kind: "walkthrough", artifact: WalkthroughArtifact): TourScript;
export function tourSteps(kind: "example", artifact: ExampleArtifact): TourScript;
export function tourSteps(kind: "walkthrough" | "example", artifact: WalkthroughArtifact | ExampleArtifact): TourScript {
  if (kind === "walkthrough") {
    const a = artifact as WalkthroughArtifact;
    return {
      intro: { title: "Walkthrough", body: a.intro, detail: "" },
      steps: a.steps.map((s) => ({ block_id: s.block_id, title: s.title, body: s.narration, detail: s.shape_change })),
      outro: a.outro,
    };
  }
  const a = artifact as ExampleArtifact;
  return {
    intro: { title: a.analogy.title, body: a.analogy.text, detail: `Toy input: “${a.toy_input}”` },
    steps: a.steps.map((s) => ({ block_id: s.block_id, title: "", body: s.what_happens, detail: s.toy_value })),
    outro: a.takeaway,
  };
}

const PLAY_MS = 4000;

export class Tour {
  private script: TourScript | null = null;
  private graph: LocatedArchitecture | null = null;
  private index = -1; // -1 = intro, steps.length = outro
  private run = new Run();
  private timer: ReturnType<typeof setTimeout> | undefined;
  onStop: () => void = () => {};

  constructor(
    private scene: ForgeScene,
    private root: HTMLElement,
  ) {
    root.addEventListener("click", (e) => {
      const action = (e.target as HTMLElement).closest<HTMLElement>("button")?.dataset.tour;
      if (action === "next") this.next();
      else if (action === "back") this.back();
      else if (action === "play") this.toggleAuto();
      else if (action === "stop") this.stop();
    });
  }

  get active() {
    return this.script !== null;
  }

  start(script: TourScript, graph: LocatedArchitecture) {
    this.stop(false);
    this.script = script;
    this.graph = graph;
    this.index = -1;
    this.root.hidden = false;
    this.show();
  }

  next() {
    if (!this.script || this.index >= this.script.steps.length) return this.stopAuto();
    this.index++;
    this.show();
  }

  back() {
    if (!this.script || this.index < 0) return;
    this.index--;
    this.show();
  }

  stop(notify = true) {
    this.stopAuto();
    this.run.cancel();
    if (!this.script) return;
    this.script = null;
    this.root.hidden = true;
    this.scene.highlight(null);
    void this.scene.focusOn(null, (this.run = new Run()));
    if (notify) this.onStop();
  }

  private toggleAuto() {
    if (this.timer !== undefined) return this.stopAuto();
    const tick = () => {
      this.next();
      if (this.script && this.index < this.script.steps.length) this.timer = setTimeout(tick, PLAY_MS);
      else this.stopAuto();
    };
    this.timer = setTimeout(tick, 0);
    this.render();
  }

  private stopAuto() {
    clearTimeout(this.timer);
    this.timer = undefined;
    this.render();
  }

  private show() {
    const step = this.script?.steps[this.index];
    this.run.cancel();
    this.run = new Run();
    this.scene.highlight(step?.block_id ?? null);
    void this.scene.focusOn(step?.block_id ?? null, this.run);
    this.render();
  }

  private render() {
    const s = this.script;
    if (!s) return;
    const n = s.steps.length;
    const step = s.steps[this.index];
    const label = step && (step.title || this.graph?.blocks.find((b) => b.id === step.block_id)?.label || step.block_id);
    const [title, body, detail] =
      this.index < 0 ? [s.intro.title, s.intro.body, s.intro.detail] : step ? [label!, step.body, step.detail] : ["That's the whole model", s.outro, ""];
    const where = this.index < 0 ? "Intro" : step ? `${this.index + 1} / ${n}` : "Done";
    this.root.innerHTML = `
      <div class="tour-head"><span class="tour-count">${where}</span><button data-tour="stop" title="Exit (Esc)">×</button></div>
      <h4>${escapeHtml(title)}</h4>
      <p>${escapeHtml(body).replace(/\n/g, "<br>")}</p>
      ${detail ? `<code>${escapeHtml(detail)}</code>` : ""}
      <div class="tour-nav">
        <button data-tour="back" ${this.index < 0 ? "disabled" : ""}>‹ Back</button>
        <button data-tour="play">${this.timer !== undefined ? "Pause" : "Play"}</button>
        <button data-tour="next" ${step || this.index < 0 ? "" : "disabled"}>Next ›</button>
      </div>`;
  }
}
