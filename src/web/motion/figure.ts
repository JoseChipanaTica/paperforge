import type { MotionArtifact, Op } from "../../shared/motion";
import { sanitizeSvg } from "../../shared/svg";
import { runChain } from "./compute";
import { H, W } from "./frames";
import { Player } from "./player";
import { framesFor } from "./templates";

/**
 * The animation under a walkthrough step: the block's ops played one after another in a printed
 * figure, with each op's caption, "1 / 3", Replay, and under reduced motion a "Next stage" button
 * instead of tweening.
 */

export type FigureStatus = "loading" | "error" | "needs_key" | "ready";
export type FigureState = { kind: "loading" | "error" | "needs_key" | "none" } | { kind: "ops"; ops: Op[] };

/** What a step's figure shows, given where the motion artifact is. */
export function figureState(motion: MotionArtifact | null, status: FigureStatus, blockId: string): FigureState {
  if (status !== "ready" || !motion) return { kind: status === "ready" ? "loading" : status };
  const step = motion.steps.find((s) => s.block_id === blockId);
  return step?.ops.length ? { kind: "ops", ops: step.ops } : { kind: "none" };
}

/**
 * What the tour's Play waits on before the next step: the current figure's whole run (localizing
 * its captions, then playing). Set synchronously, so Play never reads a promise from before the
 * step changed; a failing run settles too, so Play can't get stuck.
 */
export class FigureSlot {
  played: Promise<void> = Promise.resolve();

  set(run: (() => Promise<void>) | null) {
    this.played = run ? run().catch(() => {}) : Promise.resolve();
  }
}

const NS = "http://www.w3.org/2000/svg";
const BETWEEN_OPS = 700;
const CUSTOM_MS = 3500; // a custom SVG animates itself; give it this long before the next op

const wait = (ms: number) => new Promise<void>((r) => setTimeout(r, ms));

export class StepFigure {
  private stage: HTMLElement;
  private svg: SVGSVGElement;
  private player: Player;
  private caption: HTMLElement;
  private count: HTMLElement;
  private controls: HTMLElement;
  private run = 0;

  constructor(
    readonly root: HTMLElement,
    private reduced: boolean,
  ) {
    root.replaceChildren();
    const figure = document.createElement("figure");
    figure.className = "motion";
    this.stage = document.createElement("div");
    this.stage.className = "motion-stage";
    this.svg = this.makeSvg();
    this.stage.append(this.svg);
    const cap = document.createElement("figcaption");
    this.count = document.createElement("span");
    this.count.className = "motion-count";
    this.caption = document.createElement("span");
    this.caption.className = "motion-caption";
    cap.append(this.count, this.caption);
    this.controls = document.createElement("div");
    this.controls.className = "motion-controls";
    figure.append(this.stage, cap, this.controls);
    root.append(figure);
    this.player = new Player(this.svg);
  }

  private makeSvg(): SVGSVGElement {
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
    svg.setAttribute("role", "img");
    return svg;
  }

  /** A message instead of an animation: loading, an error with Retry, a key form, or nothing. */
  message(kind: Exclude<FigureState["kind"], "ops">, onRetry?: () => void, onKey?: (key: string) => void) {
    this.stop();
    this.root.hidden = kind === "none";
    this.count.textContent = "";
    this.controls.replaceChildren();
    this.player.clear();
    if (kind === "loading") this.caption.textContent = "Preparing the animation…";
    if (kind === "error") {
      this.caption.textContent = "Couldn't animate this model.";
      if (onRetry) this.controls.append(this.button("Try again", onRetry));
    }
    if (kind === "needs_key") {
      this.caption.textContent = "Add your API key to animate this paper.";
      const form = document.createElement("form");
      form.className = "motion-key";
      const input = document.createElement("input");
      input.type = "password";
      input.placeholder = "sk-ant-…";
      input.autocomplete = "off";
      input.setAttribute("aria-label", "Anthropic API key");
      form.append(input, this.button("Save key", () => {}, "submit"));
      form.addEventListener("submit", (e) => {
        e.preventDefault();
        if (input.value.trim()) onKey?.(input.value.trim());
      });
      this.controls.append(form);
    }
    this.stage.classList.toggle("motion-waiting", kind === "loading");
  }

  /** Play the ops in order; resolves when the chain has played (at once under reduced motion). */
  async show(ops: Op[]): Promise<void> {
    const run = ++this.run;
    this.root.hidden = false;
    this.stage.classList.remove("motion-waiting");
    const chain = runChain(ops);
    this.controls.replaceChildren(this.button("Replay", () => void this.show(ops)));
    if (this.reduced) return this.showStage(chain, 0, run);
    for (let i = 0; i < chain.length; i++) {
      if (run !== this.run) return;
      await this.playOne(chain, i);
      if (i < chain.length - 1) await wait(BETWEEN_OPS);
    }
  }

  private label(chain: ReturnType<typeof runChain>, i: number) {
    const op = chain[i]!.op;
    this.count.textContent = chain.length > 1 ? `${i + 1} / ${chain.length}` : "";
    this.caption.textContent = op.caption;
    this.svg.setAttribute("aria-label", op.caption || op.kind.replace(/_/g, " "));
  }

  private async playOne(chain: ReturnType<typeof runChain>, i: number) {
    const { op, input, result } = chain[i]!;
    this.label(chain, i);
    if (this.custom(op)) return wait(CUSTOM_MS);
    this.useOwnSvg();
    this.player.clear();
    await this.player.play(framesFor(op, input, result));
  }

  /** Reduced motion: the final frame of one op, with "Next stage" to move through the chain. */
  private showStage(chain: ReturnType<typeof runChain>, i: number, run: number) {
    if (run !== this.run) return;
    const { op, input, result } = chain[i]!;
    this.label(chain, i);
    if (!this.custom(op)) {
      this.useOwnSvg();
      this.player.clear();
      this.player.showFinal(framesFor(op, input, result));
    }
    const buttons = [this.button("Replay", () => void this.show(chain.map((c) => c.op)))];
    if (i < chain.length - 1) buttons.push(this.button("Next stage", () => this.showStage(chain, i + 1, run)));
    this.controls.replaceChildren(...buttons);
  }

  /** Draw Claude's own SVG for a custom op (sanitized again here); false when there is none to draw. */
  private custom(op: Op): boolean {
    if (op.kind !== "custom") return false;
    const clean = sanitizeSvg(op.svg);
    const doc = clean ? new DOMParser().parseFromString(clean, "image/svg+xml") : null;
    const parsed = doc?.documentElement;
    // A parse error still yields a root <svg> holding a <parsererror>: never show that box.
    if (!parsed || parsed.nodeName !== "svg" || doc!.getElementsByTagName("parsererror").length) return false; // framesFor draws the placeholder
    const node = document.importNode(parsed, true) as unknown as SVGSVGElement;
    node.setAttribute("viewBox", node.getAttribute("viewBox") ?? `0 0 ${W} ${H}`);
    node.setAttribute("role", "img");
    node.setAttribute("aria-label", op.caption);
    this.player.stop();
    this.stage.replaceChildren(node);
    return true;
  }

  private useOwnSvg() {
    if (this.svg.parentNode !== this.stage) this.stage.replaceChildren(this.svg);
  }

  private button(text: string, onClick: () => void, type: "button" | "submit" = "button") {
    const b = document.createElement("button");
    b.type = type;
    b.className = "btn btn-small";
    b.textContent = text;
    if (type === "button") b.addEventListener("click", onClick);
    return b;
  }

  stop() {
    this.run++;
    this.player.stop();
  }
}
