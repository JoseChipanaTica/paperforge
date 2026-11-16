import type { LocatedArchitecture } from "../shared/graph";
import { escapeHtml } from "./scene";
import type { TourDriver, TourScript, TourStep } from "./tour";
import { shapeHtml } from "./trace";

/**
 * The walkthrough as a list in the study panel: the analogy, the input, every step, the output,
 * with the current step marked and Back / Play / Next pinned at the bottom. The camera and the
 * specimen follow.
 */

/** One step: what happens in a plain sentence, why, the shape change (traceable), what the specimen shows. */
export function stepHtml(s: TourStep, traced: string | null): string {
  return `
    <p class="step-what"><b>${escapeHtml(s.what)}</b></p>
    <div class="step-figure" hidden></div>
    <p>${escapeHtml(s.body)}</p>
    ${s.shape ? `<p class="step-shape">${shapeHtml(s.shape, traced)}</p>` : ""}
    ${s.views.length ? `<ul class="step-views" aria-label="On the card">${s.views.map((v) => `<li>${escapeHtml(v.caption)}</li>`).join("")}</ul>` : ""}`;
}

/** What comes after the outro: a primary button (the walkthrough's "Train on this example →"). */
export interface StepsContinue {
  label: string;
  run: () => void;
}

/** The step list and its controls, as markup. */
export function stepsHtml(script: TourScript, graph: LocatedArchitecture, traced: string | null, continueLabel?: string): string {
  const label = (blockId: string) => graph.blocks.find((b) => b.id === blockId)?.label ?? blockId;
  const captions = (views: { caption: string }[]) =>
    views.length ? `<ul class="step-views">${views.map((v) => `<li>${escapeHtml(v.caption)}</li>`).join("")}</ul>` : "";
  return `
    <div class="steps">
      <aside class="step-analogy">
        <h3>${escapeHtml(script.analogy.title)}</h3>
        <p>${escapeHtml(script.analogy.text)}</p>
      </aside>
      <section class="step step-intro" data-index="-1">
        <p>${escapeHtml(script.intro.body)}</p>
        <p class="step-moment"><span>Input</span> ${escapeHtml(script.intro.label)}</p>
        ${captions(script.intro.views)}
        <p class="step-note">Values are illustrative</p>
      </section>
      <ol class="step-list">
        ${script.steps
          .map(
            (s, i) => `
          <li class="step" data-index="${i}">
            <button type="button" class="step-head" data-go="${i}">
              <span class="step-n">${i + 1}</span>
              <span class="step-title">${escapeHtml(s.title || label(s.block_id))}</span>
            </button>
            <div class="step-body">${stepHtml(s, traced)}</div>
          </li>`,
          )
          .join("")}
      </ol>
      <section class="step step-outro" data-index="${script.steps.length}">
        <p class="step-moment"><span>Output</span> ${escapeHtml(script.output.label)}</p>
        ${captions(script.output.views)}
        <p>${escapeHtml(script.outro)}</p>
        ${continueLabel ? `<button type="button" class="btn btn-primary step-continue" data-continue>${escapeHtml(continueLabel)}</button>` : ""}
      </section>
    </div>
    <nav class="step-nav" aria-label="Tour controls">
      <button type="button" class="btn" data-nav="back">Back</button>
      <button type="button" class="btn" data-nav="play" aria-pressed="false">Play</button>
      <span class="step-pos" aria-live="polite"></span>
      <button type="button" class="btn btn-primary" data-nav="next">Next</button>
    </nav>`;
}

/** A click anywhere in the step list: trace a dimension, jump to a step, a control, or continue. */
export function stepsClick(
  t: HTMLElement,
  script: TourScript,
  driver: TourDriver,
  onTrace: (token: string) => void,
  onContinue: StepsContinue | undefined,
) {
  const dim = t.closest<HTMLElement>("[data-dim]")?.dataset.dim;
  if (dim) return onTrace(dim);
  if (t.closest("[data-continue]")) return onContinue?.run();
  const go = t.closest<HTMLElement>("[data-go]")?.dataset.go;
  if (go !== undefined) return driver.goTo(script, Number(go));
  const action = t.closest<HTMLElement>("[data-nav]")?.dataset.nav;
  if (action === "next") driver.next();
  else if (action === "back") driver.back();
  else if (action === "play") driver.togglePlay();
}

export function renderSteps(
  root: HTMLElement,
  script: TourScript,
  graph: LocatedArchitecture,
  driver: TourDriver,
  start = -1,
  onTrace: (token: string) => void = () => {},
  traced: string | null = null,
  /** The current step changed: its figure container and block (null at the intro and outro). */
  onStep: (figure: HTMLElement | null, blockId: string | null) => void = () => {},
  /** A primary button at the end of the outro. */
  onContinue?: StepsContinue,
) {
  root.innerHTML = stepsHtml(script, graph, traced, onContinue?.label);

  const nav = (name: string) => root.querySelector<HTMLButtonElement>(`[data-nav="${name}"]`)!;
  const show = (i: number) => {
    root.querySelectorAll<HTMLElement>(".step").forEach((el) => {
      const on = Number(el.dataset.index) === i;
      el.classList.toggle("current", on);
      if (on) el.setAttribute("aria-current", "step");
      else el.removeAttribute("aria-current");
      if (on) el.scrollIntoView({ block: "nearest", behavior: "smooth" });
    });
    const n = script.steps.length;
    nav("back").disabled = i <= -1;
    nav("next").disabled = i >= n;
    nav("next").textContent = i < 0 ? "Start" : i >= n - 1 ? "Finish" : "Next";
    root.querySelector(".step-pos")!.textContent = i < 0 ? "" : i >= n ? "Done" : `${i + 1} of ${n}`;
    const step = script.steps[i];
    onStep(step ? root.querySelector<HTMLElement>(`.step[data-index="${i}"] .step-figure`) : null, step?.block_id ?? null);
  };
  const playing = (on: boolean) => {
    nav("play").setAttribute("aria-pressed", String(on));
    nav("play").textContent = on ? "Pause" : "Play";
  };

  driver.onIndex = show;
  driver.onPlaying = playing;
  root.onclick = (e) => stepsClick(e.target as HTMLElement, script, driver, onTrace, onContinue);

  playing(driver.playing);
  driver.goTo(script, start);
}
