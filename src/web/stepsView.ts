import type { LocatedArchitecture } from "../shared/graph";
import { escapeHtml } from "./scene";
import type { TourDriver, TourScript } from "./tour";

/**
 * The walkthrough as a list in the study panel: intro, every step, outro, with the
 * current step marked and Back / Play / Next pinned at the bottom. The camera follows.
 */
export function renderSteps(root: HTMLElement, script: TourScript, graph: LocatedArchitecture, driver: TourDriver, start = -1) {
  const label = (blockId: string) => graph.blocks.find((b) => b.id === blockId)?.label ?? blockId;
  root.innerHTML = `
    <div class="steps">
      <section class="step step-intro" data-index="-1">
        <h3>${escapeHtml(script.intro.title)}</h3>
        <p>${escapeHtml(script.intro.body).replace(/\n/g, "<br>")}</p>
        <code>${escapeHtml(script.intro.label)}</code>
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
            <div class="step-body">
              <p><b>${escapeHtml(s.what)}</b></p>
              <p>${escapeHtml(s.body)}</p>
              ${s.shape ? `<code>${escapeHtml(s.shape)}</code>` : ""}
            </div>
          </li>`,
          )
          .join("")}
      </ol>
      <section class="step step-outro" data-index="${script.steps.length}">
        <p>${escapeHtml(script.outro)}</p>
      </section>
    </div>
    <nav class="step-nav" aria-label="Tour controls">
      <button type="button" class="btn" data-nav="back">Back</button>
      <button type="button" class="btn" data-nav="play" aria-pressed="false">Play</button>
      <span class="step-pos" aria-live="polite"></span>
      <button type="button" class="btn btn-primary" data-nav="next">Next</button>
    </nav>`;

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
  };
  const playing = (on: boolean) => {
    nav("play").setAttribute("aria-pressed", String(on));
    nav("play").textContent = on ? "Pause" : "Play";
  };

  driver.onIndex = show;
  driver.onPlaying = playing;
  root.onclick = (e) => {
    const t = e.target as HTMLElement;
    const go = t.closest<HTMLElement>("[data-go]")?.dataset.go;
    if (go !== undefined) return driver.goTo(script, Number(go));
    const action = t.closest<HTMLElement>("[data-nav]")?.dataset.nav;
    if (action === "next") driver.next();
    else if (action === "back") driver.back();
    else if (action === "play") driver.togglePlay();
  };

  playing(driver.playing);
  driver.goTo(script, start);
}
