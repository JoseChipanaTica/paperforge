import type { LocatedArchitecture, PaperSummary } from "../shared/graph";
import sample from "../../samples/1706.03762.json";
import { escapeHtml, type ForgeScene } from "./scene";

/**
 * The landing page: headline and paste box, the live printed Transformer beside them,
 * how it works, and the gallery. The preview is the committed sample, bundled with the
 * page, so the first thing a visitor sees costs no request.
 */

export interface LandingDeps {
  scene: ForgeScene;
  stage: HTMLElement; // the scene's container; moved into the hero while the landing shows
  onForge(input: string): void;
}

const PREVIEW: LocatedArchitecture = {
  ...(sample as unknown as Omit<LocatedArchitecture, "arxiv_id" | "blocks">),
  arxiv_id: "1706.03762",
  blocks: (sample as unknown as LocatedArchitecture).blocks.map((b) => ({ ...b, source: null })),
};

export function initLanding(root: HTMLElement, deps: LandingDeps) {
  const heroStage = root.querySelector<HTMLElement>("#hero-stage")!;
  const form = root.querySelector<HTMLFormElement>("#hero-form")!;
  const input = form.querySelector<HTMLInputElement>("input")!;
  let home: HTMLElement | null = null;

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    const value = input.value.trim();
    if (value) deps.onForge(value);
    else input.focus();
  });
  root.querySelectorAll<HTMLButtonElement>("[data-paper]").forEach((b) =>
    b.addEventListener("click", () => deps.onForge(b.dataset.paper!)),
  );
  void loadGallery(root.querySelector("#gallery")!, deps.onForge);

  return {
    /** Put the live model in the hero. */
    preview() {
      home ??= deps.stage.parentElement;
      heroStage.append(deps.stage);
      deps.scene.showAssembled(PREVIEW, 1.15);
      deps.scene.setViewShift(0.04); // left-hand labels need more room than right-hand ones
    },
    /** Hand the stage back to the workspace for a real forge. */
    hide() {
      if (home && deps.stage.parentElement !== home) home.prepend(deps.stage);
      deps.scene.pulsing = false;
      deps.scene.setViewShift(0);
      deps.scene.clear();
    },
  };
}

async function loadGallery(el: HTMLElement, onOpen: (id: string) => void): Promise<void> {
  let papers: PaperSummary[];
  try {
    const res = await fetch("/api/papers?limit=24");
    if (!res.ok) throw new Error(String(res.status));
    papers = await res.json();
  } catch {
    el.innerHTML = `<p class="gallery-note">The gallery isn't available right now. You can still forge any paper above.</p>`;
    return;
  }
  if (!papers.length) {
    el.replaceChildren();
    return;
  }
  el.innerHTML = `<h2>From the gallery</h2>
    <p class="gallery-lede">Papers other visitors have forged and published. They open instantly, no key needed.</p>
    <div class="gallery-grid">${papers
      .map(
        (p) => `<button type="button" class="gallery-card" data-id="${escapeHtml(p.arxiv_id)}">
          <b>${escapeHtml(p.model_name)}</b>
          <span>${escapeHtml(p.paper_title)}</span>
          <small>arXiv:${escapeHtml(p.arxiv_id)}, ${p.block_count} blocks</small>
        </button>`,
      )
      .join("")}</div>`;
  el.querySelectorAll<HTMLButtonElement>(".gallery-card").forEach((btn) => btn.addEventListener("click", () => onOpen(btn.dataset.id!)));
}
