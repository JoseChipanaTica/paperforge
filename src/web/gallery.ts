import type { PaperSummary } from "../shared/graph";
import { escapeHtml } from "./scene";

/** Fill `el` with cards for published papers; empty when there are none. */
export async function loadGallery(el: HTMLElement, onOpen: (id: string) => void): Promise<void> {
  let papers: PaperSummary[];
  try {
    const res = await fetch("/api/papers?limit=24");
    if (!res.ok) throw new Error(String(res.status));
    papers = await res.json();
  } catch {
    el.innerHTML = `<p class="gallery-note">Gallery unavailable</p>`;
    return;
  }
  if (!papers.length) {
    el.innerHTML = "";
    return;
  }
  el.innerHTML = `<h3>From the gallery</h3><div class="gallery-grid">${papers
    .map(
      (p) => `<button class="gallery-card" data-id="${escapeHtml(p.arxiv_id)}">
        <b>${escapeHtml(p.model_name)}</b>
        <span>${escapeHtml(p.paper_title)}</span>
        <small>arXiv:${escapeHtml(p.arxiv_id)} · ${p.block_count} blocks</small>
      </button>`,
    )
    .join("")}</div>`;
  el.querySelectorAll<HTMLButtonElement>(".gallery-card").forEach((btn) =>
    btn.addEventListener("click", () => onOpen(btn.dataset.id!)),
  );
}
