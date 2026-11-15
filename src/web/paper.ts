import type { LocatedBlock } from "../shared/graph";
import { Run, ease } from "./anim";
import { escapeHtml } from "./scene";

interface Mark {
  start: number;
  end: number;
  ids: string[];
}

/** The paper page that flips, gets highlighted, and launches blocks into the scene. */
export class PaperView {
  private sheet: HTMLElement;
  private body: HTMLElement;
  private pageLabel: HTMLElement;
  private idLabel: HTMLElement;
  private pages: string[] = [];
  private marks = new Map<number, Mark[]>();
  current = -1;
  private readingTimer: ReturnType<typeof setInterval> | null = null;

  constructor(readonly root: HTMLElement) {
    this.sheet = root.querySelector(".paper-sheet")!;
    this.body = root.querySelector(".paper-body")!;
    this.pageLabel = root.querySelector(".paper-page")!;
    this.idLabel = root.querySelector(".paper-id")!;
  }

  setPaper(arxivId: string, pages: string[]) {
    this.pages = pages;
    this.marks.clear();
    this.idLabel.textContent = `arXiv:${arxivId}`;
    this.current = -1;
  }

  /** Register where each block's quote lives; overlapping quotes share one mark. */
  setBlocks(blocks: LocatedBlock[]) {
    this.marks.clear();
    for (const b of blocks) {
      if (!b.source) continue;
      const list = this.marks.get(b.source.page) ?? [];
      const hit = list.find((m) => b.source!.start < m.end && b.source!.end > m.start);
      if (hit) {
        hit.ids.push(b.id);
        hit.start = Math.min(hit.start, b.source.start);
        hit.end = Math.max(hit.end, b.source.end);
      } else list.push({ start: b.source.start, end: b.source.end, ids: [b.id] });
      this.marks.set(b.source.page, list);
    }
    for (const list of this.marks.values()) list.sort((a, b) => a.start - b.start);
  }

  show(visible: boolean) {
    this.root.classList.toggle("visible", visible);
  }

  get visible() {
    return this.root.classList.contains("visible");
  }

  private render(page: number) {
    const text = reflow(this.pages[page] ?? "");
    let html = "";
    let cursor = 0;
    for (const m of this.marks.get(page) ?? []) {
      if (m.start < cursor) continue;
      html += escapeHtml(text.slice(cursor, m.start));
      html += `<mark data-ids="${escapeHtml(m.ids.join(" "))}">${escapeHtml(text.slice(m.start, m.end))}</mark>`;
      cursor = m.end;
    }
    html += escapeHtml(text.slice(cursor));
    this.body.innerHTML = html;
    this.body.scrollTop = 0;
    this.pageLabel.textContent = `p. ${page + 1} / ${this.pages.length}`;
    this.current = page;
  }

  /** Turn to a page with a flip around the spine. */
  async flipTo(page: number, run: Run, ms = 420) {
    if (page === this.current) return;
    const forward = page > this.current;
    await run.tween(ms / 2, (t) => this.setTurn((forward ? -1 : 1) * 90 * t), ease.inCubic);
    this.render(page);
    await run.tween(ms / 2, (t) => this.setTurn((forward ? 1 : -1) * 90 * (1 - t)), ease.outCubic);
  }

  private setTurn(deg: number) {
    this.sheet.style.transform = `perspective(1800px) rotateY(${deg}deg)`;
    this.sheet.style.setProperty("--shade", String(Math.abs(deg) / 90));
  }

  /** While Claude reads: idle-flip through the pages with a scan line. */
  startReading() {
    this.stopReading();
    this.root.classList.add("reading");
    if (this.current < 0) this.render(0);
    const free = new Run(); // reading flips are never skipped
    this.readingTimer = setInterval(() => {
      const next = (this.current + 1) % Math.max(1, Math.min(this.pages.length, 10));
      void this.flipTo(next, free, 700);
    }, 2200);
  }

  stopReading() {
    if (this.readingTimer) clearInterval(this.readingTimer);
    this.readingTimer = null;
    this.root.classList.remove("reading");
  }

  private markFor(blockId: string): HTMLElement | null {
    return [...this.body.querySelectorAll<HTMLElement>("mark")].find((m) => m.dataset.ids!.split(" ").includes(blockId)) ?? null;
  }

  /** Scroll the quote into view and sweep the highlighter across it. Returns where it ends on screen. */
  async sweep(blockId: string, run: Run): Promise<{ x: number; y: number } | null> {
    const mark = this.markFor(blockId);
    if (!mark) return null;
    const from = this.body.scrollTop;
    const view = this.body.clientHeight;
    const inView = mark.offsetTop > from + view * 0.08 && mark.offsetTop + mark.offsetHeight < from + view * 0.85;
    const target = Math.max(0, mark.offsetTop - view * 0.35);
    if (!inView) await run.tween(220, (t) => (this.body.scrollTop = from + (target - from) * t));
    this.body.querySelectorAll("mark.active").forEach((m) => m.classList.remove("active"));
    mark.classList.add("active");
    if (mark.classList.contains("lit")) {
      mark.classList.remove("pop");
      void mark.offsetWidth;
      mark.classList.add("pop");
    } else {
      mark.classList.add("lit");
    }
    await run.wait(run.skipping ? 0 : 300);
    const rects = mark.getClientRects();
    const last = rects[rects.length - 1] ?? mark.getBoundingClientRect();
    return { x: last.right, y: last.top + last.height / 2 };
  }

  /** Jump straight to a block's quote (used when inspecting a finished model). */
  async reveal(block: LocatedBlock, run: Run) {
    if (!block.source) return;
    this.show(true);
    await this.flipTo(block.source.page, run, 380);
    this.body.querySelectorAll("mark").forEach((m) => m.classList.add("lit"));
    await this.sweep(block.id, run);
  }

  /** Center of the card on screen, for blocks without a located quote. */
  center(): { x: number; y: number } {
    const r = this.sheet.getBoundingClientRect();
    return { x: r.left + r.width * 0.7, y: r.top + r.height / 2 };
  }
}

/**
 * pdf.js keeps the PDF's hard line breaks, which wrap raggedly in a narrower card.
 * Join full-width lines into paragraphs while keeping every character offset intact
 * (each "\n" becomes one space), so quote spans still line up.
 */
export function reflow(text: string): string {
  const lines = text.split("\n");
  const lengths = lines.map((l) => l.length).sort((a, b) => a - b);
  const typical = lengths[Math.floor(lengths.length * 0.7)] ?? 0;
  return lines
    .map((line, i) => {
      if (i === lines.length - 1) return line;
      const full = line.length >= typical * 0.8;
      return line + (full ? (line.endsWith("-") ? "\u200b" : " ") : "\n");
    })
    .join("");
}
