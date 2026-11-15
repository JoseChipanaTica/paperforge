import type { LocatedArchitecture, LocatedBlock } from "../shared/graph";
import type { AskTurn, LearnKind } from "../shared/learn";
import { TYPE_COLOR } from "./blocks";
import { LearnError } from "./learn";
import { escapeHtml, renderTex } from "./scene";
import { shapeHtml } from "./trace";

/** The Overview tab: the paper at a glance, or one block with its source sentence and Ask. */

export interface OverviewDeps {
  onStart(kind: LearnKind): void;
  onShowInPaper(blockId: string): void;
  /** A dimension or hyperparameter was clicked: trace it through the model (again to stop). */
  onTrace(token: string): void;
  onLookInside(blockId: string): void;
  onOutside(): void;
  ask(blockId: string, question: string, history: AskTurn[], onDelta: (text: string) => void): Promise<void>;
}

const START: { kind: LearnKind; title: string; body: string }[] = [
  { kind: "walkthrough", title: "Walk me through it", body: "A guided tour, block by block, in the order data flows." },
  { kind: "example", title: "Easy example", body: "An everyday analogy, then a tiny input followed through the model." },
  { kind: "pytorch", title: "Show me the PyTorch", body: "The architecture as code, with the paper's hyperparameters." },
];

// Blocks whose inside is just what they are: nothing to open.
const SOLID = new Set(["input", "output", "dropout", "activation", "softmax", "residual_add", "concat"]);

/** Hyperparameter chips that start a trace of their first name ("d_k = d_v" traces d_k, and 64 with it). */
const chips = (hps: { name: string; value: string }[], traced: string | null) =>
  hps
    .map((h) => {
      const token = h.name.split(/[=,]/)[0]!.trim();
      return `<button type="button" class="chip" data-dim="${escapeHtml(token)}" aria-pressed="${token === traced}" title="Trace ${escapeHtml(token)} through the model">${escapeHtml(h.name)} = <b>${escapeHtml(h.value)}</b></button>`;
    })
    .join("");

export class Overview {
  private graph: LocatedArchitecture | null = null;
  /** The model this is the inside of (its name), when showing a block's inside view. */
  private insideOf: string | null = null;
  private traced: string | null = null;
  private history = new Map<string, AskTurn[]>();
  private asking = false;
  private shown: string | null = null; // what is rendered, so a hover can't wipe a typed question

  constructor(
    private root: HTMLElement,
    private deps: OverviewDeps,
  ) {
    root.addEventListener("click", (e) => {
      const el = (e.target as HTMLElement).closest<HTMLElement>("[data-start], [data-show-in-paper], [data-dim], [data-inside], [data-outside]");
      if (el?.dataset.start) deps.onStart(el.dataset.start as LearnKind);
      if (el?.dataset.showInPaper) deps.onShowInPaper(el.dataset.showInPaper);
      if (el?.dataset.dim) deps.onTrace(el.dataset.dim);
      if (el?.dataset.inside) deps.onLookInside(el.dataset.inside);
      if (el?.dataset.outside !== undefined) deps.onOutside();
    });
    root.addEventListener("submit", (e) => {
      e.preventDefault();
      void this.submit(e.target as HTMLFormElement);
    });
  }

  /** True while a question is being typed or answered: hover must not replace the view. */
  get askBusy(): boolean {
    const input = this.root.querySelector<HTMLInputElement>(".ask-form input");
    return this.asking || (!!input && (document.activeElement === input || input.value.trim() !== ""));
  }

  setGraph(g: LocatedArchitecture | null) {
    this.graph = g;
    this.insideOf = null;
    this.traced = null;
    this.history.clear();
    this.shown = null;
    this.showBlock(null, null);
  }

  /** Show a block's inside view (or the whole model again with null); conversations stay. */
  setInside(g: LocatedArchitecture, insideOf: string | null) {
    this.graph = g;
    this.insideOf = insideOf;
    this.traced = null;
    this.shown = null;
  }

  /** Mark which dimension is being traced (null: none); re-renders on the next show. */
  setTraced(token: string | null) {
    this.traced = token;
    if (!this.askBusy) this.shown = null;
  }

  /** The same paper in another language: conversations stay, and an answer being typed or streamed isn't wiped. */
  relabel(g: LocatedArchitecture) {
    this.graph = g;
    if (!this.askBusy) this.shown = null;
  }

  /** Show a block (with Ask when it is the pinned one), or the paper summary for null. */
  showBlock(id: string | null, pinned: string | null) {
    const block = id ? this.graph?.blocks.find((b) => b.id === id) : undefined;
    const view = block ? `${block.id}:${block.id === pinned ? "pinned" : ""}:${this.history.get(block.id)?.length ?? 0}` : "summary";
    if (view === this.shown) return;
    this.shown = view;
    this.root.innerHTML = block ? this.blockHtml(block, block.id === pinned) : this.summaryHtml();
  }

  private summaryHtml(): string {
    const g = this.graph;
    if (!g) return "";
    if (this.insideOf) {
      return `
      <div class="ov-summary">
        <p class="ov-inside">Inside the ${escapeHtml(this.insideOf)}</p>
        <h2 class="ov-model">${escapeHtml(g.model_name)}</h2>
        <p class="ov-text">${escapeHtml(g.summary)}</p>
        ${g.hyperparameters.length ? `<div class="ov-hp">${chips(g.hyperparameters, this.traced)}</div>` : ""}
      </div>
      <button type="button" class="btn" data-outside>← Back to the whole model</button>
      <p class="ov-hint">Hover a part to see the sentence it came from. Click a dimension to trace it.</p>`;
    }
    return `
      <div class="ov-summary">
        <h2 class="ov-model">${escapeHtml(g.model_name)}</h2>
        <p class="ov-paper">${escapeHtml(g.paper_title)}, arXiv:${escapeHtml(g.arxiv_id)}</p>
        <p class="ov-text">${escapeHtml(g.summary)}</p>
        ${g.hyperparameters.length ? `<div class="ov-hp">${chips(g.hyperparameters, this.traced)}</div>` : ""}
      </div>
      <div class="ov-start">
        ${START.map((s) => `<button type="button" class="start-card" data-start="${s.kind}"><b>${s.title}</b><span>${s.body}</span></button>`).join("")}
      </div>
      <p class="ov-hint">Hover a block to see the sentence it came from. Click it to keep it here and ask about it; double-click to look inside. Click a dimension like 512 to trace it.</p>`;
  }

  private blockHtml(b: LocatedBlock, pinned: boolean): string {
    const g = this.graph!;
    const page = b.source ? b.source.page + 1 : null;
    const turns = this.history.get(b.id) ?? [];
    return `
      <p class="in-type">${escapeHtml(b.type.replace(/_/g, " "))}${b.group ? `, in ${escapeHtml(b.group.replace(/_/g, " "))}` : ""}</p>
      <h2 class="in-title"><span class="dot" style="background:${TYPE_COLOR[b.type]}"></span>${escapeHtml(b.label)}</h2>
      <p class="in-shape">${shapeHtml(b.input_shape, this.traced)} → ${shapeHtml(b.output_shape, this.traced)}</p>
      ${b.hyperparameters.length ? `<div class="in-hp">${chips(b.hyperparameters, this.traced)}</div>` : ""}
      ${b.equation ? `<div class="in-eq">${renderTex(b.equation, true)}</div>` : ""}
      <p class="in-desc">${escapeHtml(b.description)}</p>
      <figure class="in-source">
        <blockquote class="in-quote">${escapeHtml(b.source_quote)}</blockquote>
        <figcaption class="in-cite ${page ? "" : "missing"}">
          ${page ? `Page ${page} of the paper <button type="button" class="link" data-show-in-paper="${escapeHtml(b.id)}">Show in paper</button>` : "This sentence wasn't found word for word in the extracted text."}
        </figcaption>
      </figure>
      ${
        this.insideOf
          ? `<p class="ov-hint">Part of the ${escapeHtml(g.model_name)}. <button type="button" class="link" data-outside>Back to the whole model</button> to ask about it.</p>`
          : !SOLID.has(b.type) ? `<button type="button" class="btn in-inside" data-inside="${escapeHtml(b.id)}">Look inside ${escapeHtml(b.label)}</button>` : ""
      }
      ${
        this.insideOf ? ""
        : pinned
          ? `<div class="in-ask">
              ${turns.map((t) => `<p class="ask-${t.role}">${escapeHtml(t.text)}</p>`).join("")}
              <form class="ask-form"><label class="sr-only" for="ask-q">Ask about ${escapeHtml(b.label)}</label><input id="ask-q" name="q" maxlength="1000" placeholder="Ask about this block" autocomplete="off" /><button>Ask</button></form>
            </div>`
          : `<p class="ov-hint">Click the block to ask about it.</p>`
      }`;
  }

  private async submit(form: HTMLFormElement) {
    const input = form.elements.namedItem("q") as HTMLInputElement;
    const question = input.value.trim();
    const blockId = this.shown?.split(":")[0];
    const g = this.graph;
    if (!question || !blockId || !g) return;
    const history = this.history.get(blockId) ?? [];
    this.asking = true;
    const answer = document.createElement("p");
    answer.className = "ask-assistant streaming";
    form.before(Object.assign(document.createElement("p"), { className: "ask-user", textContent: question }), answer);
    input.value = "";
    input.disabled = true;
    let text = "";
    try {
      await this.deps.ask(blockId, question, history, (d) => (answer.textContent = text += d));
      if (this.graph === g) this.history.set(blockId, [...history, { role: "user", text: question }, { role: "assistant", text }]);
    } catch (err) {
      answer.textContent = err instanceof LearnError || err instanceof Error ? err.message : String(err);
      answer.classList.add("ask-error");
    } finally {
      this.asking = false;
      answer.classList.remove("streaming");
      input.disabled = false;
    }
  }
}
