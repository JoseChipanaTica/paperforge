import { ease } from "../anim";
import type { Frame, Item } from "./frames";

/**
 * Plays frames into an <svg>: items are matched by id, numeric attributes tween between frames,
 * new items fade in, missing ones fade out. Built with createElementNS only (never innerHTML).
 * `stop()`, or the SVG leaving the page, ends the loop at the next animation frame.
 */

const NS = "http://www.w3.org/2000/svg";
const TWEENED = new Set(["x", "y", "width", "height", "cx", "cy", "r", "x1", "y1", "x2", "y2", "opacity"]);

type Doc = Pick<Document, "createElementNS">;
type Raf = (cb: (t: number) => void) => number;

export class Player {
  private els = new Map<string, Element>();
  private values = new Map<string, Record<string, number>>(); // current numeric attributes per item
  private token = 0;
  private running = false;

  constructor(
    private svg: SVGSVGElement,
    private doc: Doc = document,
    private raf: Raf = (cb) => requestAnimationFrame(cb),
    private now: () => number = () => performance.now(),
    private easing: (t: number) => number = ease.inOutCubic,
  ) {}

  get playing() {
    return this.running;
  }

  /** Play every frame in order; resolves when done or stopped. */
  async play(frames: Frame[]): Promise<void> {
    const token = ++this.token;
    this.running = true;
    for (const f of frames) {
      const finished = await this.tweenTo(f, token);
      if (!finished) break;
    }
    if (token === this.token) this.running = false;
  }

  /** Jump to the last frame (reduced motion, or a step shown again). */
  showFinal(frames: Frame[]) {
    this.stop();
    const last = frames.at(-1);
    if (!last) return;
    const keep = new Set(last.items.map((i) => i.id));
    for (const [id, el] of this.els) if (!keep.has(id)) this.drop(id, el);
    for (const item of last.items) this.apply(item, this.element(item), 1, item.attrs);
  }

  stop() {
    this.token++;
    this.running = false;
  }

  /** Clear the figure (a new chain is about to play). */
  clear() {
    this.stop();
    for (const [id, el] of this.els) this.drop(id, el);
  }

  private element(item: Item): Element {
    let el = this.els.get(item.id);
    if (!el) {
      el = this.doc.createElementNS(NS, item.tag);
      el.setAttribute("data-id", item.id);
      this.svg.appendChild(el);
      this.els.set(item.id, el);
    }
    return el;
  }

  private drop(id: string, el: Element) {
    el.remove();
    this.els.delete(id);
    this.values.delete(id);
  }

  /** Set an item's attributes at tween position t from `from` (numeric ones interpolate). */
  private apply(item: Item, el: Element, t: number, from: Record<string, string | number>) {
    const now: Record<string, number> = {};
    for (const [k, v] of Object.entries(item.attrs)) {
      if (typeof v === "number" && TWEENED.has(k)) {
        const a = typeof from[k] === "number" ? (from[k] as number) : v;
        now[k] = a + (v - a) * t;
        el.setAttribute(k, String(Math.round(now[k]! * 100) / 100));
      } else if (t === 0 || t === 1 || !(k in from)) el.setAttribute(k, String(v));
    }
    if (item.text !== undefined && el.textContent !== item.text) el.textContent = item.text;
    this.values.set(item.id, now);
  }

  private tweenTo(f: Frame, token: number): Promise<boolean> {
    const next = new Set(f.items.map((i) => i.id));
    const leaving = [...this.els].filter(([id]) => !next.has(id));
    const starts = f.items.map((item) => {
      const fresh = !this.els.has(item.id);
      const el = this.element(item);
      // A new item fades in from transparent, at its final place.
      const from: Record<string, string | number> = fresh ? { ...item.attrs, opacity: 0 } : { ...item.attrs, ...this.values.get(item.id) };
      for (const [k, v] of Object.entries(item.attrs)) if (typeof v !== "number" || !TWEENED.has(k)) el.setAttribute(k, String(v));
      if (item.text !== undefined) el.textContent = item.text;
      return { item, el, from };
    });
    const begin = this.now();
    return new Promise((resolve) => {
      const step = (time: number) => {
        if (token !== this.token || !this.svg.isConnected) return resolve(false);
        const raw = f.dur > 0 ? Math.min(1, (time - begin) / f.dur) : 1;
        const t = this.easing(raw);
        for (const s of starts) this.apply(s.item, s.el, t, s.from);
        for (const [, el] of leaving) el.setAttribute("opacity", String(1 - t));
        if (raw < 1) this.raf(step);
        else {
          for (const [id, el] of leaving) this.drop(id, el);
          resolve(true);
        }
      };
      this.raf(step);
    });
  }
}
