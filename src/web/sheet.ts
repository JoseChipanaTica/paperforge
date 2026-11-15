import { snapSheet } from "./panelState";

/**
 * On phones the study panel is a bottom sheet with three resting heights: just the tabs,
 * half the screen, nearly full. Drag the handle (a flick carries it to the next snap),
 * tap the handle to toggle tabs / half. Inactive on desktop.
 */

export type SheetSnap = "tabs" | "half" | "full";
const TABS_H = 64;

export class Sheet {
  readonly media = matchMedia("(max-width: 899px)");
  private height = TABS_H;

  constructor(
    private panel: HTMLElement,
    private handle: HTMLButtonElement,
  ) {
    let startY = 0;
    let startH = 0;
    let lastY = 0;
    let lastT = 0;
    let velocity = 0;
    let moved = false;

    handle.addEventListener("pointerdown", (e) => {
      if (!this.active) return;
      handle.setPointerCapture(e.pointerId);
      startY = lastY = e.clientY;
      startH = this.height;
      lastT = e.timeStamp;
      velocity = 0;
      moved = false;
      panel.classList.add("dragging");
    });
    handle.addEventListener("pointermove", (e) => {
      if (!handle.hasPointerCapture(e.pointerId)) return;
      const dt = Math.max(1, e.timeStamp - lastT);
      velocity = ((lastY - e.clientY) / dt) * 1000; // px/s, positive = up
      lastY = e.clientY;
      lastT = e.timeStamp;
      if (Math.abs(e.clientY - startY) > 4) moved = true;
      this.set(Math.min(this.snaps()[2]!, Math.max(TABS_H, startH + (startY - e.clientY))));
    });
    const end = (e: PointerEvent) => {
      if (!handle.hasPointerCapture(e.pointerId)) return;
      handle.releasePointerCapture(e.pointerId);
      panel.classList.remove("dragging");
      if (moved) this.set(snapSheet(this.height, velocity, this.snaps()));
      else this.snapTo(this.open ? "tabs" : "half");
    };
    handle.addEventListener("pointerup", end);
    handle.addEventListener("pointercancel", end);
    // Keyboard: the handle is a button; Enter/Space toggle like a tap.
    handle.addEventListener("click", (e) => {
      if (e.detail === 0 && this.active) this.snapTo(this.open ? "tabs" : "half");
    });

    this.media.addEventListener("change", () => this.apply());
    addEventListener("resize", () => this.active && this.set(Math.min(this.height, this.snaps()[2]!)));
    this.apply();
  }

  get active() {
    return this.media.matches;
  }

  get open() {
    return this.height > TABS_H + 8;
  }

  snapTo(which: SheetSnap) {
    if (!this.active) return;
    const [tabs, half, full] = this.snaps();
    this.set(which === "tabs" ? tabs! : which === "half" ? half! : full!);
  }

  /** Open to half if only the tabs are showing (tapping a tab or a block). */
  reveal() {
    if (this.active && !this.open) this.snapTo("half");
  }

  private snaps(): number[] {
    const h = innerHeight;
    return [TABS_H, Math.round(h * 0.45), Math.round(h * 0.9 - 56)];
  }

  private set(px: number) {
    this.height = px;
    this.panel.style.setProperty("--sheet-h", `${px}px`);
    this.handle.setAttribute("aria-expanded", String(this.open));
    this.handle.setAttribute("aria-label", this.open ? "Collapse the study panel" : "Expand the study panel");
  }

  private apply() {
    this.panel.classList.toggle("sheet", this.active);
    this.set(TABS_H);
  }
}
