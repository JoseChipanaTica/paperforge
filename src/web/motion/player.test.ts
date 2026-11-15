import { describe, expect, test } from "bun:test";
import type { Frame } from "./frames";
import { Player } from "./player";

/** Just enough DOM for the player: elements with attributes, text and children. */
class FakeEl {
  attrs = new Map<string, string>();
  children: FakeEl[] = [];
  parent: FakeEl | null = null;
  textContent = "";
  connected = true;
  constructor(readonly tag: string) {}
  setAttribute(k: string, v: string) {
    this.attrs.set(k, v);
  }
  getAttribute(k: string) {
    return this.attrs.get(k) ?? null;
  }
  appendChild(c: FakeEl) {
    c.remove(); // like the DOM: appending an existing child moves it
    c.parent = this;
    this.children.push(c);
    return c;
  }
  remove() {
    if (this.parent) this.parent.children = this.parent.children.filter((c) => c !== this);
    this.parent = null;
  }
  replaceChildren() {
    this.children = [];
  }
  get isConnected() {
    return this.connected;
  }
}

const doc = { createElementNS: (_ns: string, tag: string) => new FakeEl(tag) };

/** A manual clock: each tick runs the queued animation frames at the given time. */
function clock() {
  let queue: ((t: number) => void)[] = [];
  let now = 0;
  return {
    raf: (cb: (t: number) => void) => (queue.push(cb), queue.length),
    now: () => now,
    /** Advance time, run due frames, then let promise continuations (the next beat) start. */
    async tick(ms: number) {
      now += ms;
      const q = queue;
      queue = [];
      q.forEach((cb) => cb(now));
      for (let i = 0; i < 5; i++) await Promise.resolve();
    },
    get pending() {
      return queue.length;
    },
  };
}

const frames: Frame[] = [
  { dur: 100, items: [{ id: "a", tag: "rect", attrs: { x: 0, y: 0, width: 10, height: 10, fill: "#fff" } }] },
  { dur: 100, items: [{ id: "a", tag: "rect", attrs: { x: 100, y: 0, width: 10, height: 10, fill: "#000" } }, { id: "t", tag: "text", attrs: { x: 5, y: 5 }, text: "hi" }] },
];

describe("Player", () => {
  test("tweens numeric attributes and adds new items", async () => {
    const svg = new FakeEl("svg");
    const c = clock();
    const p = new Player(svg as never, doc as never, c.raf, c.now);
    const done = p.play(frames);
    for (let i = 0; i < 10; i++) await c.tick(25);
    await done;
    const a = svg.children.find((e) => e.attrs.get("data-id") === "a")!;
    expect(a.attrs.get("x")).toBe("100");
    expect(a.attrs.get("fill")).toBe("#000");
    expect(svg.children.find((e) => e.tag === "text")!.textContent).toBe("hi");
  });

  test("halfway through a beat, a moving item is halfway there", async () => {
    const svg = new FakeEl("svg");
    const c = clock();
    const p = new Player(svg as never, doc as never, c.raf, c.now, (t) => t);
    void p.play(frames);
    await c.tick(0);
    await c.tick(100); // first frame done; the second starts now
    await c.tick(50);
    const a = svg.children.find((e) => e.attrs.get("data-id") === "a")!;
    expect(Number(a.attrs.get("x"))).toBeCloseTo(50, 0);
  });

  test("stop ends the loop: no more frames are requested or written", async () => {
    const svg = new FakeEl("svg");
    const c = clock();
    const p = new Player(svg as never, doc as never, c.raf, c.now);
    const done = p.play(frames);
    await c.tick(30);
    p.stop();
    await c.tick(30);
    await done;
    expect(c.pending).toBe(0);
    expect(p.playing).toBe(false);
  });

  test("a detached figure stops writing", async () => {
    const svg = new FakeEl("svg");
    const c = clock();
    const p = new Player(svg as never, doc as never, c.raf, c.now);
    const done = p.play(frames);
    await c.tick(30);
    svg.connected = false;
    const before = svg.children.find((e) => e.attrs.get("data-id") === "a")!.attrs.get("x");
    await c.tick(30);
    await c.tick(30);
    await done;
    expect(svg.children.find((e) => e.attrs.get("data-id") === "a")!.attrs.get("x")).toBe(before);
    expect(c.pending).toBe(0);
  });

  test("paint order follows the frame's item order, even for items added later", async () => {
    const svg = new FakeEl("svg");
    const c = clock();
    const p = new Player(svg as never, doc as never, c.raf, c.now);
    const behind: Frame = { dur: 10, items: [{ id: "ghost", tag: "rect", attrs: { x: 0 } }, { id: "a", tag: "rect", attrs: { x: 0 } }] };
    const done = p.play([{ dur: 10, items: [{ id: "a", tag: "rect", attrs: { x: 0 } }] }, behind]);
    for (let i = 0; i < 6; i++) await c.tick(10);
    await done;
    expect(svg.children.map((e) => e.attrs.get("data-id"))).toEqual(["ghost", "a"]);
  });

  test("showFinal draws the last frame at once", () => {
    const svg = new FakeEl("svg");
    const c = clock();
    const p = new Player(svg as never, doc as never, c.raf, c.now);
    p.showFinal(frames);
    expect(svg.children.find((e) => e.attrs.get("data-id") === "a")!.attrs.get("x")).toBe("100");
    expect(c.pending).toBe(0);
  });
});
