import * as THREE from "three";
import { CSS2DObject } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import type { View } from "../shared/learn";
import { ease, type Run } from "./anim";
import { HIGHLIGHTER, INK, SHEET, TYPE_COLOR, VERMILION } from "./blocks";
import type { ForgeScene } from "./scene";
import { CELL_PITCH, colLabelMode, layoutViews, rowLabelRoom, rowLabelsOf, type Cell, type Laid } from "./specimenLayout";

/**
 * The specimen: the journey's data, drawn as a small printed card that floats beside the block
 * the tour is on. Between steps it shrinks into a packet, rides the wire to the next block and
 * unfolds into its new form; cells with the same key ("cat") start where they were.
 */

const UP = new THREE.Vector3(0, 1, 0);
const CARD_PAD = 0.18;
/** Cells are laid out in small units; the card is printed this much larger next to a block. */
const SCALE = 1.6;
const CAMERA_MS = 720; // scene.focusOn's flight, plus a frame
const PALE = new THREE.Color("#fbf1cf");
const HL = new THREE.Color(HIGHLIGHTER);
const VM = new THREE.Color(VERMILION);
const HEADS = new THREE.Color(TYPE_COLOR.multi_head_attention);
const WHITE = new THREE.Color("#ffffff");

/**
 * Beside the block on the camera's right (or left, with side -1), a little above it, far enough out not to cover it, and
 * pulled toward the camera so a neighbouring column of blocks can't cut through the card.
 */
export function specimenOffset(camera: THREE.Vector3, target: THREE.Vector3, blockSize: number, width = 0, side: 1 | -1 = 1): THREE.Vector3 {
  const forward = target.clone().sub(camera).setY(0);
  if (forward.lengthSq() < 1e-9) forward.set(0, 0, -1);
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, UP).normalize();
  return target
    .clone()
    .addScaledVector(right, side * (blockSize / 2 + 0.4 + width / 2))
    .addScaledVector(forward, -2.5)
    .add(new THREE.Vector3(0, 0.6, 0));
}

/** A path over the model between two blocks no wire joins (the encoder's end to the decoder's start). */
export function arcBetween(a: THREE.Vector3, b: THREE.Vector3): THREE.QuadraticBezierCurve3 {
  const top = Math.max(a.y, b.y);
  // The control point that puts the curve's middle 1.2 above the higher end.
  const c = new THREE.Vector3((a.x + b.x) / 2, 2 * top - (a.y + b.y) / 2 + 2.4, (a.z + b.z) / 2);
  return new THREE.QuadraticBezierCurve3(a.clone(), c, b.clone());
}

/** Two moments show the same data when everything but their captions matches (a new language, say). */
export const sameData = (a: View[], b: View[]) => {
  const data = (vs: View[]) => JSON.stringify(vs.map(({ caption: _, ...rest }) => rest));
  return data(a) === data(b);
};

function shadeColor(cell: Cell, split: number): THREE.Color {
  const s = Math.max(0, Math.min(1, cell.shade));
  const c = s < 0.5 ? PALE.clone().lerp(HL, s * 2) : HL.clone().lerp(VM, (s - 0.5) * 2);
  if (cell.slice === undefined) return c;
  // Heads: each a lighter print of the attention colour, still shaded by value.
  return HEADS.clone().lerp(WHITE, 0.1 + (0.55 * cell.slice) / Math.max(1, split)).lerp(c, 0.35);
}

const esc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);

function label(text: string, cls: string, at: THREE.Vector3, center: [number, number]): CSS2DObject {
  const div = document.createElement("div");
  div.className = `specimen-label ${cls}`;
  div.innerHTML = esc(text);
  const obj = new CSS2DObject(div);
  obj.position.copy(at);
  obj.center.set(...center);
  return obj;
}

interface Built {
  group: THREE.Group;
  mesh: THREE.InstancedMesh;
  cells: Cell[];
  width: number;
  scale: number;
  at: string;
  views: View[];
  captions: HTMLElement[]; // one per view, in order
  /** Each matrix's column labels and how they print now (colLabelMode), rechecked as the camera moves. */
  cols: { labels: HTMLElement[]; texts: string[]; mode: string }[];
}

export class Specimen {
  private current: Built | null = null;
  private token = 0;
  private packet: THREE.Mesh;
  private tip: CSS2DObject;
  private cellGeometry = new THREE.BoxGeometry(1, 1, 1);
  private matrix = new THREE.Matrix4();

  constructor(private scene: ForgeScene) {
    this.packet = new THREE.Mesh(new THREE.SphereGeometry(0.25, 20, 14), new THREE.MeshBasicMaterial({ color: VERMILION }));
    this.packet.visible = false;
    scene.scene.add(this.packet);
    const tip = document.createElement("div");
    tip.className = "specimen-tip";
    this.tip = new CSS2DObject(tip);
    this.tip.center.set(0.5, 1.3);
    this.tip.visible = false;
    scene.scene.add(this.tip);
    scene.frameHooks.push(() => this.frame());
    scene.occluders = () => (this.current ? [this.current.group] : []);
  }

  get visible() {
    return this.current !== null;
  }

  /** Show these views beside `at`; with `from`, the old specimen flies there along the wire first. */
  async show(views: View[], at: string, from: string | null, run: Run): Promise<void> {
    const token = ++this.token;
    const started = performance.now();
    const anchor = this.scene.blockAnchor(at);
    if (!anchor || !views.length) return this.hide();
    const old = this.current;
    if (old && old.at === at && sameData(old.views, views)) {
      // Same data at the same block (a language or level change): only the words change.
      views.forEach((v, i) => old.captions[i] && (old.captions[i]!.textContent = v.caption));
      old.views = views;
      return;
    }
    const before = new Map(old?.cells.map((c) => [c.key, c] as const));

    if (old && from && from !== at) {
      await run.tween(220, (t) => old.group.scale.setScalar(old.scale * (1 - 0.92 * t)), ease.inCubic);
      this.drop(old);
      const path = this.scene.pathBetween(from, at);
      if (path && token === this.token) {
        this.packet.visible = true;
        await run.tween(600, (t) => path.getPoint(t, this.packet.position));
        this.packet.visible = false;
      }
    } else if (old) {
      await run.tween(160, (t) => old.group.scale.setScalar(old.scale * (1 - t)), ease.inCubic);
      this.drop(old);
    }
    // Measure where the card fits only once the camera has arrived at the block.
    await run.wait(Math.max(0, CAMERA_MS - (performance.now() - started)));
    if (token !== this.token) return;

    const built = this.build(views, at, anchor, this.scene.blockSize(at));
    this.current = built;
    this.face(built.group);
    this.scene.scene.add(built.group);

    const n = built.cells.length;
    const place = (t: number) => {
      built.cells.forEach((c, i) => {
        // Cells unfold left to right; a row that was already there slides from where it was.
        const local = Math.max(0, Math.min(1, t * 1.6 - (i / Math.max(1, n)) * 0.6));
        const prev = before.get(c.key);
        const k = ease.outCubic(local);
        const s = prev ? 1 : k;
        const x = prev ? prev.x + (c.x - prev.x) * k : c.x;
        const y = prev ? prev.y + (c.y - prev.y) * k : c.y;
        this.matrix.compose(new THREE.Vector3(x, y, c.d / 2), new THREE.Quaternion(), new THREE.Vector3(c.w * s || 1e-4, c.h * s || 1e-4, c.d * s || 1e-4));
        built.mesh.setMatrixAt(i, this.matrix);
      });
      built.mesh.instanceMatrix.needsUpdate = true;
      for (const child of built.group.children) if (child instanceof CSS2DObject) (child.element as HTMLElement).style.opacity = String(Math.min(1, t * 1.5));
    };
    place(0);
    await run.tween(480, place, ease.linear);
  }

  /**
   * Beside the block, opposite its label, shrunk until its outer edge is in view: a wide card next
   * to the right-hand column would otherwise run under the study panel.
   */
  private place(group: THREE.Group, width: number, at: string, anchor: THREE.Vector3, blockSize: number): number {
    const camera = this.scene.camera;
    // Away from the block's own label; with the label on top, toward the roomier side of the screen.
    const labelSide = this.scene.labelSide(at);
    const side = labelSide === "right" ? -1 : labelSide === "left" ? 1 : anchor.clone().project(camera).x > 0 ? -1 : 1;
    const right = new THREE.Vector3().crossVectors(anchor.clone().sub(camera.position).setY(0).normalize(), UP).normalize();
    let scale = SCALE;
    for (let i = 0; i < 6; i++) {
      group.position.copy(specimenOffset(camera.position, anchor, blockSize, width * scale, side));
      const edge = group.position.clone().addScaledVector(right, (side * width * scale) / 2).project(camera).x;
      if (Math.abs(edge) <= 0.94) break;
      scale *= 0.82;
    }
    group.scale.setScalar(scale);
    return scale;
  }

  /** Screen pixels per world unit at a point (the card's labels are fixed-size DOM text). */
  private pxPerWorld(at: THREE.Vector3): number {
    const camera = this.scene.camera;
    const h = this.scene.renderer.domElement.clientHeight || 1;
    const dist = Math.max(1e-3, camera.position.distanceTo(at));
    return h / (2 * dist * Math.tan(THREE.MathUtils.degToRad(camera.fov / 2)));
  }

  /** Take the specimen off the model (the tour ended, or it no longer applies). */
  hide() {
    this.token++;
    this.packet.visible = false;
    this.tip.visible = false;
    if (this.current) this.drop(this.current);
  }

  private drop(b: Built) {
    // CSS2D elements stay in the DOM unless removed one by one.
    b.group.traverse((o) => {
      if (o instanceof CSS2DObject) o.element.remove();
      const mesh = o as THREE.Mesh;
      if (mesh.isMesh) {
        if (mesh.geometry !== this.cellGeometry) mesh.geometry.dispose();
        (mesh.material as THREE.Material).dispose();
      }
    });
    b.group.removeFromParent();
    if (this.current === b) this.current = null;
  }

  /**
   * The card, placed: its row labels get room measured at the size it ends up printed (a smaller card needs relatively
   * more), so the fit check and the printed sheet include them.
   */
  private build(views: View[], at: string, anchor: THREE.Vector3, blockSize: number): Built {
    const group = new THREE.Group();
    const labels = views.slice(0, 2).map(rowLabelsOf);
    const extent = (l: Laid[]) =>
      Math.max(...l.flatMap((x) => x.cells.map((c) => c.x + c.w / 2))) - Math.min(...l.flatMap((x) => x.cells.map((c) => c.x - c.w / 2)));
    let scale = SCALE;
    let rooms: number[] = [];
    let laid = layoutViews(views);
    let width = 0;
    for (let i = 0; i < 4; i++) {
      const px = this.pxPerWorld(anchor) * scale;
      rooms = labels.map((ls) => rowLabelRoom(ls, px));
      laid = layoutViews(views, rooms);
      // Symmetric, so the fit check holds on either side.
      width = extent(laid) + 2 * Math.max(CARD_PAD, rooms[0] ?? 0);
      const next = this.place(group, width, at, anchor, blockSize);
      if (Math.abs(next - scale) < 1e-6) break;
      scale = next;
    }
    const cells = laid.flatMap((l) => l.cells);
    const splits = laid.flatMap((l, i) => l.cells.map(() => views[i]!.split));
    const mesh = new THREE.InstancedMesh(this.cellGeometry, new THREE.MeshStandardMaterial({ roughness: 0.9, metalness: 0 }), cells.length);
    cells.forEach((c, i) => mesh.setColorAt(i, shadeColor(c, splits[i]!)));
    mesh.userData.cells = cells;
    mesh.userData.occludes = true;
    group.add(mesh);
    const cols: Built["cols"] = [];
    const captions = laid.map((l, i) => this.decorate(group, l, views[i]!, cols));
    return { group, mesh, cells, width, scale, at, views, captions, cols };
  }

  /** The printed card behind a view, its row labels, caption and "…512" cap. */
  private decorate(group: THREE.Group, l: Laid, v: View, cols: Built["cols"]): HTMLElement {
    const xs = l.cells.map((c) => c.x);
    // The sheet reaches past the longest row label on the left.
    const leftPad = Math.max(CARD_PAD, (l.labelRoom ?? 0) + 0.04);
    const cx = (Math.min(...l.cells.map((c) => c.x - c.w / 2)) - leftPad + Math.max(...l.cells.map((c) => c.x + c.w / 2)) + CARD_PAD) / 2;
    const w = l.width + leftPad + CARD_PAD;
    const h = l.height + 2 * CARD_PAD;
    const plane = (color: string, dx: number, dy: number, z: number, pad: number) => {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(w + pad, h + pad), new THREE.MeshBasicMaterial({ color }));
      m.position.set(cx + dx, dy, z);
      m.userData.occludes = true;
      return m;
    };
    group.add(plane(INK, 0.09, -0.09, -0.06, 0), plane(INK, 0, 0, -0.05, 0.05), plane(SHEET, 0, 0, -0.04, 0));

    for (const c of l.cells) {
      if (!c.label) continue;
      if (v.kind === "tokens") group.add(label(c.label, "on", new THREE.Vector3(c.x, c.y, c.d + 0.01), [0.5, 0.5]));
      else if (v.kind === "patches") group.add(label(c.label, "on small", new THREE.Vector3(c.x, c.y, c.d + 0.01), [0.5, 0.5]));
      else group.add(label(c.label, "left", new THREE.Vector3(c.x - c.w / 2 - 0.06, c.y, 0), [1, 0.5]));
    }
    if (v.kind === "matrix" && v.cols.length) {
      const top = Math.max(...l.cells.map((c) => c.y + c.h / 2));
      const entry = { labels: [] as HTMLElement[], texts: v.cols, mode: "flat" };
      for (const [j, col] of v.cols.entries()) {
        const c = l.cells.find((cell) => cell.key.endsWith(`:${col}`)) ?? l.cells[j];
        if (!c) continue;
        const obj = label(col, "col", new THREE.Vector3(c.x, top + 0.04, 0), [0.5, 1]);
        entry.labels.push(obj.element as HTMLElement);
        group.add(obj);
      }
      cols.push(entry);
    }
    if (l.more) {
      const right = Math.max(...xs.map((x, i) => x + l.cells[i]!.w / 2));
      const top = Math.max(...l.cells.map((c) => c.y));
      group.add(label(l.more, "more", new THREE.Vector3(right + 0.06, top, 0), [0, 0.5]));
    }
    const caption = label(l.caption, "caption", new THREE.Vector3(cx, -h / 2 - 0.06, 0), [0.5, 0]);
    group.add(caption);
    return caption.element as HTMLElement;
  }

  /** Turn to the camera around the vertical axis only, like a card standing on the table. */
  private face(g: THREE.Object3D) {
    const cam = this.scene.camera.position;
    g.rotation.set(0, Math.atan2(cam.x - g.position.x, cam.z - g.position.z), 0);
  }

  private frame() {
    const b = this.current;
    if (!b) return;
    this.face(b.group);
    // Column labels side by side, standing on end, or left to the tooltip, as the card's size on screen allows.
    if (b.cols.length) {
      const pitch = CELL_PITCH * b.scale * this.pxPerWorld(b.group.position);
      for (const c of b.cols) {
        const mode = colLabelMode(c.texts, pitch);
        if (mode === c.mode) continue;
        c.mode = mode;
        for (const el of c.labels) {
          el.classList.toggle("rotated", mode === "rotated");
          el.hidden = mode === "none";
        }
      }
    }
    const hit = this.scene.hits([b.mesh])[0];
    const cell = hit?.instanceId !== undefined ? b.cells[hit.instanceId] : undefined;
    this.tip.visible = !!cell;
    if (!cell) return;
    (this.tip.element as HTMLElement).textContent = cell.tip;
    this.tip.position.copy(b.group.localToWorld(new THREE.Vector3(cell.x, cell.y + cell.h / 2, cell.d)));
  }
}
