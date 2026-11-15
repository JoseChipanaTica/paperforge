import katex from "katex";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { CSS2DObject, CSS2DRenderer } from "three/examples/jsm/renderers/CSS2DRenderer.js";
import type { Edge, LocatedArchitecture, LocatedBlock } from "../shared/graph";
import type { Run } from "./anim";
import { INK, VERMILION, buildBlock, buildGroupFrame, type BlockObject } from "./blocks";
import { layoutGraph, type Layout, type PlacedBlock, type PlacedGroup } from "./layout";
import { arcBetween } from "./specimen";

export interface SceneBlock {
  block: LocatedBlock;
  placed: PlacedBlock;
  obj: BlockObject;
  label: CSS2DObject;
  labelSide: "left" | "right" | "top";
  assembled: THREE.Vector3;
  exploded: THREE.Vector3;
  /** 0 = at rest, decays from 1 after a flash */
  glow: number;
  index: number;
}

interface SceneEdge {
  edge: Edge;
  mesh: THREE.Mesh;
  curve: THREE.Curve<THREE.Vector3>;
}

interface Pulse {
  edge: SceneEdge;
  start: number; // seconds into the pulse cycle
  dots: THREE.Mesh[];
}

const QUIET_TYPES = new Set(["add_norm", "layer_norm", "normalization", "residual_add", "dropout", "activation"]);

const titleHtml = (block: LocatedBlock, color: THREE.Color) =>
  `<span class="dot" style="background:${color.getStyle()}"></span>${escapeHtml(block.label)}<span class="bl-shape">${escapeHtml(block.output_shape)}</span>`;
// Printed wires: ink for the data path, ochre for skips, vermilion between branches.
const EDGE_COLOR = { data: INK, residual: "#e0b04c", cross: VERMILION } as const;
const PULSE_STEP = 0.55; // seconds per rank
const PULSE_PAUSE = 1.6;

export function renderTex(tex: string, display = false): string {
  try {
    return katex.renderToString(tex, { throwOnError: false, displayMode: display, output: "html" });
  } catch {
    return `<code>${tex.replace(/</g, "&lt;")}</code>`;
  }
}

export class ForgeScene {
  readonly renderer: THREE.WebGLRenderer;
  readonly camera = new THREE.PerspectiveCamera(32, 1, 0.1, 400);
  readonly scene = new THREE.Scene();
  readonly controls: OrbitControls;
  private labels: CSS2DRenderer;
  private composer: EffectComposer;
  private model = new THREE.Group();
  private edgeGroup = new THREE.Group();
  private pulseGroup = new THREE.Group();
  private groupFrames: { placed: PlacedGroup; frame: THREE.Group; label: CSS2DObject }[] = [];
  private raycaster = new THREE.Raycaster();
  private pointer = new THREE.Vector2(9, 9);
  private clock = new THREE.Clock();

  blocks = new Map<string, SceneBlock>();
  edges: SceneEdge[] = [];
  layout: Layout | null = null;
  graph: LocatedArchitecture | null = null;

  explode = 0;
  edgeProgress = 0;
  groupAlpha = 0;
  labelAlpha = 0;
  pulsing = false;
  private pulseClock = 0;
  private pulses: Pulse[] = [];
  private pulseCycle = 1;

  /** Horizontal shift of the model on screen, as a fraction of the canvas width. */
  viewShift = 0;
  /** Camera sways gently until the user grabs the controls. */
  sway = true;
  /** prefers-reduced-motion: load() leaves the camera still. */
  reduceMotion = false;
  private swayTime = 0;
  fitDistance = 30;
  private hovered: string | null = null;
  /** A block lit up by the tour or the code panel, independent of the mouse. */
  private highlighted: string | null = null;
  /** Blocks a traced dimension flows through; everything else fades back. */
  private traced: ReadonlySet<string> | null = null;
  onDoubleClick: (id: string) => void = () => {};
  onHover: (id: string | null) => void = () => {};
  onClick: (id: string | null) => void = () => {};
  /** Things drawn in front of the model (the specimen card) that the pointer can't reach through. */
  occluders: () => THREE.Object3D[] = () => [];
  /** Called every frame before rendering (the specimen faces the camera and tracks hover). */
  frameHooks: ((dt: number) => void)[] = [];

  constructor(private container: HTMLElement) {
    // Transparent canvas: the CSS paper shows through.
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true, powerPreference: "high-performance", preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0);
    this.renderer.toneMapping = THREE.NeutralToneMapping;
    this.renderer.toneMappingExposure = 1;
    container.appendChild(this.renderer.domElement);

    this.labels = new CSS2DRenderer();
    this.labels.domElement.className = "label-layer";
    container.appendChild(this.labels.domElement);

    this.scene.background = null;
    const pmrem = new THREE.PMREMGenerator(this.renderer);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    this.scene.environmentIntensity = 0.35;

    // Soft studio light on paper: warm fill from above, one key light, no coloured rim.
    this.scene.add(new THREE.HemisphereLight("#fffaf0", "#d9d0bd", 1.3));
    const key = new THREE.DirectionalLight("#ffffff", 1.5);
    key.position.set(8, 14, 12);
    this.scene.add(key);

    this.scene.add(this.model, this.edgeGroup, this.pulseGroup);

    this.controls = new OrbitControls(this.camera, this.labels.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.addEventListener("start", () => (this.sway = false));

    this.composer = new EffectComposer(this.renderer);
    this.composer.addPass(new RenderPass(this.scene, this.camera));
    this.composer.addPass(new OutputPass());

    const el = this.labels.domElement;
    el.addEventListener("pointermove", (e) => {
      const r = el.getBoundingClientRect();
      this.pointer.set(((e.clientX - r.left) / r.width) * 2 - 1, -((e.clientY - r.top) / r.height) * 2 + 1);
    });
    el.addEventListener("pointerleave", () => this.pointer.set(9, 9));
    let down = { x: 0, y: 0 };
    el.addEventListener("pointerdown", (e) => (down = { x: e.clientX, y: e.clientY }));
    el.addEventListener("click", (e) => {
      if (Math.hypot(e.clientX - down.x, e.clientY - down.y) < 5) this.onClick(this.hovered);
    });
    el.addEventListener("dblclick", () => {
      if (this.hovered) this.onDoubleClick(this.hovered);
    });

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.renderer.setAnimationLoop(() => this.tick());
  }

  private resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h);
    this.composer.setSize(w, h);
    this.labels.setSize(w, h);
    this.camera.aspect = w / h;
    this.applyViewShift();
    this.frame(false);
  }

  private applyViewShift() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (this.viewShift) this.camera.setViewOffset(w, h, -this.viewShift * w, 0, w, h);
    else this.camera.clearViewOffset();
    this.camera.updateProjectionMatrix();
  }

  setViewShift(v: number) {
    this.viewShift = v;
    this.applyViewShift();
  }

  clear() {
    this.model.clear();
    this.edgeGroup.clear();
    this.pulseGroup.clear();
    for (const sb of this.blocks.values()) sb.label.element.remove();
    for (const g of this.groupFrames) g.label.element.remove();
    this.blocks.clear();
    this.edges = [];
    this.pulses = [];
    this.groupFrames = [];
    this.graph = null;
    this.layout = null;
    this.traced = null;
  }

  load(graph: LocatedArchitecture) {
    this.clear();
    this.graph = graph;
    const layout = (this.layout = layoutGraph(graph));
    const midX = (layout.bounds.minX + layout.bounds.maxX) / 2;
    // Blocks side by side in one row (Q, K and V projections): the outer ones label outward, the middle ones above.
    const rows = new Map<string, PlacedBlock[]>();
    for (const p of layout.blocks.values()) {
      const key = `${p.column}:${p.rank}`;
      rows.set(key, [...(rows.get(key) ?? []), p].sort((a, b) => a.x - b.x));
    }

    graph.blocks.forEach((block, index) => {
      const placed = layout.blocks.get(block.id)!;
      const obj = buildBlock(block, placed);
      obj.root.visible = false;
      this.model.add(obj.root);

      const assembled = new THREE.Vector3(placed.x, placed.y, 0);
      // Exploded view: pull rows apart, push alternate blocks forward/back, fan columns out.
      const exploded = new THREE.Vector3(
        midX + (placed.x - midX) * 1.45,
        placed.y * 1.6,
        (index % 2 === 0 ? 1 : -1) * 1.1 + (placed.column % 2 ? -0.6 : 0.6),
      );

      const row = rows.get(`${placed.column}:${placed.rank}`)!;
      const at = row.indexOf(placed);
      const side =
        row.length > 1 ? (at === 0 ? "left" : at === row.length - 1 ? "right" : "top")
        : layout.columns > 1 && placed.x < midX ? "left"
        : "right";
      const div = document.createElement("div");
      div.className = `block-label ${side}`;
      // Norm/residual blocks repeat one equation many times, and labels above sit between rows: one line.
      const showEq = block.equation && !QUIET_TYPES.has(block.type) && side !== "top";
      div.innerHTML = `
        <div class="bl-title">${titleHtml(block, obj.color)}</div>
        ${showEq ? `<div class="bl-eq">${renderTex(block.equation)}</div>` : ""}`;
      div.style.opacity = "0";
      const label = new CSS2DObject(div);
      if (side === "top") {
        label.center.set(0.5, 1);
        label.position.set(0, placed.h / 2 + 0.2, 0);
      } else {
        label.center.set(side === "left" ? 1 : 0, 0.5);
        label.position.set((side === "left" ? -1 : 1) * (placed.w / 2 + 0.35), 0, 0);
      }
      obj.root.add(label);

      this.blocks.set(block.id, { block, placed, obj, label, labelSide: side, assembled, exploded, glow: 0, index });
    });

    for (const g of layout.groups) {
      const frame = buildGroupFrame(g.w, g.h, g.d, g.repeat);
      frame.position.set(g.x, g.y, 0);
      frame.visible = false;
      this.model.add(frame);
      const div = document.createElement("div");
      div.className = "group-label";
      div.innerHTML = `<span>${escapeHtml(g.label)}</span>${g.repeat > 1 ? `<b>×${g.repeat}</b>` : ""}`;
      // Left column: sit on the frame's top edge. Otherwise: tuck into the top-left
      // corner, away from the block labels on the right.
      const leftCol = layout.columns > 1 && g.x < midX;
      const label = new CSS2DObject(div);
      label.center.set(leftCol ? 0 : 1, 1);
      label.position.set(leftCol ? -g.w / 2 : -(g.w / 2 + 0.15), g.h / 2 + (leftCol ? 0.1 : 0), 0);
      frame.add(label);
      div.style.opacity = "0";
      this.groupFrames.push({ placed: g, frame, label });
    }

    this.buildEdges();
    this.buildPulses();
    this.sway = !this.reduceMotion;
    this.frame(true);
  }

  /** The same model in other words (a language change): swap the label text, keep everything else. */
  relabel(graph: LocatedArchitecture) {
    if (!this.graph) return;
    this.graph = graph;
    for (const block of graph.blocks) {
      const sb = this.blocks.get(block.id);
      if (!sb) continue;
      sb.block = block;
      sb.label.element.querySelector(".bl-title")!.innerHTML = titleHtml(block, sb.obj.color);
    }
    const labels = new Map(graph.groups.map((g) => [g.id, g.label]));
    for (const g of this.groupFrames) {
      g.placed.label = labels.get(g.placed.id) ?? g.placed.label;
      g.label.element.querySelector("span")!.textContent = g.placed.label;
    }
  }

  /** Load a graph already assembled: no flight, wires drawn, pulse running (the landing preview). */
  showAssembled(graph: LocatedArchitecture, pad = 1) {
    this.explode = 0;
    this.labelAlpha = 1;
    this.groupAlpha = 1;
    this.load(graph);
    for (const sb of this.blocks.values()) {
      sb.obj.root.visible = true;
      sb.obj.root.position.copy(sb.assembled);
    }
    this.setEdgeProgress(1);
    this.startPulse();
    // Pull the camera back so side labels fit a narrower container (the landing hero).
    const target = this.controls.target;
    this.camera.position.sub(target).multiplyScalar(pad).add(target);
    this.fitDistance *= pad;
  }

  /** Camera distance that fits the model, assembled (0) or exploded (1). */
  distanceFor(explode: number): number {
    if (!this.layout) return 30;
    const b = this.layout.bounds;
    const w = (b.maxX - b.minX) * (1 + 0.45 * explode) + 11; // room for side labels
    const h = (b.maxY - b.minY) * (1.3 + 0.6 * explode) + 4; // and for the top bar
    const vFov = THREE.MathUtils.degToRad(this.camera.fov);
    return Math.max(h / 2 / Math.tan(vFov / 2), w / 2 / (Math.tan(vFov / 2) * this.camera.aspect));
  }

  /** Point the camera at the model so it fills the view. */
  frame(reset: boolean) {
    if (!this.layout) {
      if (reset) {
        this.camera.position.set(0, 0, 30);
        this.controls.target.set(0, 0, 0);
      }
      return;
    }
    const b = this.layout.bounds;
    this.fitDistance = this.distanceFor(this.explode);
    if (reset) {
      const dist = this.fitDistance;
      this.controls.target.set(0, (b.minY + b.maxY) / 2, 0);
      this.camera.position.set(dist * 0.28, this.controls.target.y + dist * 0.12, dist * 0.95);
    }
    this.controls.maxDistance = this.fitDistance * 3;
    this.controls.minDistance = this.fitDistance * 0.15;
  }

  // ---------- blocks ----------

  showBlock(id: string, visible = true) {
    const sb = this.blocks.get(id);
    if (sb) sb.obj.root.visible = visible;
  }

  /** World point on the camera ray through a screen pixel, at a fraction of the view distance. */
  screenToWorld(clientX: number, clientY: number, depth = 0.55): THREE.Vector3 {
    const r = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(((clientX - r.left) / r.width) * 2 - 1, -((clientY - r.top) / r.height) * 2 + 1);
    this.raycaster.setFromCamera(ndc, this.camera);
    const dist = this.camera.position.distanceTo(this.controls.target) * depth;
    return this.raycaster.ray.at(dist, new THREE.Vector3());
  }

  restPosition(sb: SceneBlock, explode = this.explode): THREE.Vector3 {
    return sb.assembled.clone().lerp(sb.exploded, explode);
  }

  setExplode(e: number) {
    this.explode = e;
    for (const sb of this.blocks.values()) {
      if (!sb.obj.root.userData.flying) sb.obj.root.position.copy(this.restPosition(sb));
    }
    this.buildEdges();
    this.buildPulses();
  }

  /** Light up one block (tour step, code symbol) until cleared with null. */
  highlight(id: string | null) {
    this.highlighted = id;
    if (id) this.flash(id, 0.9);
  }

  /** Light up the blocks a traced dimension flows through and fade the rest, or clear with null. */
  setTrace(ids: ReadonlySet<string> | null) {
    this.traced = ids;
    this.fadeEdges();
  }

  /** Edges off the traced path fade with their blocks. */
  private fadeEdges() {
    for (const { edge, mesh } of this.edges) {
      const mat = mesh.material as THREE.MeshBasicMaterial;
      mat.userData.base ??= mat.opacity;
      const on = !this.traced || (this.traced.has(edge.from) && this.traced.has(edge.to));
      mat.opacity = mat.userData.base * (on ? 1 : 0.15);
    }
  }

  /** Fly the camera to a block, or back to the whole model with null. */
  focusOn(id: string | null, run: Run): Promise<void> {
    const sb = id ? this.blocks.get(id) : null;
    if (id && !sb) return Promise.resolve();
    this.sway = false;
    const fromTarget = this.controls.target.clone();
    const fromCam = this.camera.position.clone();
    let toTarget: THREE.Vector3;
    let toCam: THREE.Vector3;
    if (sb) {
      toTarget = sb.obj.root.position.clone();
      // Keep the current viewing direction, closer in.
      const dir = fromCam.clone().sub(fromTarget).normalize();
      toCam = toTarget.clone().addScaledVector(dir, this.fitDistance * 0.45);
    } else {
      const b = this.layout?.bounds;
      const dist = this.fitDistance;
      toTarget = new THREE.Vector3(0, b ? (b.minY + b.maxY) / 2 : 0, 0);
      toCam = new THREE.Vector3(dist * 0.28, toTarget.y + dist * 0.12, dist * 0.95);
    }
    return run.tween(700, (t) => {
      this.controls.target.lerpVectors(fromTarget, toTarget, t);
      this.camera.position.lerpVectors(fromCam, toCam, t);
    });
  }

  flash(id: string, amount = 1) {
    const sb = this.blocks.get(id);
    if (sb) sb.glow = Math.max(sb.glow, amount);
  }

  /** Where a block sits now, for things placed beside it. */
  blockAnchor(id: string): THREE.Vector3 | null {
    return this.blocks.get(id)?.obj.root.position.clone() ?? null;
  }

  /** Which side of a block its label tag sits on. */
  labelSide(id: string): "left" | "right" | "top" | null {
    return this.blocks.get(id)?.labelSide ?? null;
  }

  /** A block's larger side, in world units. */
  blockSize(id: string): number {
    const p = this.blocks.get(id)?.placed;
    return p ? Math.max(p.w, p.h) : 1;
  }

  /** The wire from one block to another, or an arc over the model when no edge joins them. */
  pathBetween(from: string, to: string): THREE.Curve<THREE.Vector3> | null {
    const wire = this.edges.find((e) => e.edge.from === from && e.edge.to === to);
    if (wire) return wire.curve;
    const [a, b] = [this.blockAnchor(from), this.blockAnchor(to)];
    return a && b ? arcBetween(a, b) : null;
  }

  /** What the pointer is over among these objects, nearest first. */
  hits(objects: THREE.Object3D[]): THREE.Intersection[] {
    if (Math.abs(this.pointer.x) > 1 || !objects.length) return [];
    this.raycaster.setFromCamera(this.pointer, this.camera);
    return this.raycaster.intersectObjects(objects, false);
  }

  // ---------- edges and pulses ----------

  private anchor(sb: SceneBlock, where: "top" | "bottom" | "left" | "right"): THREE.Vector3 {
    const p = sb.obj.root.userData.flying ? this.restPosition(sb) : sb.obj.root.position.clone();
    const { w, h } = sb.placed;
    if (where === "top") p.y += h / 2;
    if (where === "bottom") p.y -= h / 2;
    if (where === "left") p.x -= w / 2;
    if (where === "right") p.x += w / 2;
    return p;
  }

  private curveFor(e: Edge): THREE.Curve<THREE.Vector3> | null {
    const a = this.blocks.get(e.from);
    const b = this.blocks.get(e.to);
    if (!a || !b) return null;
    const midX = this.layout ? (this.layout.bounds.minX + this.layout.bounds.maxX) / 2 : 0;
    if (e.kind === "residual") {
      // Arc around the blocks on the side facing away from the labels.
      const side = this.layout && this.layout.columns > 1 && a.placed.x < midX ? 1 : -1;
      const p0 = this.anchor(a, "top").add(new THREE.Vector3(side * (a.placed.w / 2 - 0.25), 0, 0));
      const p3 = this.anchor(b, side > 0 ? "right" : "left");
      const out = Math.max(a.placed.w, b.placed.w) / 2 + 0.55;
      const sx = (side > 0 ? Math.max : Math.min)(p0.x, p3.x) + side * (out - a.placed.w / 2 + 0.25);
      return new THREE.CubicBezierCurve3(p0, new THREE.Vector3(sx, p0.y + 0.35, p0.z), new THREE.Vector3(sx + side * 0.1, p3.y, p3.z), p3);
    }
    if (e.kind === "cross") {
      const p0 = this.anchor(a, "top");
      const side = Math.sign(b.assembled.x - a.assembled.x) || 1;
      const p3 = this.anchor(b, side > 0 ? "left" : "right");
      return new THREE.CubicBezierCurve3(
        p0,
        new THREE.Vector3(p0.x, p0.y + 1.6, p0.z),
        new THREE.Vector3(p3.x - side * 1.6, p3.y, p3.z),
        p3,
      );
    }
    const p0 = this.anchor(a, "top");
    const p3 = this.anchor(b, "bottom");
    const dy = Math.max(0.3, (p3.y - p0.y) * 0.5);
    return new THREE.CubicBezierCurve3(p0, new THREE.Vector3(p0.x, p0.y + dy, p0.z), new THREE.Vector3(p3.x, p3.y - dy, p3.z), p3);
  }

  private buildEdges() {
    disposeChildren(this.edgeGroup);
    this.edges = [];
    if (!this.graph) return;
    for (const edge of this.graph.edges) {
      const curve = this.curveFor(edge);
      if (!curve) continue;
      const geo = new THREE.TubeGeometry(curve, 48, edge.kind === "data" ? 0.035 : 0.028, 6);
      const mat = new THREE.MeshBasicMaterial({
        color: EDGE_COLOR[edge.kind],
        transparent: true,
        opacity: edge.kind === "data" ? 0.8 : 0.9,
        depthWrite: false,
      });
      const mesh = new THREE.Mesh(geo, mat);
      this.edgeGroup.add(mesh);
      this.edges.push({ edge, mesh, curve });
    }
    this.fadeEdges();
    this.setEdgeProgress(this.edgeProgress);
  }

  setEdgeProgress(t: number) {
    this.edgeProgress = t;
    for (const e of this.edges) {
      const count = e.mesh.geometry.index?.count ?? 0;
      // Draw-on: tube indices run along the path.
      e.mesh.geometry.setDrawRange(0, Math.floor((count * t) / 6) * 6);
      e.mesh.visible = t > 0;
    }
  }

  private buildPulses() {
    disposeChildren(this.pulseGroup);
    this.pulses = [];
    if (!this.layout) return;
    let maxRank = 0;
    for (const e of this.edges) {
      const fromRank = this.blocks.get(e.edge.from)!.placed.rank;
      maxRank = Math.max(maxRank, fromRank);
      const color = new THREE.Color(VERMILION);
      const dots = [0, 1, 2].map((i) => {
        const dot = new THREE.Mesh(
          new THREE.SphereGeometry(i === 0 ? 0.16 : 0.24 + i * 0.06, 16, 12),
          // A solid vermilion dot with two soft rings around it.
          new THREE.MeshBasicMaterial({ color, transparent: true, opacity: i === 0 ? 1 : 0.18 / i, depthWrite: false, toneMapped: false }),
        );
        dot.visible = false;
        this.pulseGroup.add(dot);
        return dot;
      });
      // Cross edges fire once the encoder is done, so the pulse visibly hands over.
      this.pulses.push({ edge: e, start: fromRank * PULSE_STEP, dots });
    }
    this.pulseCycle = (maxRank + 2) * PULSE_STEP + PULSE_PAUSE;
  }

  startPulse() {
    this.pulsing = true;
    this.pulseClock = 0;
  }

  // ---------- frame loop ----------

  private tick() {
    const dt = Math.min(this.clock.getDelta(), 0.05);

    if (this.sway && this.layout) {
      this.swayTime += dt;
      const target = this.controls.target;
      const az = Math.sin(this.swayTime * 0.18) * 0.32 + 0.28;
      const r = this.fitDistance;
      this.camera.position.lerp(new THREE.Vector3(target.x + Math.sin(az) * r, target.y + r * 0.12, target.z + Math.cos(az) * r), 0.02);
    }
    this.controls.update();

    // Pulses
    if (this.pulsing) {
      this.pulseClock = (this.pulseClock + dt) % this.pulseCycle;
      for (const p of this.pulses) {
        const t = (this.pulseClock - p.start) / PULSE_STEP;
        p.dots.forEach((dot, i) => {
          const ti = t; // dot and rings travel together
          dot.visible = ti > 0 && ti < 1;
          if (dot.visible) p.edge.curve.getPointAt(Math.min(1, ti), dot.position);
        });
        if (t >= 1 && t - dt / PULSE_STEP < 1) this.flash(p.edge.edge.to, 0.8);
        if (t > 0 && t - dt / PULSE_STEP <= 0) this.flash(p.edge.edge.from, 0.5);
      }
    } else {
      for (const p of this.pulses) for (const d of p.dots) d.visible = false;
    }

    // Hover + glow
    let hit: string | null = null;
    if (this.blocks.size && Math.abs(this.pointer.x) <= 1) {
      this.raycaster.setFromCamera(this.pointer, this.camera);
      hit = blockUnder(this.raycaster.intersectObjects([...this.model.children, ...this.occluders()], true));
    }
    if (hit !== this.hovered) {
      this.hovered = hit;
      this.labels.domElement.style.cursor = hit ? "pointer" : "";
      this.onHover(hit);
    }
    for (const sb of this.blocks.values()) {
      sb.glow = Math.max(0, sb.glow - dt * 1.6);
      const traced = this.traced?.has(sb.block.id) ?? false;
      const lit = sb.block.id === this.hovered || sb.block.id === this.highlighted || traced;
      const faded = this.traced !== null && !traced;
      // Highlighter halo instead of glow: on while hovered/highlighted, flashes fade out.
      const halo = Math.min(1, Math.max(lit ? 0.85 : 0, sb.glow));
      sb.obj.haloMaterial.opacity = halo * 0.9;
      sb.obj.halo.visible = halo > 0.01;
      (sb.label.element as HTMLElement).style.opacity = String(this.labelAlpha * (sb.obj.root.visible ? 1 : 0) * (faded ? 0.3 : 1));
      sb.label.element.classList.toggle("hot", lit);
    }
    for (const g of this.groupFrames) {
      const a = this.groupAlpha * (1 - this.explode);
      g.frame.visible = a > 0.01;
      g.frame.traverse((o) => {
        const mat = (o as THREE.Mesh).material as THREE.Material & { userData: { base?: number } };
        if (!mat || !("opacity" in mat)) return;
        mat.userData.base ??= mat.opacity;
        mat.opacity = mat.userData.base * a;
      });
      (g.label.element as HTMLElement).style.opacity = String(a);
    }

    for (const hook of this.frameHooks) hook(dt);
    this.composer.render();
    this.labels.render(this.scene, this.camera);
  }
}

function disposeChildren(group: THREE.Group) {
  for (const child of group.children as THREE.Mesh[]) {
    child.geometry?.dispose();
    (child.material as THREE.Material | undefined)?.dispose();
  }
  group.clear();
}

/**
 * The block under the pointer: the nearest visible block, unless an occluder (the specimen card)
 * is in front of it. Group frames and other see-through things don't count.
 */
export function blockUnder(hits: THREE.Intersection[]): string | null {
  const first = hits.find((h) => h.object.visible && (h.object.userData.blockId || h.object.userData.occludes));
  return (first?.object.userData.blockId as string | undefined) ?? null;
}

export function escapeHtml(s: string): string {
  return s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]!);
}
