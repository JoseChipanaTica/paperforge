import * as THREE from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import type { BlockType, LocatedBlock } from "../shared/graph";
import type { PlacedBlock } from "./layout";

/** Design tokens shared with styles/tokens.css. */
export const INK = "#1e1d1a";
export const HIGHLIGHTER = "#f6d77a";
export const VERMILION = "#c2410c";
export const SHEET = "#fffdf7";

/** The printed palette: one earthy ink per block type. */
export const TYPE_COLOR: Record<BlockType, string> = {
  input: "#cfc5b0",
  output: "#cfc5b0",
  embedding: "#e8dcc4",
  positional_encoding: "#7c8f6a",
  multi_head_attention: "#c2410c",
  attention: "#c2410c",
  cross_attention: "#d9733b",
  feed_forward: "#1e3a5f",
  add_norm: "#e0b04c",
  layer_norm: "#e0b04c",
  normalization: "#e0b04c",
  residual_add: "#9c8f78",
  linear: "#4b5d73",
  activation: "#a3533c",
  softmax: "#b8860b",
  convolution: "#2f6f6a",
  pooling: "#8a6f4e",
  recurrent: "#6d4c6f",
  dropout: "#b9ae98",
  concat: "#9c8f78",
  other: "#8f8a80",
};

export interface BlockObject {
  root: THREE.Group;
  materials: THREE.MeshStandardMaterial[];
  color: THREE.Color;
  /** Ink shells just behind each mesh: the printed outline. */
  outline: THREE.Group;
  /** A flat ink copy offset down-right: the hard print shadow. */
  shadow: THREE.Group;
  /** Highlighter-yellow shells, faded in on hover / highlight. */
  halo: THREE.Group;
  haloMaterial: THREE.MeshBasicMaterial;
}

// Matte paint: no glow, no shine. (The renderer's NeutralToneMapping keeps hues close to the palette.)
function material(color: THREE.Color, opts: Partial<THREE.MeshStandardMaterialParameters> = {}) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.9, metalness: 0, ...opts });
}

const inkMaterial = new THREE.MeshBasicMaterial({ color: INK, side: THREE.BackSide });
const shadowMaterial = new THREE.MeshBasicMaterial({ color: INK });
const SHADOW_OFFSET = new THREE.Vector3(0.09, -0.09, -0.09);
const OUTLINE_PAD = 0.035; // world units: an even ink line on every piece, thin or wide
const HALO_PAD = 0.16;

/** Per-axis scale that grows a geometry by a fixed padding on every side. */
function padScale(geometry: THREE.BufferGeometry, pad: number): THREE.Vector3 {
  geometry.computeBoundingBox();
  const size = geometry.boundingBox!.getSize(new THREE.Vector3());
  return new THREE.Vector3(
    (size.x + 2 * pad) / Math.max(size.x, 1e-3),
    (size.y + 2 * pad) / Math.max(size.y, 1e-3),
    (size.z + 2 * pad) / Math.max(size.z, 1e-3),
  );
}
const noRaycast = () => {};

/** Wrap every mesh of a block in an ink outline, an offset ink shadow and a hidden halo. */
function printDecorations(root: THREE.Group) {
  const outline = new THREE.Group();
  const shadow = new THREE.Group();
  const halo = new THREE.Group();
  const haloMaterial = new THREE.MeshBasicMaterial({
    color: HIGHLIGHTER,
    side: THREE.BackSide,
    transparent: true,
    opacity: 0,
    depthWrite: false,
  });
  for (const child of [...root.children] as THREE.Mesh[]) {
    const shell = (mat: THREE.Material, pad: number, offset?: THREE.Vector3) => {
      const m = new THREE.Mesh(child.geometry, mat);
      m.position.copy(child.position);
      if (offset) m.position.add(offset);
      m.rotation.copy(child.rotation);
      m.scale.copy(child.scale).multiply(padScale(child.geometry, pad));
      m.raycast = noRaycast;
      return m;
    };
    outline.add(shell(inkMaterial, OUTLINE_PAD));
    shadow.add(shell(shadowMaterial, 0, SHADOW_OFFSET));
    halo.add(shell(haloMaterial, HALO_PAD));
  }
  halo.visible = false;
  root.add(shadow, outline, halo);
  return { outline, shadow, halo, haloMaterial };
}

const box = (w: number, h: number, d: number, r = 0.08) =>
  new RoundedBoxGeometry(w, h, d, 3, Math.min(r, w / 2.2, h / 2.2, d / 2.2));

function hyper(block: LocatedBlock, name: string): number | null {
  const hp = block.hyperparameters.find((h) => h.name.trim().toLowerCase() === name);
  const n = hp ? parseInt(hp.value, 10) : NaN;
  return Number.isFinite(n) ? n : null;
}

/** Builds the block's silhouette centered on the origin, sized to its layout box. */
export function buildBlock(block: LocatedBlock, placed: PlacedBlock): BlockObject {
  const color = new THREE.Color(TYPE_COLOR[block.type] ?? TYPE_COLOR.other);
  const root = new THREE.Group();
  const materials: THREE.MeshStandardMaterial[] = [];
  const add = (geo: THREE.BufferGeometry, mat: THREE.MeshStandardMaterial, x = 0, y = 0, z = 0) => {
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, y, z);
    mesh.castShadow = true;
    mesh.userData.blockId = block.id;
    root.add(mesh);
    if (!materials.includes(mat)) materials.push(mat);
    return mesh;
  };
  const { w, h, d } = placed;
  const main = material(color);

  switch (block.type) {
    case "input":
    case "output": {
      // A row of token cubes.
      const n = 6;
      const s = Math.min(h, (w - 0.15 * (n - 1)) / n);
      for (let i = 0; i < n; i++) add(box(s, s, s, 0.06), main, -w / 2 + s / 2 + i * (s + (w - n * s) / (n - 1)), 0, 0);
      break;
    }
    case "embedding": {
      // A lookup table: a slab of vector columns.
      const n = 12;
      const bw = w / n - 0.04;
      for (let i = 0; i < n; i++) {
        const shade = color.clone().offsetHSL(0, 0, (i % 3) * 0.04 - 0.04);
        add(box(bw, h, d, 0.03), material(shade), -w / 2 + bw / 2 + i * (w / n), 0, 0);
      }
      break;
    }
    case "positional_encoding": {
      add(box(w, h * 0.35, d, 0.05), material(color, { transparent: true, opacity: 0.55 }), 0, -h * 0.3, 0);
      // Sine + cosine ribbons.
      for (const [phase, z] of [[0, 0.18], [Math.PI / 2, -0.18]] as const) {
        const pts = Array.from({ length: 64 }, (_, i) => {
          const t = i / 63;
          return new THREE.Vector3(-w / 2 + t * w, Math.sin(t * Math.PI * 4 + phase) * h * 0.32 + h * 0.08, z);
        });
        add(new THREE.TubeGeometry(new THREE.CatmullRomCurve3(pts), 96, 0.045, 8), main);
      }
      break;
    }
    case "multi_head_attention":
    case "attention":
    case "cross_attention": {
      // Parallel heads, then the output projection W^O on top.
      const heads = block.type === "attention" ? 1 : Math.max(2, Math.min(16, hyper(block, "h") ?? 8));
      const paneH = h * 0.72;
      const gap = w / heads;
      for (let i = 0; i < heads; i++) {
        const shade = color.clone().offsetHSL((i / heads) * 0.06 - 0.03, 0, 0);
        add(box(gap * 0.72, paneH, d * 0.85, 0.05), material(shade, { transparent: true, opacity: 0.9 }), -w / 2 + gap / 2 + i * gap, -h / 2 + paneH / 2, 0);
      }
      add(box(w, h * 0.18, d, 0.05), main, 0, h / 2 - h * 0.09, 0);
      break;
    }
    case "feed_forward": {
      // d_model → d_ff → d_model: narrow, wide, narrow.
      const part = h / 3;
      add(box(w * 0.62, part * 0.86, d * 0.8), main, 0, -part, 0);
      add(box(w, part * 0.96, d), material(color.clone().offsetHSL(0, 0, 0.06)), 0, 0, 0);
      add(box(w * 0.62, part * 0.86, d * 0.8), main, 0, part, 0);
      break;
    }
    case "add_norm":
    case "layer_norm":
    case "normalization":
    case "residual_add": {
      if (block.type === "residual_add") {
        add(new THREE.TorusGeometry(w * 0.35, 0.07, 12, 40), main);
      } else {
        add(box(w, h, d, h / 2.2), main);
      }
      break;
    }
    case "softmax": {
      add(box(w, h * 0.25, d, 0.05), main, 0, -h / 2 + h * 0.125, 0);
      const bars = [0.15, 0.3, 0.95, 0.4, 0.2, 0.1, 0.55, 0.12];
      const bw = w / bars.length;
      bars.forEach((v, i) => add(box(bw * 0.6, h * 0.7 * v, d * 0.5, 0.03), main, -w / 2 + bw / 2 + i * bw, -h / 4 + (h * 0.7 * v) / 2, 0));
      break;
    }
    case "convolution": {
      const k = 3;
      const s = Math.min(w, h * 1.4) / (k + 1);
      for (let i = 0; i < k; i++)
        for (let j = 0; j < k; j++) add(box(s * 0.9, h * 0.8, s * 0.9, 0.04), main, (i - 1) * s, 0, (j - 1) * s * 0.9);
      break;
    }
    case "pooling":
      add(new THREE.CylinderGeometry(w * 0.18, w * 0.42, h, 4, 1), main).rotation.y = Math.PI / 4;
      break;
    case "recurrent":
      add(new THREE.TorusGeometry(Math.min(w, h) * 0.38, 0.12, 16, 48), main).rotation.x = Math.PI / 2;
      add(box(w, h * 0.2, d, 0.05), material(color, { transparent: true, opacity: 0.6 }));
      break;
    default:
      add(box(w, h, d), main);
  }
  return { root, materials, color, ...printDecorations(root) };
}

/** Translucent frame around a repeated group, with ghost copies stacked behind it (×N). */
export function buildGroupFrame(w: number, h: number, d: number, repeat: number): THREE.Group {
  const g = new THREE.Group();
  const shell = new THREE.Mesh(
    box(w, h, d, 0.25),
    new THREE.MeshBasicMaterial({ color: INK, transparent: true, opacity: 0.025, depthWrite: false }),
  );
  g.add(shell);
  const edgesMat = new THREE.LineBasicMaterial({ color: INK, transparent: true, opacity: 0.45 });
  g.add(new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d)), edgesMat));
  const ghosts = Math.min(repeat - 1, 3);
  for (let i = 1; i <= ghosts; i++) {
    const ghost = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(w, h, d)),
      new THREE.LineBasicMaterial({ color: INK, transparent: true, opacity: 0.22 * (1 - i / (ghosts + 1)) }),
    );
    ghost.position.set(i * 0.12, i * 0.12, -i * 0.5);
    ghost.userData.ghost = i;
    g.add(ghost);
  }
  return g;
}
