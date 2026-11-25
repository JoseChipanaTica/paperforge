import type { BlockType } from "../shared/graph";
import type { Layout } from "./layout";

/**
 * The director's cut: where the camera looks during assembly. Pure (no Three.js) so the shots can be tested; the
 * camera rig springs between them. Camera position = target + offset, in world units.
 */

export interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export type ShotName = "discover" | "hero" | "reveal" | "settle" | "follow";

export interface Shot {
  name: ShotName;
  target: Vec3;
  offset: Vec3;
  ms: number;
}

/** The block the camera dollies in on: the paper's idea, as far as its type says. */
export const HERO_PRIORITY: readonly BlockType[] = [
  "multi_head_attention",
  "cross_attention",
  "attention",
  "residual_add",
  "recurrent",
  "convolution",
];

const MIN_DISTANCE = 8;

const vec = (x: number, y: number, z: number): Vec3 => ({ x, y, z });
const scaled = (v: Vec3, s: number) => vec(v.x * s, v.y * s, v.z * s);
const unit = (x: number, y: number, z: number) => scaled(vec(x, y, z), 1 / Math.hypot(x, y, z));

export function pickHero(blocks: readonly { id: string; type: BlockType }[], layout: Layout): string | null {
  for (const type of HERO_PRIORITY) {
    const hit = blocks.find((b) => b.type === type);
    if (hit) return hit.id;
  }
  let best: string | null = null;
  let most = -1;
  for (const b of blocks) {
    const p = layout.blocks.get(b.id);
    const volume = p ? p.w * p.h * p.d : 0;
    if (volume > most) [best, most] = [b.id, volume];
  }
  return best;
}

/** Camera distance that fits these points, with room for side labels and the top bar (as `scene.distanceFor(0)`). */
export function framePoints(points: readonly Vec3[], aspect: number, fovDeg: number): number {
  if (!points.length) return MIN_DISTANCE;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const w = Math.max(...xs) - Math.min(...xs) + 11;
  const h = (Math.max(...ys) - Math.min(...ys)) * 1.3 + 4;
  const tan = Math.tan((fovDeg * Math.PI) / 180 / 2);
  return Math.max(h / 2 / tan, w / 2 / (tan * aspect));
}

const middle = (layout: Layout) => vec(0, (layout.bounds.minY + layout.bounds.maxY) / 2, 0);

/** The angle `scene.frame(true)` has always used. */
export function restingShot(layout: Layout, dist: number): Shot {
  return { name: "settle", target: middle(layout), offset: vec(dist * 0.28, dist * 0.12, dist * 0.95), ms: 1600 };
}

/** A wide three-quarter view of the whole model as it clicks together. */
export function revealShot(layout: Layout, dist: number): Shot {
  return { name: "reveal", target: middle(layout), offset: scaled(unit(0.62, 0.2, 0.78), dist * 1.1), ms: 1400 };
}

export function heroShot(at: Vec3, dist: number): Shot {
  return { name: "hero", target: { ...at }, offset: scaled(unit(0.3, 0.08, 1), Math.max(6, dist * 0.38)), ms: 900 };
}

/** Low and close on what has landed so far, orbiting a little further with each block. */
export function discoverShot(points: readonly Vec3[], landed: number, aspect: number, fovDeg: number): Shot {
  let target = vec(0, 0, 0);
  if (points.length) {
    const xs = points.map((p) => p.x), ys = points.map((p) => p.y), zs = points.map((p) => p.z);
    const mid = (a: number[]) => (Math.min(...a) + Math.max(...a)) / 2;
    target = vec(mid(xs), mid(ys), mid(zs));
  }
  const dist = Math.max(MIN_DISTANCE, framePoints(points, aspect, fovDeg) * 1.05);
  const az = -0.35 + 0.03 * Math.min(landed, 12);
  return { name: "discover", target, offset: scaled(vec(Math.sin(az), -0.04, Math.cos(az)), dist), ms: 1100 };
}

const FOLLOW_SCALE = 0.75; // of the fit distance: close enough to feel the pulse, wide enough to keep both columns

/** Where the follow shot looks: the pulse's height, but only part of its sideways jumps between columns. */
export function followPoint(head: Vec3): Vec3 {
  return vec(head.x * 0.4, head.y, 0);
}

/** Rides along with the pulse; the target is replaced every frame by the pulse head. */
export function followShot(dist: number): Shot {
  const s = dist * FOLLOW_SCALE;
  return { name: "follow", target: vec(0, 0, 0), offset: vec(0.28 * s, 0.12 * s, 0.95 * s), ms: 500 };
}

/** Index of the pulse furthest along the model (the latest to start) that is in flight, or -1. */
export function leadingPulse(starts: readonly number[], clock: number, step: number): number {
  let lead = -1;
  starts.forEach((start, i) => {
    const t = (clock - start) / step;
    if (t > 0 && t < 1 && (lead < 0 || start > starts[lead]!)) lead = i;
  });
  return lead;
}
