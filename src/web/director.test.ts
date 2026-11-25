import { expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "../shared/graph";
import {
  discoverShot,
  followShot,
  framePoints,
  heroShot,
  leadingPulse,
  pickHero,
  restingShot,
  revealShot,
  type Shot,
  type Vec3,
} from "./director";
import { layoutGraph } from "./layout";

const sample = async (id: string) =>
  Architecture.parse(await Bun.file(join(import.meta.dir, `../../samples/${id}.json`)).json());
const transformer = await sample("1706.03762");
const resnet = await sample("1512.03385");

const len = (v: Vec3) => Math.hypot(v.x, v.y, v.z);
const centres = (layout: ReturnType<typeof layoutGraph>): Vec3[] =>
  [...layout.blocks.values()].map((b) => ({ x: b.x, y: b.y, z: 0 }));

test("pickHero picks the Transformer's attention", () => {
  expect(pickHero(transformer.blocks, layoutGraph(transformer))).toBe("encoder_self_attention");
});

test("pickHero picks ResNet's residual add", () => {
  expect(pickHero(resnet.blocks, layoutGraph(resnet))).toBe("c2_add");
});

test("pickHero falls back to the largest box", () => {
  const graph = {
    blocks: [
      { id: "a", type: "input" as const, group: "" },
      { id: "b", type: "linear" as const, group: "" },
      { id: "c", type: "softmax" as const, group: "" },
    ],
    edges: [
      { from: "a", to: "b", kind: "data" as const },
      { from: "b", to: "c", kind: "data" as const },
    ],
    groups: [],
  };
  expect(pickHero(graph.blocks, layoutGraph(graph as never))).toBe("c");
  expect(pickHero([], layoutGraph({ blocks: [], edges: [], groups: [] }))).toBeNull();
});

test("restingShot matches frame()", () => {
  const layout = layoutGraph(transformer);
  const s = restingShot(layout, 40);
  expect(s.target).toEqual({ x: 0, y: (layout.bounds.minY + layout.bounds.maxY) / 2, z: 0 });
  expect(s.offset.x).toBeCloseTo(11.2);
  expect(s.offset.y).toBeCloseTo(4.8);
  expect(s.offset.z).toBeCloseTo(38);
});

test("revealShot fits the whole model", () => {
  const layout = layoutGraph(transformer);
  const d = framePoints(centres(layout), 1.6, 32);
  expect(len(revealShot(layout, d).offset)).toBeGreaterThanOrEqual(d);
  expect(revealShot(layout, d).target).toEqual(restingShot(layout, d).target);
});

test("heroShot targets the hero", () => {
  const at = { x: -3, y: 2, z: 0.5 };
  const s = heroShot(at, 10);
  expect(s.target).toEqual(at);
  expect(len(s.offset)).toBeGreaterThanOrEqual(6);
});

test("discoverShot frames the landed points", () => {
  const points = [
    { x: -4, y: 0, z: 0 },
    { x: 2, y: 6, z: 0 },
    { x: 0, y: 3, z: 0 },
  ];
  const s = discoverShot(points, 3, 1.6, 32);
  expect(s.target).toEqual({ x: -1, y: 3, z: 0 });
  expect(len(s.offset)).toBeGreaterThanOrEqual(framePoints(points, 1.6, 32));
  expect(s.offset.y).toBeLessThan(0);
});

test("leadingPulse picks the latest-started pulse in flight", () => {
  expect(leadingPulse([0, 0.55, 1.1], 0.8, 0.55)).toBe(1);
  expect(leadingPulse([0, 0.55, 1.1], 5, 0.55)).toBe(-1);
});

test("degenerate graphs give finite shots", () => {
  const one = layoutGraph({ blocks: [{ id: "x", type: "linear", group: "" }], edges: [], groups: [] });
  const d = framePoints(centres(one), 1.6, 32);
  const shots: Shot[] = [
    restingShot(one, d),
    revealShot(one, d),
    heroShot({ x: 0, y: 0, z: 0 }, d),
    discoverShot([], 0, 1.6, 32),
    discoverShot(centres(one), 1, 1.6, 32),
    followShot(d),
  ];
  for (const s of shots) {
    for (const v of [s.target, s.offset]) for (const n of [v.x, v.y, v.z]) expect(Number.isFinite(n)).toBe(true);
    expect(s.ms).toBeGreaterThan(0);
  }
});
