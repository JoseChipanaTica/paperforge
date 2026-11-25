import { expect, test } from "bun:test";
import { CameraRig } from "./cameraRig";
import type { Shot } from "./director";

const shot = (x: number, ms = 900): Shot => ({ name: "settle", target: { x, y: 0, z: 0 }, offset: { x: 0, y: 0, z: 5 }, ms });
const rigAtOrigin = () => {
  const target = { x: 0, y: 0, z: 0 };
  const position = { x: 0, y: 0, z: 5 };
  return { target, position, rig: new CameraRig(target, position) };
};
const run = (rig: CameraRig, seconds: number, each?: () => void) => {
  for (let t = 0; t < seconds; t += 1 / 60) {
    rig.step(1 / 60);
    each?.();
  }
};

test("converges without overshoot", () => {
  const { target, position, rig } = rigAtOrigin();
  rig.cue(shot(10));
  let most = 0;
  run(rig, 2, () => (most = Math.max(most, target.x)));
  expect(target.x).toBeCloseTo(10, 1);
  expect(Math.abs(target.x - 10)).toBeLessThan(0.01);
  expect(position.x).toBeCloseTo(10, 1);
  expect(position.z).toBeCloseTo(5, 1);
  expect(most).toBeLessThanOrEqual(10.01);
});

test("follow tracks a moving point", () => {
  const { target, rig } = rigAtOrigin();
  const point = { x: 0, y: 0, z: 0 };
  rig.follow(() => point, shot(0, 500));
  run(rig, 3, () => (point.x += 1 / 60));
  expect(Math.abs(target.x - point.x)).toBeLessThan(0.6);
});

test("releaseWhenSettled lets go", () => {
  const { rig } = rigAtOrigin();
  rig.cue(shot(4), true);
  expect(rig.active).toBe(true);
  run(rig, 3);
  expect(rig.active).toBe(false);
});

test("release stops all motion", () => {
  const { target, position, rig } = rigAtOrigin();
  rig.cue(shot(10));
  rig.step(1 / 60);
  rig.release();
  const t = { ...target }, p = { ...position };
  run(rig, 1);
  expect(target).toEqual(t);
  expect(position).toEqual(p);
  expect(rig.active).toBe(false);
});

test("a cue after release takes over again from the current position", () => {
  const { target, rig } = rigAtOrigin();
  rig.cue(shot(10));
  run(rig, 0.3);
  rig.release();
  target.x = 3; // the visitor dragged
  rig.cue(shot(3));
  rig.step(1 / 60);
  expect(Math.abs(target.x - 3)).toBeLessThan(0.05);
});

test("jump lands at once", () => {
  const { target, position, rig } = rigAtOrigin();
  rig.jump(shot(7));
  expect(target.x).toBe(7);
  expect(position).toEqual({ x: 7, y: 0, z: 5 });
  expect(rig.active).toBe(true);
});

test("after the visitor takes over, cues are ignored until the rig is reclaimed", () => {
  const { target, rig } = rigAtOrigin();
  rig.cue(shot(10));
  rig.handOver();
  rig.cue(shot(10));
  rig.follow(() => ({ x: 10, y: 0, z: 0 }), shot(10));
  rig.jump(shot(10));
  run(rig, 1);
  expect(rig.active).toBe(false);
  expect(target.x).toBe(0);
  expect(rig.handedOver).toBe(true);
  rig.reclaim();
  expect(rig.handedOver).toBe(false);
  rig.jump(shot(10));
  expect(target.x).toBe(10);
});
