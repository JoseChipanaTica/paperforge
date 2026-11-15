import { describe, expect, test } from "bun:test";
import * as THREE from "three";
import { arcBetween, specimenOffset } from "./specimen";

describe("specimenOffset", () => {
  const block = new THREE.Vector3(0, 0, 0);
  const horizontal = (p: THREE.Vector3) => Math.hypot(p.x - block.x, p.z - block.z);

  test("sits beside the block, never over it, from any viewing side", () => {
    for (const camera of [new THREE.Vector3(0, 0, 10), new THREE.Vector3(10, 2, 0), new THREE.Vector3(-6, 3, -6)]) {
      // The card's near edge clears the block's half-width.
      const p = specimenOffset(camera, block, 2, 3);
      expect(horizontal(p) - 3 / 2).toBeGreaterThanOrEqual(1);
      expect(p.y).toBeGreaterThan(0);
    }
  });

  test("side -1 puts it on the camera's left instead", () => {
    const camera = new THREE.Vector3(0, 0, 10);
    expect(specimenOffset(camera, block, 2, 0, 1).x).toBeGreaterThan(0);
    expect(specimenOffset(camera, block, 2, 0, -1).x).toBeLessThan(0);
  });

  test("a wider specimen moves further out", () => {
    const camera = new THREE.Vector3(0, 0, 10);
    expect(horizontal(specimenOffset(camera, block, 2, 4))).toBeGreaterThan(horizontal(specimenOffset(camera, block, 2, 0)));
  });
});

describe("arcBetween", () => {
  test("starts and ends at the two blocks and arcs above both", () => {
    const a = new THREE.Vector3(-3, 1, 0);
    const b = new THREE.Vector3(4, 6, 0);
    const curve = arcBetween(a, b);
    expect(curve.getPoint(0).distanceTo(a)).toBeLessThan(1e-9);
    expect(curve.getPoint(1).distanceTo(b)).toBeLessThan(1e-9);
    expect(curve.getPoint(0.5).y).toBeGreaterThan(Math.max(a.y, b.y));
  });
});
