import { describe, expect, test } from "bun:test";
import * as THREE from "three";
import { blockUnder } from "./scene";

const hit = (userData: Record<string, unknown>, visible = true) => {
  const object = new THREE.Object3D();
  object.userData = userData;
  object.visible = visible;
  return { object, distance: 1, point: new THREE.Vector3() } as THREE.Intersection;
};

describe("blockUnder", () => {
  test("the nearest visible block wins", () => {
    expect(blockUnder([hit({ blockId: "a" }, false), hit({ blockId: "b" }), hit({ blockId: "c" })])).toBe("b");
  });

  test("the specimen card in front of a block takes the pointer", () => {
    expect(blockUnder([hit({ occludes: true }), hit({ blockId: "b" })])).toBeNull();
  });

  test("a group frame around the block doesn't", () => {
    expect(blockUnder([hit({}), hit({ blockId: "b" })])).toBe("b");
  });
});
