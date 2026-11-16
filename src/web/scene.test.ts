import { describe, expect, test } from "bun:test";
import * as THREE from "three";
import type { LocatedArchitecture, LocatedBlock } from "../shared/graph";
import { VERMILION, buildBlock } from "./blocks";
import { ForgeScene, blockUnder } from "./scene";

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

const graph = (await Bun.file(new URL("../../samples/1706.03762.json", import.meta.url)).json()) as LocatedArchitecture;

describe("setHeat", () => {
  const sceneBlock = (block: LocatedBlock) => {
    const obj = buildBlock(block, { id: block.id, column: 0, rank: 0, x: 0, y: 0, w: 2, h: 1, d: 1 });
    return { block, obj, heat: 0, heatTarget: 0 };
  };
  const [a, b] = graph.blocks;
  const fake = (reduceMotion: boolean) => ({ blocks: new Map([a!, b!].map((x) => [x.id, sceneBlock(x)])), reduceMotion });
  const setHeat = (s: ReturnType<typeof fake>, levels: ReadonlyMap<string, number> | null) =>
    ForgeScene.prototype.setHeat.call(s as unknown as ForgeScene, levels);

  test("setHeat sets and clears per-block heat", () => {
    const s = fake(true);
    setHeat(s, new Map([[a!.id, 1], [b!.id, 0.25]]));
    const [ha, hb] = [s.blocks.get(a!.id)!.obj, s.blocks.get(b!.id)!.obj];
    expect(ha.heat.visible).toBe(true);
    expect(hb.heat.visible).toBe(true);
    expect(ha.heatMaterial.opacity).toBeGreaterThan(hb.heatMaterial.opacity);
    expect(`#${ha.heatMaterial.color.getHexString()}`).toBe(VERMILION);
    setHeat(s, new Map([[a!.id, 1]]));
    expect(hb.heat.visible).toBe(false);
    setHeat(s, null);
    expect(ha.heat.visible).toBe(false);
    expect(ha.heatMaterial.opacity).toBe(0);
  });

  test("with motion, heat eases in from the frame loop instead of appearing at once", () => {
    const s = fake(false);
    setHeat(s, new Map([[a!.id, 1]]));
    const sb = s.blocks.get(a!.id)!;
    expect(sb.heatTarget).toBe(1);
    expect(sb.obj.heat.visible).toBe(false);
  });
});
