import { expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture, type LocatedArchitecture } from "../shared/graph";
import { Run } from "./anim";
import { CameraRig } from "./cameraRig";
import { layoutGraph } from "./layout";
import type { ForgeScene } from "./scene";
import { director } from "./timeline";

// Run's tweens tick on animation frames.
globalThis.requestAnimationFrame ??= (cb: FrameRequestCallback) => setTimeout(() => cb(performance.now()), 16) as unknown as number;

const graph = Architecture.parse(
  await Bun.file(join(import.meta.dir, "../../samples/1706.03762.json")).json(),
) as unknown as LocatedArchitecture;

const fakeScene = () => {
  const target = { x: 0, y: 0, z: 0 };
  const position = { x: 0, y: 0, z: 30 };
  return {
    layout: layoutGraph(graph),
    rig: new CameraRig(target, position),
    camera: { aspect: 1.6, fov: 32 },
    distanceFor: () => 40,
    pulseHead: () => ({ x: 1, y: 2, z: 0 }),
    pulseCycle: 0.05,
    reduceMotion: false,
    sway: false,
  } as unknown as ForgeScene;
};

test("after the follow, a camera the app has since moved is not taken back", async () => {
  const scene = fakeScene();
  director(scene, graph, new Run()).pulse();
  expect(scene.rig.active).toBe(true);
  scene.rig.release(); // focusOn: the walkthrough flies to a block
  await new Promise((r) => setTimeout(r, 200)); // the follow's pulse cycle has passed
  expect(scene.rig.active).toBe(false);
  expect(scene.sway).toBe(false);
});

test("an undisturbed follow eases back to the resting shot", async () => {
  const scene = fakeScene();
  director(scene, graph, new Run()).pulse();
  await new Promise((r) => setTimeout(r, 200));
  expect(scene.rig.active).toBe(true); // springing to rest, then lets go when settled
});
