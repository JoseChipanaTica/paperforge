import * as THREE from "three";
import type { LocatedArchitecture } from "../shared/graph";
import { Run, ease } from "./anim";
import type { PaperView } from "./paper";
import type { ForgeScene, SceneBlock } from "./scene";

const PAPER_SHIFT = 0.17; // model sits right of the paper card while it's open

/**
 * The demo moment: highlights sweep the paper, each block flies out of its sentence into
 * an exploded view, the paper slides away, blocks click together, edges draw on, and a
 * data pulse runs through the model.
 */
export async function assemble(scene: ForgeScene, paper: PaperView, graph: LocatedArchitecture, run: Run) {
  scene.explode = 1;
  scene.pulsing = false;
  scene.edgeProgress = 0;
  scene.groupAlpha = 0;
  scene.labelAlpha = 1;
  scene.load(graph);
  scene.setEdgeProgress(0);
  scene.setViewShift(PAPER_SHIFT);
  paper.setBlocks(graph.blocks);
  paper.show(true);

  const blocks = [...scene.blocks.values()];
  // Highlights sweep in reading order; the snap-together below goes bottom-up.
  const reading = [...blocks].sort((a, b) => {
    const pa = a.block.source, pb = b.block.source;
    if (!pa || !pb) return (pa ? 0 : 1) - (pb ? 0 : 1) || a.index - b.index;
    return pa.page - pb.page || pa.start - pb.start || a.index - b.index;
  });
  const flights: Promise<void>[] = [];
  for (const sb of reading) {
    if (run.cancelled) return;
    const src = sb.block.source;
    let from = paper.center();
    if (src) {
      await paper.flipTo(src.page, run);
      from = (await paper.sweep(sb.block.id, run)) ?? from;
    }
    flights.push(fly(scene, sb, scene.screenToWorld(from.x, from.y), run));
    await run.wait(40);
  }
  await Promise.all(flights);
  if (run.cancelled) return;
  await run.wait(500);

  // Paper leaves, model slides to center, everything clicks together bottom-up.
  paper.show(false);
  const fitFrom = scene.fitDistance;
  const fitTo = scene.distanceFor(0);
  void run.tween(1100, (t) => {
    scene.setViewShift(PAPER_SHIFT * (1 - t));
    scene.fitDistance = fitFrom + (fitTo - fitFrom) * t;
  });
  const byRank = [...blocks].sort((a, b) => a.placed.rank - b.placed.rank || a.index - b.index);
  const snaps = byRank.map(async (sb, i) => {
    await run.wait(i * 45);
    const from = sb.obj.root.position.clone();
    await run.tween(420, (t) => sb.obj.root.position.lerpVectors(from, sb.assembled, t), ease.outBack);
    scene.flash(sb.block.id, 1);
  });
  await Promise.all(snaps);
  if (run.cancelled) return;
  scene.setExplode(0);

  await run.tween(800, (t) => scene.setEdgeProgress(t), ease.outCubic);
  await run.tween(600, (t) => (scene.groupAlpha = t));
  if (!run.cancelled) scene.startPulse();
}

async function fly(scene: ForgeScene, sb: SceneBlock, start: THREE.Vector3, run: Run) {
  const root = sb.obj.root;
  const end = scene.restPosition(sb, 1);
  const control = start.clone().lerp(end, 0.5).add(new THREE.Vector3(0, 2.2, 2.5));
  const curve = new THREE.QuadraticBezierCurve3(start, control, end);
  root.userData.flying = true;
  root.visible = true;
  const spin = (sb.index % 2 ? 1 : -1) * 1.4;
  await run.tween(
    950,
    (t) => {
      curve.getPoint(t, root.position);
      root.scale.setScalar(0.12 + 0.88 * ease.outBack(t));
      root.rotation.set(spin * 0.4 * (1 - t), spin * (1 - t), 0);
    },
    ease.outCubic,
  );
  root.userData.flying = false;
  scene.flash(sb.block.id, 0.7);
}
