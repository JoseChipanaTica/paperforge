import * as THREE from "three";
import type { LocatedArchitecture } from "../shared/graph";
import { Run, ease } from "./anim";
import { discoverShot, followPoint, followShot, heroShot, pickHero, restingShot, revealShot, type Vec3 } from "./director";
import type { PaperView } from "./paper";
import type { ForgeScene, SceneBlock } from "./scene";

const PAPER_SHIFT = 0; // the paper lives in the study panel now, beside the stage

/**
 * The demo moment: highlights sweep the paper, each block flies out of its sentence into
 * an exploded view, the paper slides away, blocks click together, edges draw on, and a
 * data pulse runs through the model. `direct`: the camera follows the director's cut (discover the blocks as they
 * land, dolly in on the hero, pull back for the reveal, settle, ride the pulse).
 */
export async function assemble(scene: ForgeScene, paper: PaperView, graph: LocatedArchitecture, run: Run, direct = false) {
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
  const cam = direct ? director(scene, graph, run) : null;
  if (reading[0]) cam?.start(scene.restPosition(reading[0], 1));
  const flights: Promise<void>[] = [];
  for (const sb of reading) {
    if (run.cancelled) return;
    const src = sb.block.source;
    let from = paper.center();
    if (src) {
      await paper.flipTo(src.page, run);
      from = (await paper.sweep(sb.block.id, run)) ?? from;
    }
    flights.push(fly(scene, sb, scene.screenToWorld(from.x, from.y), run).then(() => cam?.landed(sb)));
    await run.wait(40);
  }
  await Promise.all(flights);
  if (run.cancelled) return;
  await run.wait(500);

  // Paper leaves, model slides to center, everything clicks together bottom-up.
  paper.show(false);
  cam?.reveal();
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
  cam?.settle();
  await run.tween(600, (t) => (scene.groupAlpha = t));
  if (!run.cancelled) scene.startPulse();
  cam?.pulse();
}

const HERO_HOLD = 1500; // ms the hero shot holds before landings reframe again

/** The director's cues for one assembly, in world space. */
function director(scene: ForgeScene, graph: LocatedArchitecture, run: Run) {
  const layout = scene.layout!;
  const hero = pickHero(graph.blocks, layout);
  const dist = () => scene.distanceFor(0);
  const lens = () => [scene.camera.aspect, scene.camera.fov] as const;
  const landed: Vec3[] = [];
  let holdUntil = 0;
  // Skip (or cancel) lands on the resting angle at once and hands the camera back to the sway.
  const rest = () => {
    if (scene.rig.handedOver) return;
    scene.rig.jump(restingShot(layout, dist()));
    scene.rig.release();
    scene.sway = !scene.reduceMotion;
  };
  // A later assembly (Replay doesn't cancel this run) loads a new layout: this one's cues stop.
  const ours = () => scene.layout === layout;
  const live = () => ours() && !run.skipping;
  return {
    start(first: THREE.Vector3) {
      if (live()) scene.rig.jump(discoverShot([first], 0, ...lens()));
    },
    landed(sb: SceneBlock) {
      if (!live()) return;
      const at = sb.obj.root.position;
      landed.push({ x: at.x, y: at.y, z: at.z });
      if (sb.block.id === hero) {
        scene.rig.cue(heroShot(at, dist()));
        holdUntil = performance.now() + HERO_HOLD;
      } else if (performance.now() >= holdUntil) {
        scene.rig.cue(discoverShot(landed, landed.length, ...lens()));
      }
    },
    reveal() {
      if (live()) scene.rig.cue(revealShot(layout, dist()));
    },
    settle() {
      if (live()) scene.rig.cue(restingShot(layout, dist()));
    },
    pulse() {
      if (!ours()) return;
      if (run.skipping) return rest();
      scene.rig.follow(() => {
        const head = scene.pulseHead();
        return head && followPoint(head);
      }, followShot(dist()));
      void run.wait(scene.pulseCycle * 1000).then(() => {
        if (run.cancelled || !ours()) return;
        if (run.skipping) rest();
        else scene.rig.cue(restingShot(layout, dist()), true);
      });
    },
  };
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
