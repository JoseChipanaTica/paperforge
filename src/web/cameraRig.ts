import type { Shot, Vec3 } from "./director";

const SUBSTEP = 1 / 120;
const SETTLED = 0.01;

/**
 * Springs the camera between shots. Holds the live `controls.target` and `camera.position` and moves them in place,
 * critically damped so a new cue mid-move bends the path instead of jumping. Inactive, it touches nothing.
 */
export class CameraRig {
  private on = false;
  private goal: Vec3 = { x: 0, y: 0, z: 0 };
  private offset: Vec3 = { x: 0, y: 0, z: 0 };
  private omega = 1;
  private vTarget: Vec3 = { x: 0, y: 0, z: 0 };
  private vPosition: Vec3 = { x: 0, y: 0, z: 0 };
  private tracking: (() => Vec3 | null) | null = null;
  private releaseWhenSettled = false;

  constructor(
    private target: Vec3,
    private position: Vec3,
  ) {}

  get active() {
    return this.on;
  }

  cue(shot: Shot, releaseWhenSettled = false) {
    this.aim(shot);
    this.tracking = null;
    this.releaseWhenSettled = releaseWhenSettled;
  }

  follow(at: () => Vec3 | null, shot: Shot) {
    this.aim(shot);
    this.tracking = at;
    this.releaseWhenSettled = false;
  }

  jump(shot: Shot) {
    this.aim(shot);
    this.tracking = null;
    this.releaseWhenSettled = false;
    set(this.target, this.goal);
    set(this.position, add(this.goal, this.offset));
    zero(this.vTarget);
    zero(this.vPosition);
  }

  release() {
    this.on = false;
    this.tracking = null;
    zero(this.vTarget);
    zero(this.vPosition);
  }

  step(dt: number) {
    if (!this.on) return;
    const at = this.tracking?.();
    if (at) set(this.goal, at);
    const goalPosition = add(this.goal, this.offset);
    for (let left = dt; left > 1e-9; left -= SUBSTEP) {
      const h = Math.min(SUBSTEP, left);
      spring(this.target, this.vTarget, this.goal, this.omega, h);
      spring(this.position, this.vPosition, goalPosition, this.omega, h);
    }
    if (this.releaseWhenSettled && settled(this.target, this.vTarget, this.goal) && settled(this.position, this.vPosition, goalPosition)) {
      this.release();
    }
  }

  private aim(shot: Shot) {
    if (!this.on) {
      zero(this.vTarget);
      zero(this.vPosition);
    }
    this.on = true;
    this.goal = { ...shot.target };
    this.offset = { ...shot.offset };
    this.omega = 6 / (Math.max(1, shot.ms) / 1000);
  }
}

const AXES = ["x", "y", "z"] as const;

function spring(x: Vec3, v: Vec3, goal: Vec3, omega: number, h: number) {
  for (const k of AXES) {
    v[k] += (omega * omega * (goal[k] - x[k]) - 2 * omega * v[k]) * h;
    x[k] += v[k] * h;
  }
}

function settled(x: Vec3, v: Vec3, goal: Vec3) {
  return AXES.every((k) => Math.abs(goal[k] - x[k]) < SETTLED && Math.abs(v[k]) < SETTLED);
}

const add = (a: Vec3, b: Vec3): Vec3 => ({ x: a.x + b.x, y: a.y + b.y, z: a.z + b.z });
const set = (to: Vec3, from: Vec3) => {
  to.x = from.x;
  to.y = from.y;
  to.z = from.z;
};
const zero = (v: Vec3) => set(v, { x: 0, y: 0, z: 0 });
