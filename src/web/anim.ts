export const ease = {
  linear: (t: number) => t,
  outCubic: (t: number) => 1 - (1 - t) ** 3,
  inOutCubic: (t: number) => (t < 0.5 ? 4 * t ** 3 : 1 - (-2 * t + 2) ** 3 / 2),
  inCubic: (t: number) => t ** 3,
  outBack: (t: number) => {
    const c1 = 1.70158;
    const c3 = c1 + 1;
    return 1 + c3 * (t - 1) ** 3 + c1 * (t - 1) ** 2;
  },
};

/**
 * A cancellable run of tweens. `skip()` jumps every pending and future tween to its end
 * state; `cancel()` does the same and lets the owner bail out of its async sequence.
 */
export class Run {
  skipping = false;
  cancelled = false;

  skip() {
    this.skipping = true;
  }

  cancel() {
    this.cancelled = this.skipping = true;
  }

  tween(ms: number, fn: (t: number) => void, easing: (t: number) => number = ease.inOutCubic): Promise<void> {
    if (this.skipping || ms <= 0) {
      fn(1);
      return Promise.resolve();
    }
    return new Promise((resolve) => {
      const start = performance.now();
      const step = (now: number) => {
        const t = this.skipping ? 1 : Math.min(1, (now - start) / ms);
        fn(easing(t));
        if (t < 1) requestAnimationFrame(step);
        else resolve();
      };
      requestAnimationFrame(step);
    });
  }

  wait(ms: number): Promise<void> {
    return this.tween(ms, () => {});
  }
}
