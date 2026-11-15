import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import type { JourneyArtifact, View } from "../shared/learn";
import type { ForgeScene } from "./scene";
import { TourDriver, tourSteps, type SpecimenHost } from "./tour";

const SAMPLES = join(import.meta.dir, "../../samples");
const journey = (await Bun.file(join(SAMPLES, "1706.03762.journey.json")).json()) as JourneyArtifact;

describe("tourSteps", () => {
  test("journey: analogy, input and output frame the steps", () => {
    const t = tourSteps(journey);
    expect(t.analogy).toEqual(journey.analogy);
    expect(t.intro.body).toBe(journey.intro);
    expect(t.intro.label).toBe(journey.input.label);
    expect(t.intro.views).toEqual(journey.input.views);
    expect(t.steps.map((s) => s.block_id)).toEqual(journey.steps.map((s) => s.block_id));
    expect(t.steps[0]!.what).toBe(journey.steps[0]!.what_happens);
    expect(t.steps[0]!.body).toBe(journey.steps[0]!.narration);
    expect(t.steps[0]!.shape).toBe(journey.steps[0]!.shape_change);
    expect(t.steps[0]!.views).toEqual(journey.steps[0]!.views);
    expect(t.output).toEqual(journey.output);
    expect(t.outro).toBe(journey.outro);
  });
});

describe("TourDriver", () => {
  const fakeScene = { highlight: () => {}, focusOn: () => Promise.resolve() } as unknown as ForgeScene;

  test("Next and Back still work after Esc stopped the tour", () => {
    const driver = new TourDriver(fakeScene);
    const script = tourSteps(journey);
    driver.goTo(script, 3);
    driver.stop();
    driver.next();
    expect(driver.active).toBe(true);
    expect(driver.index).toBe(4);
    driver.stop();
    driver.back();
    expect(driver.index).toBe(3);
  });
});

describe("TourDriver and the specimen", () => {
  const fakeScene = { highlight: () => {}, focusOn: () => Promise.resolve() } as unknown as ForgeScene;
  const script = tourSteps(journey);
  const ids = script.steps.map((s) => s.block_id);
  function recording() {
    const shown: [View[], string, string | null][] = [];
    let hidden = 0;
    const host: SpecimenHost = {
      show: (views, at, from) => (shown.push([views, at, from]), Promise.resolve()),
      hide: () => void hidden++,
    };
    return { host, shown, hidden: () => hidden };
  }

  test("each step shows its views, flying from the previous block", () => {
    const r = recording();
    const driver = new TourDriver(fakeScene, r.host);
    driver.goTo(script, 0);
    driver.next();
    expect(r.shown).toEqual([
      [script.steps[0]!.views, ids[0]!, null],
      [script.steps[1]!.views, ids[1]!, ids[0]!],
    ]);
  });

  test("the intro shows the input and the outro shows the output", () => {
    const r = recording();
    const driver = new TourDriver(fakeScene, r.host);
    driver.goTo(script, -1);
    driver.goTo(script, script.steps.length);
    expect(r.shown[0]).toEqual([script.intro.views, ids[0]!, null]);
    expect(r.shown[1]).toEqual([script.output.views, ids.at(-1)!, null]);
  });

  test("jumping steps doesn't fly", () => {
    const r = recording();
    const driver = new TourDriver(fakeScene, r.host);
    driver.goTo(script, 0);
    driver.goTo(script, 5);
    expect(r.shown[1]![2]).toBeNull();
  });

  test("re-rendering at the same step (a new language) keeps the same data", () => {
    const r = recording();
    const driver = new TourDriver(fakeScene, r.host);
    const translated = tourSteps({ ...journey, steps: journey.steps.map((s) => ({ ...s, title: `ES ${s.title}` })) });
    driver.goTo(script, 3);
    driver.goTo(translated, 3);
    expect(r.shown[1]![0]).toEqual(r.shown[0]![0]);
    expect(r.shown[1]![2]).toBeNull();
  });

  test("stop and forget take the specimen off the model", () => {
    const r = recording();
    const driver = new TourDriver(fakeScene, r.host);
    driver.goTo(script, 2);
    driver.stop();
    expect(r.hidden()).toBe(1);
    driver.goTo(script, 2);
    driver.forget();
    expect(r.hidden()).toBe(2);
  });
});

describe("TourDriver Play", () => {
  test("waits for the step's figure before moving on", async () => {
    const fakeScene = { highlight: () => {}, focusOn: () => Promise.resolve() } as unknown as ForgeScene;
    const script = tourSteps(journey);
    let release!: () => void;
    const driver = new TourDriver(fakeScene, null, { playMs: 5, waitFor: () => new Promise<void>((r) => (release = r)) });
    driver.goTo(script, 0);
    driver.togglePlay();
    await new Promise((r) => setTimeout(r, 10));
    expect(driver.index).toBe(1); // Play's first beat is immediate
    await new Promise((r) => setTimeout(r, 40));
    expect(driver.index).toBe(1); // well past playMs, still waiting on step 2's figure
    release();
    await new Promise((r) => setTimeout(r, 20));
    expect(driver.index).toBe(2);
    driver.pause();
    expect(driver.playing).toBe(false);
  });
});
