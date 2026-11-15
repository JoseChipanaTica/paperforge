import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import type { JourneyArtifact } from "../shared/learn";
import type { ForgeScene } from "./scene";
import { TourDriver, tourSteps } from "./tour";

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
