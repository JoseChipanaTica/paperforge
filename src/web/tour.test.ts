import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import type { ExampleArtifact, WalkthroughArtifact } from "../shared/learn";
import { tourSteps } from "./tour";

const SAMPLES = join(import.meta.dir, "../../samples");
const walkthrough = (await Bun.file(join(SAMPLES, "1706.03762.walkthrough.json")).json()) as WalkthroughArtifact;
const example = (await Bun.file(join(SAMPLES, "1706.03762.example.json")).json()) as ExampleArtifact;

describe("tourSteps", () => {
  test("walkthrough: one step per block, narration as body, shape change as detail", () => {
    const t = tourSteps("walkthrough", walkthrough);
    expect(t.steps).toHaveLength(18);
    expect(t.steps[0]!.body).toBe(walkthrough.steps[0]!.narration);
    expect(t.steps[0]!.detail).toBe(walkthrough.steps[0]!.shape_change);
    expect(t.steps[0]!.title).toBe(walkthrough.steps[0]!.title);
    expect(t.outro).toBe(walkthrough.outro);
  });

  test("example: analogy first, toy values as detail, takeaway last", () => {
    const t = tourSteps("example", example);
    expect(t.intro.title).toBe(example.analogy.title);
    expect(t.intro.body).toBe(example.analogy.text);
    expect(t.steps[3]!.detail).toBe(example.steps[3]!.toy_value);
    expect(t.steps[3]!.title).toBe(""); // the tour shows the block's label instead
    expect(t.outro).toBe(example.takeaway);
  });
});
