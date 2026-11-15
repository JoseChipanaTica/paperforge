import { describe, expect, test } from "bun:test";
import { join } from "node:path";
import { Architecture } from "./graph";
import type { JourneyArtifact } from "./learn";
import { MotionArtifact, cleanMotion, motionFits, type Op } from "./motion";

const SAMPLES = join(import.meta.dir, "../../samples");
const graph = Architecture.parse(await Bun.file(join(SAMPLES, "1706.03762.json")).json());
const journey = (await Bun.file(join(SAMPLES, "1706.03762.journey.json")).json()) as JourneyArtifact;
const ids = journey.steps.map((s) => s.block_id);

const op = (o: Partial<Op>): Op => ({
  kind: "linear",
  caption: "c",
  tokens: [],
  input: [[1, 2]],
  weights: [[1], [1]],
  extra: [],
  fn: "",
  size: 1,
  stride: 1,
  count: 1,
  svg: "",
  ...o,
});
const motion = (steps: MotionArtifact["steps"]): MotionArtifact => ({ steps });
const only = (o: Partial<Op>) => cleanMotion(motion([{ block_id: ids[0]!, ops: [op(o)] }]), graph, journey).steps[0]!.ops[0]!;

describe("cleanMotion", () => {
  test("drops unknown blocks and keeps the journey's order", () => {
    const m = cleanMotion(motion([{ block_id: ids[2]!, ops: [op({})] }, { block_id: "nope", ops: [op({})] }, { block_id: ids[0]!, ops: [op({})] }]), graph, journey);
    expect(m.steps.map((s) => s.block_id)).toEqual([ids[0]!, ids[2]!]);
  });

  test("clamps matrices, size, stride and count", () => {
    const big = Array.from({ length: 10 }, () => Array(10).fill(1));
    const o = only({ kind: "convolution", input: big, weights: big, size: 9, stride: 0, count: 9999 });
    expect(o.input.length).toBe(8);
    expect(o.input[0]!.length).toBe(8);
    expect(o.weights.length).toBe(8);
    expect([o.size, o.stride, o.count]).toEqual([5, 1, 512]);
  });

  test("pads ragged rows and zeroes non-finite numbers", () => {
    expect(only({ input: [[1, 2, 3], [4], [NaN, Infinity, 5]] }).input).toEqual([[1, 2, 3], [4, 0, 0], [0, 0, 5]]);
  });

  test("keeps at most three ops", () => {
    const m = cleanMotion(motion([{ block_id: ids[0]!, ops: [op({}), op({}), op({}), op({})] }]), graph, journey);
    expect(m.steps[0]!.ops.length).toBe(3);
  });

  test("an op that can't be computed becomes a captioned placeholder", () => {
    expect(only({ kind: "convolution", caption: "Slide the kernel", input: [[1]], weights: [] })).toMatchObject({
      kind: "custom",
      caption: "Slide the kernel",
      svg: "",
    });
    expect(only({ kind: "attention", weights: [[1]], extra: [[1]] }).kind).toBe("custom");
    expect(only({ kind: "message_passing", extra: [] }).kind).toBe("custom");
  });

  test("self-attention with queries in the input is computable", () => {
    expect(only({ kind: "attention", input: [[1, 0], [0, 1]], weights: [], extra: [[1, 0], [0, 1], [1, 2], [3, 4]] }).kind).toBe("attention");
  });

  test("custom svg is sanitized", () => {
    const o = only({ kind: "custom", svg: '<svg><script>alert(1)</script><rect width="1"/></svg>' });
    expect(o.svg).toBe('<svg xmlns="http://www.w3.org/2000/svg"><rect width="1"/></svg>');
  });

  test("works without a journey (graph order only)", () => {
    expect(cleanMotion(motion([{ block_id: "nope", ops: [op({})] }, { block_id: ids[1]!, ops: [op({})] }]), graph, null).steps.length).toBe(1);
  });
});

describe("motionFits", () => {
  const all = () => motion(ids.map((block_id) => ({ block_id, ops: [op({})] })));

  test("motion that animates only a corner of the walkthrough doesn't fit", () => {
    expect(motionFits(motion([{ block_id: ids[0]!, ops: [op({})] }]), journey)).toBe(false); // VGG16: 1 of 18
    expect(motionFits(motion(all().steps.slice(0, Math.ceil(ids.length / 2))), journey)).toBe(true);
  });

  test("true for the journey's blocks, false for a block the journey doesn't visit", () => {
    expect(motionFits(all(), journey)).toBe(true);
    expect(motionFits(motion([...all().steps, { block_id: "elsewhere", ops: [op({})] }]), journey)).toBe(false);
  });
});

describe("committed motion samples", () => {
  for (const [id, needs] of [["1706.03762", ["attention"]], ["1512.03385", ["convolution", "pooling"]]] as const) {
    test(`${id}: clean, fits its walkthrough, and the math runs`, async () => {
      const { runChain } = await import("../web/motion/compute");
      const g = Architecture.parse(await Bun.file(join(SAMPLES, `${id}.json`)).json());
      const j = (await Bun.file(join(SAMPLES, `${id}.journey.json`)).json()) as JourneyArtifact;
      const m = MotionArtifact.parse(await Bun.file(join(SAMPLES, `${id}.motion.json`)).json());
      expect(cleanMotion(m, g, j)).toEqual(m);
      expect(motionFits(m, j)).toBe(true);
      const results = m.steps.flatMap((s) => runChain(s.ops));
      expect(results.filter((r) => r.result.ok).length / results.length).toBeGreaterThanOrEqual(0.8);
      const kinds = new Set(m.steps.flatMap((s) => s.ops.map((o) => o.kind)));
      for (const k of needs) expect(kinds.has(k)).toBe(true);
    });
  }
});
