import { describe, expect, test } from "bun:test";
import type { Architecture } from "../shared/graph";
import type { View } from "../shared/learn";
import { RAISING_TOY, TRANSFORMER_TOY } from "../shared/train/fixtures";
import { train } from "../shared/train/run";
import type { TrainingArtifact } from "../shared/training";
import { applyText, pickText } from "../shared/translate";
import type { TourScript } from "./tour";
import { traceViews, trainingScript } from "./trainingTour";

const graph = (await Bun.file(new URL("../../samples/1706.03762.json", import.meta.url)).json()) as Architecture;
const sample = (await Bun.file(new URL("../../samples/1706.03762.training.json", import.meta.url)).json()) as TrainingArtifact;

const script = (a: TrainingArtifact) => trainingScript(a, train(a));
const allViews = (s: TourScript): View[] => [...s.intro.views, ...s.steps.flatMap((st) => st.views), ...s.output.views];

describe("trainingScript", () => {
  const s = script(TRANSFORMER_TOY);

  test("the script starts at the loss and ends with update then again", () => {
    expect(s.steps[0]!.block_id).toBe("softmax");
    expect(s.steps[0]!.body).toMatch(/L = \d+\.\d{3}\b/);
    expect(s.steps[0]!.body).toContain(TRANSFORMER_TOY.loss.caption);
    const [update, again] = s.steps.slice(-2);
    expect(update!.block_id).toBe("input_embedding"); // the first node with weights
    expect(update!.body).toContain("Adam");
    expect(update!.body).toContain("0.0007"); // the paper's rate
    expect(update!.body).toContain("0.05"); // the toy's rate
    expect(update!.views.some((v) => v.kind === "matrix" && v.values.length === 2 && v.values[0]!.length === 4)).toBe(true);
    expect(again!.block_id).toBe("softmax");
    expect(again!.body).toMatch(/Loss \d+\.\d{3} → \d+\.\d{3}/);
    expect(again!.views.some((v) => v.kind === "probs")).toBe(true);
    expect(s.intro.views.map((v) => v.kind)).toEqual(["probs", "probs"]);
    expect(s.output.views[0]!.kind).toBe("probs");
    expect(s.outro).toBe(TRANSFORMER_TOY.outro);
  });

  test("backward steps follow the artifact's reverse order", () => {
    const backward = s.steps.slice(1, -2);
    expect(backward.map((st) => st.block_id)).toEqual(TRANSFORMER_TOY.steps.map((st) => st.block_id));
    expect(backward.map((st) => st.what)).toEqual(TRANSFORMER_TOY.steps.map((st) => st.learns));
    expect(backward[0]!.views[0]!.caption).toContain("gradient arriving");
  });

  test("every step's block is in the graph", () => {
    const ids = new Set(graph.blocks.map((b) => b.id));
    for (const a of [TRANSFORMER_TOY, sample]) for (const st of script(a).steps) expect(ids.has(st.block_id)).toBe(true);
  });

  test("the again step reports a rise honestly", () => {
    const t = train(RAISING_TOY);
    expect(t.lossAfter).toBeGreaterThan(t.lossBefore);
    const again = trainingScript(RAISING_TOY, t).steps.at(-1)!;
    expect(again.body).toContain(`${t.lossBefore.toFixed(3)} to ${t.lossAfter.toFixed(3)}`);
    expect(again.body).toContain("overshoot");
  });

  test("a translated artifact gives the same step count and block ids", () => {
    const text = pickText("training", TRANSFORMER_TOY) as { steps: { block_id: string }[] };
    const fake = JSON.parse(JSON.stringify(text, (_, v) => (typeof v === "string" ? `«${v}»` : v)));
    fake.steps = text.steps.map((st, i) => ({ ...fake.steps[i], block_id: st.block_id }));
    const translated = applyText("training", TRANSFORMER_TOY, fake);
    const t = script(translated);
    expect(t.steps.length).toBe(s.steps.length);
    expect(t.steps.map((st) => st.block_id)).toEqual(s.steps.map((st) => st.block_id));
    expect(t.steps[1]!.title).toStartWith("«");
  });

  test("all view values are finite and ≤ 12×12", () => {
    for (const a of [TRANSFORMER_TOY, RAISING_TOY, sample]) {
      for (const v of allViews(script(a))) {
        expect(v.values.length).toBeLessThanOrEqual(12);
        for (const r of v.values) {
          expect(r.length).toBeLessThanOrEqual(12);
          for (const x of r) expect(Number.isFinite(x)).toBe(true);
        }
      }
    }
  });

  test("pass-through blocks say so", () => {
    const a: TrainingArtifact = {
      ...TRANSFORMER_TOY,
      toy: {
        ...TRANSFORMER_TOY.toy,
        nodes: TRANSFORMER_TOY.toy.nodes.map((n) => (n.id === "enc_pos" ? { ...n, kind: "mystery" } : n)),
      },
    };
    const st = script(a).steps.find((x) => x.block_id === "encoder_positional")!;
    expect(st.body).toContain("This toy can't compute this block, so the gradient passes through unchanged.");
  });

  test("the update step names the fallback and Adam notes", () => {
    const t = train(TRANSFORMER_TOY);
    const update = trainingScript(TRANSFORMER_TOY, t).steps.at(-2)!;
    if (t.note) expect(update.body).toContain(t.note);
    expect(update.body).toContain(TRANSFORMER_TOY.optimizer.schedule_note);
    const odd = { ...TRANSFORMER_TOY, optimizer: { ...TRANSFORMER_TOY.optimizer, kind: "Adafactor", fallback: true } };
    expect(script(odd).steps.at(-2)!.body).toContain("plain SGD");
  });
});

describe("traceViews", () => {
  const t = train(TRANSFORMER_TOY);
  const vocab = TRANSFORMER_TOY.vocab;
  const node = (id: string) => t.nodes.find((n) => n.id === id)!;

  test("at most two views, gradient arriving first, token rows labelled", () => {
    const vs = traceViews(t, node("enc_attn"), vocab);
    expect(vs.length).toBeLessThanOrEqual(2);
    expect(vs[0]!.kind).toBe("matrix");
    expect(vs[0]!.caption).toContain("gradient arriving");
    expect(vs[0]!.rows).toEqual(["the", "cat", "sat"]);
    expect(vs[0]!.dims).toBe("[3, 4]");
    expect(vs[1]!.caption).toContain("what Wq should change");
  });

  test("decoder rows come from the decoder's own input", () => {
    expect(traceViews(t, node("final_linear"), vocab)[0]!.rows).toEqual(["<s>", "die", "Katze"]);
  });

  test("a node without weights shows the gradient leaving", () => {
    const vs = traceViews(t, node("enc_ffn_relu"), vocab);
    expect(vs[1]!.caption).toContain("gradient leaving");
  });

  test("rows fall back to positions when there are no token ids", () => {
    const r = train(RAISING_TOY);
    expect(traceViews(r, r.nodes[1]!, [])[0]!.rows).toEqual(["pos 1", "pos 2"]);
  });
});
