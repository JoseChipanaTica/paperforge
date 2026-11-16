import { type View, cleanView } from "../shared/learn";
import type { NodeTrace, Trace } from "../shared/train/run";
import type { TrainingArtifact } from "../shared/training";
import type { TourScript, TourStep } from "./tour";

/**
 * One training step as an ordinary tour: the loss at the output, one step per block going backward (the gradient
 * arriving and what the weights should change), the update, and the same example again. Every number comes from the
 * `Trace` that `train()` computed; the artifact only brings words. Pure, no DOM.
 */

export const PASSTHROUGH = "This toy can't compute this block, so the gradient passes through unchanged.";

const dims = (m: number[][]) => `[${m.length}, ${m[0]?.length ?? 0}]`;
const loss3 = (x: number) => x.toFixed(3);
/** A rate as written: 0.0007, 0.05, 1000. */
const rate = (x: number) => String(Number(x.toPrecision(3)));
const pct = (p: number) => `${Math.round(p * 100)}%`;

function view(kind: View["kind"], caption: string, values: number[][], rows: string[] = [], cols: string[] = [], d = dims(values)): View {
  return cleanView({ kind, caption, rows, cols, values, dims: d, split: 1 });
}

/** The token ids of an input node, when its value is one row of ids inside the vocabulary. */
function idsOf(n: NodeTrace, vocab: string[]): number[] | null {
  if (n.kind !== "input" || n.value.length !== 1) return null;
  const ids = n.value[0]!;
  return ids.length > 0 && ids.every((i) => Number.isInteger(i) && i >= 0 && i < vocab.length) ? ids : null;
}

/**
 * Row labels for a node's matrices: the words of the nearest input before it in forward order (the source sentence
 * for the encoder, the shifted target for the decoder) when the rows are its positions, else "pos 1…".
 */
function rowLabels(trace: Trace, node: NodeTrace, vocab: string[], count: number): string[] {
  const at = trace.nodes.indexOf(node);
  for (let i = at < 0 ? trace.nodes.length - 1 : at; i >= 0; i--) {
    const ids = idsOf(trace.nodes[i]!, vocab);
    if (ids) return ids.length === count ? ids.map((id) => vocab[id]!) : positions(count);
  }
  return positions(count);
}
const positions = (n: number) => Array.from({ length: n }, (_, i) => `pos ${i + 1}`);

/** Column labels: the vocabulary for the tail of the model that is already one score per word. */
function colLabels(trace: Trace, node: NodeTrace, vocab: string[], width: number): string[] {
  const out = trace.predictionBefore[0]?.length ?? 0;
  if (!vocab.length || width !== vocab.length || out !== vocab.length) return [];
  const at = trace.nodes.indexOf(node);
  return trace.nodes.slice(at).every((n) => (n.value[0]?.length ?? 0) === out) ? vocab : [];
}

const size = (m: number[][]) => m.length * (m[0]?.length ?? 0);

/**
 * What the specimen card shows at one node: the gradient arriving (∂L/∂output), then either what its largest weight
 * should change or the gradient it sends back. An input node shows the example itself: data gets no update.
 */
export function traceViews(trace: Trace, node: NodeTrace, vocab: string[]): View[] {
  if (node.kind === "input") {
    const ids = idsOf(node, vocab);
    return ids
      ? [view("tokens", "the example's token ids: data, not weights", [ids], ids.map((i) => vocab[i]!), [], `[${ids.length}]`)]
      : [view("matrix", "the example's data: not weights", node.value, positions(node.value.length))];
  }
  const rows = (m: number[][]) => rowLabels(trace, node, vocab, m.length);
  const matrix = (caption: string, m: number[][]) => view("matrix", caption, m, rows(m), colLabels(trace, node, vocab, m[0]?.length ?? 0));
  const views = [matrix("gradient arriving", node.gradOut)];
  const param = node.params.reduce<NodeTrace["params"][number] | null>((best, p) => (!best || size(p.grad) > size(best.grad) ? p : best), null);
  if (param) {
    const r = node.kind === "embedding_lookup" && param.grad.length === vocab.length ? vocab : [];
    views.push(view("matrix", `what ${param.name} should change`, param.grad, r));
  } else if (node.kind === "residual_add" && node.gradIn[1]) {
    views.push(matrix("gradient leaving along the skip (the sublayer gets the same)", node.gradIn[1]));
  } else if (node.gradIn[0]) {
    views.push(matrix("gradient leaving", node.gradIn[0]));
  }
  return views;
}

const softmaxRow = (r: number[]) => {
  const m = Math.max(...r);
  const e = r.map((x) => Math.exp(x - m));
  const s = e.reduce((a, b) => a + b, 0);
  return e.map((x) => x / s);
};
const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** What the model's output means: probabilities per row when the loss reads it that way, else the raw values. */
function asProbs(a: TrainingArtifact, out: number[][]): number[][] {
  const last = a.toy.nodes.at(-1);
  const fn = last?.fn.trim().toLowerCase() ?? "";
  const isProbs = last?.kind === "softmax" || (last?.kind === "activation" && fn === "sigmoid");
  if (isProbs || a.loss.kind === "mse") return out;
  return a.loss.kind === "cross_entropy" ? out.map(softmaxRow) : out.map((r) => r.map(sigmoid));
}

/** The target as the loss sees it: label smoothing puts 1 − ε on the right class and ε / (V − 1) on each other. */
function targetRows(a: TrainingArtifact, width: number): number[][] {
  if (!a.target.classes.length) return a.target.values;
  const eps = a.loss.kind === "cross_entropy" ? a.loss.label_smoothing : 0;
  const other = width > 1 ? eps / (width - 1) : 0;
  return a.target.classes.map((c) => Array.from({ length: width }, (_, j) => (j === c ? 1 - eps : other)));
}

export function trainingScript(a: TrainingArtifact, trace: Trace): TourScript {
  const nodes = trace.nodes;
  const last = nodes.at(-1)!;
  const vocab = a.vocab;
  const before = asProbs(a, trace.predictionBefore);
  const after = asProbs(a, trace.predictionAfter);
  const width = before[0]?.length ?? 0;
  const target = targetRows(a, width);
  const classes = a.target.classes.length > 0 && vocab.length === width;
  const outRows = rowLabels(trace, last, vocab, before.length);
  const at = before.length - 1; // the last position: the word the example ends on

  // Classification: the guess at the last position against its right word. Regression: the whole output.
  const where = outRows[at]?.startsWith("pos ") ? `at ${outRows[at]}` : `after “${outRows[at]}”`;
  const right = classes ? vocab[a.target.classes[at]!]! : "";
  const guess = (m: number[][], when: string) =>
    classes
      ? view("probs", `${when} ${where}`, [m[at]!], vocab, [], `[${vocab.length}]`)
      : view("matrix", when, m, outRows, [], dims(m));
  const targetView = classes
    ? view("probs", `the right word: “${right}”${a.loss.label_smoothing > 0 && a.loss.kind === "cross_entropy" ? " (smoothed)" : ""}`, [target[at]!], vocab, [], `[${vocab.length}]`)
    : view("matrix", "the target", target, outRows, [], dims(target));
  const rightProb = (m: number[][]) => (classes ? m[at]![a.target.classes[at]!]! : NaN);

  const lossStep: TourStep = {
    block_id: last.block_id,
    title: "The loss",
    what: "How wrong the guess is, as one number.",
    body: [
      a.loss.caption,
      `L = ${loss3(trace.lossBefore)}.`,
      a.loss.kind === "cross_entropy" && a.loss.label_smoothing > 0
        ? `With label smoothing ε = ${a.loss.label_smoothing}, the target puts ${pct(1 - a.loss.label_smoothing)} on the right word and spreads the rest over the others.`
        : "",
    ]
      .filter(Boolean)
      .join(" "),
    shape: `${dims(trace.predictionBefore)} → 1 number`,
    views: classes
      ? [
          view("matrix", "every position's guess", before, outRows, vocab),
          view("matrix", "what it should be", target, outRows, vocab),
        ]
      : [view("matrix", "the error: output − target", before.map((r, i) => r.map((x, j) => x - (target[i]?.[j] ?? 0))), outRows)],
  };

  const backward: TourStep[] = a.steps.flatMap((s) => {
    const own = nodes.filter((n) => n.block_id === s.block_id);
    if (!own.length) return [];
    const end = own.at(-1)!;
    const first = own[0]!;
    const passthrough = own.every((n) => n.passthrough);
    const leaving = first.kind === "embedding_lookup" ? undefined : first.gradIn[0]; // token ids take no gradient
    return [
      {
        block_id: s.block_id,
        title: s.title,
        what: s.learns,
        body: passthrough ? `${s.narration} ${PASSTHROUGH}` : s.narration,
        shape: leaving ? `gradient ${dims(end.gradOut)} → ${dims(leaving)}` : `gradient ${dims(end.gradOut)}`,
        views: traceViews(trace, end, vocab),
      },
    ];
  });

  const o = a.optimizer;
  const learner = nodes.find((n) => n.params.length > 0) ?? last;
  const sameRate = Math.abs(o.lr - trace.lr) <= 1e-12 * Math.max(1, Math.abs(o.lr));
  const updateStep: TourStep = {
    block_id: learner.block_id,
    title: "Update every weight",
    what: "Every weight moves a little against its gradient.",
    body: [
      o.fallback
        ? `The paper trains with ${o.kind}; the toy can't run it, so it takes a plain SGD step instead.`
        : `${o.kind} moves every weight once.`,
      sameRate
        ? `The toy uses the paper's learning rate, ${rate(o.lr)}.`
        : `The paper's learning rate is ${rate(o.lr)}; this toy uses ${rate(trace.lr)}, a toy-scale rate, so a single step on a single example visibly moves the loss.`,
      o.schedule_note,
      trace.note,
    ]
      .map((x) => x.trim())
      .filter(Boolean)
      .join(" "),
    shape: "",
    views: [beforeAfter(learner)],
  };

  const dropped = trace.lossAfter < trace.lossBefore;
  const [l0, l1] = [loss3(trace.lossBefore), loss3(trace.lossAfter)];
  const moved = classes ? ` “${right}” ${pct(rightProb(before))} → ${pct(rightProb(after))}.` : "";
  const againStep: TourStep = {
    block_id: last.block_id,
    title: "Guess again",
    what: "The same example runs forward with the new weights.",
    body: dropped
      ? `Loss ${l0} → ${l1}.${moved}`
      : `The loss went from ${l0} to ${l1}: one step at toy size can overshoot.${moved}`,
    shape: `${dims(trace.predictionAfter)} → 1 number`,
    views: [guess(after, "the new guess"), targetView],
  };

  return {
    analogy: { title: "One training step", text: a.intro },
    intro: { title: "Training", body: a.intro, label: "Train on this example", views: [guess(before, "the guess"), targetView] },
    steps: [lossStep, ...backward, updateStep, againStep],
    output: { label: "After one training step", views: [guess(after, "the new guess")] },
    outro: a.outro,
  };
}

/** Four weights of a node before and after the step: the ones that moved most. */
function beforeAfter(n: NodeTrace): View {
  const moves = n.params.flatMap((p) =>
    p.before.flatMap((r, i) => r.map((b, j) => ({ label: `${p.name}[${i},${j}]`, b, a: p.after[i]?.[j] ?? b }))),
  );
  const top = moves.sort((x, y) => Math.abs(y.a - y.b) - Math.abs(x.a - x.b)).slice(0, 4);
  return view(
    "matrix",
    "four weights, before → after",
    [top.map((m) => m.b), top.map((m) => m.a)],
    ["before", "after"],
    top.map((m) => m.label),
  );
}
