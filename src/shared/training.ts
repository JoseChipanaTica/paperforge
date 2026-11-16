import { z } from "zod";
import type { Architecture } from "./graph";
import { TOY_KINDS } from "./train/ops";
import { optimizerKind, train } from "./train/run";

/**
 * The training walkthrough's contract: a tiny toy version of the paper's model (a DAG of nodes, each tied to a 3D
 * block), the example it trains on, the paper's loss and optimizer, and the prose of the tour. Claude writes only
 * structure, small starting values and words; `train()` (src/shared/train/run.ts) computes every loss and gradient.
 */

export { optimizerKind } from "./train/run";

const Matrix = z.array(z.array(z.number()));

export const ToyNode = z.object({
  id: z.string().describe("Unique node id, e.g. 'enc_attn'. Later nodes refer to it in `inputs`"),
  block_id: z.string().describe("id of the block in <graph> this node belongs to; several nodes may share one block"),
  kind: z
    .string()
    .describe(
      "One of: input, embedding_lookup, positional_encoding, linear, activation, normalization, residual_add, attention, softmax, dropout, convolution, pooling, flatten, concat. Any other name is shown as a pass-through (identity forward, gradient passed back unchanged)",
    ),
  fn: z
    .string()
    .describe(
      "Variant: activation 'relu' | 'gelu' | 'sigmoid' | 'tanh' | 'silu'; normalization 'layer' | 'batch' | 'rms'; attention 'self' | 'causal' | 'cross'; pooling 'max' | 'avg'; positional_encoding '' (sinusoidal) | 'learned'; otherwise ''",
    ),
  inputs: z
    .array(z.string())
    .describe(
      "ids of earlier nodes this node reads, in order. input nodes: []. residual_add: always [sublayer output, skip]. attention 'cross': [queries, keys/values (the encoder output)]",
    ),
  params: z
    .array(
      z.object({
        name: z.string().describe("Parameter name"),
        values: Matrix.describe("≤ 8×8, rows of numbers; starting weights small (|w| ≤ 0.5)"),
      }),
    )
    .describe(
      "Named parameters. input: 'values' (one row of token ids, or a small data matrix — data, not learned). embedding_lookup: 'E' (vocab × width). linear: 'W' (in × out), optional 'b' (one row). normalization: 'gamma' and 'beta' (one row each; rms only gamma). attention: 'Wq', 'Wk', 'Wv', 'Wo' (width × width). positional_encoding 'learned': 'P' (positions × width). convolution: 'W' (one single-channel kh × kw kernel over an [H, W] input; stride 1, no padding), optional 'b' ([[bias]]). Others: []",
    ),
  count: z.number().int().describe("attention: number of heads (1–2, dividing the width); otherwise 1"),
});
export type ToyNode = z.infer<typeof ToyNode>;

export const TrainingArtifact = z.object({
  toy: z.object({
    dims: z.string().describe("One sentence on the reduction, e.g. 'd_model 512 → 4, 8 heads → 2, 6 layers → 1'"),
    nodes: z
      .array(ToyNode)
      .describe(
        "≤ 32 nodes in forward (data-flow) order; each reads only earlier nodes. Width ≤ 8 (hidden layers too), sequence ≤ 4, heads ≤ 2. A group the paper repeats N times is one toy layer. The last node is the model output (logits, probabilities or values). Pooling is max/avg over fixed non-overlapping 2×2 windows",
      ),
  }),
  vocab: z
    .array(z.string())
    .describe("≤ 8 labels for token ids and output classes (index = id), e.g. ['<s>', 'the', 'cat']; [] for regression"),
  target: z
    .object({
      classes: z
        .array(z.number().int())
        .describe("Classification / next token: the correct class index for each output row (indices into vocab); else []"),
      values: Matrix.describe("Regression: the target matrix, same shape as the output; else []"),
    })
    .describe("What the output should be: exactly one of `classes` or `values` is non-empty"),
  loss: z.object({
    kind: z.enum(["cross_entropy", "mse", "bce"]).describe("The paper's training loss"),
    label_smoothing: z.number().describe("Label smoothing ε (0 when the paper uses none)"),
    quote: z.string().describe("Verbatim sentence from the paper about the loss or label smoothing, or ''"),
    caption: z.string().describe("One short sentence on what this loss measures here"),
  }),
  optimizer: z.object({
    kind: z.string().describe("The paper's optimizer by name, e.g. 'Adam', 'SGD with momentum', 'AdamW', 'LAMB'"),
    lr: z.number().describe("The paper's own (peak) learning rate, as the paper states or its schedule implies, e.g. 0.0007"),
    lr_toy: z.number().describe("The learning rate this toy step uses: a toy-scale rate, usually 0.01–0.1, picked so one step visibly lowers the loss. `lr` stays the paper's value"),
    momentum: z.number().describe("Momentum coefficient (SGD with momentum), else 0"),
    beta1: z.number().describe("Adam β1 (0.9 if the paper doesn't say)"),
    beta2: z.number().describe("Adam β2 (0.999 if the paper doesn't say)"),
    eps: z.number().describe("Adam ε (1e-8 if the paper doesn't say)"),
    weight_decay: z.number().describe("Weight decay (AdamW), else 0"),
    schedule_note: z
      .string()
      .describe("One sentence on the paper's learning-rate schedule (e.g. warmup); it is explained, not simulated"),
    quote: z.string().describe("Verbatim sentence from the paper naming the optimizer, or ''"),
    fallback: z
      .boolean()
      .describe("true when the paper's optimizer isn't SGD, momentum, Adam or AdamW and the toy uses plain SGD"),
  }),
  intro: z.string().describe("One or two sentences: what one training step on this example will show"),
  steps: z
    .array(
      z.object({
        block_id: z.string().describe("A block that carries a node, in reverse data-flow order (output first)"),
        title: z.string().describe("Short step title"),
        narration: z.string().describe("2–3 sentences on the gradient arriving here and what it says"),
        learns: z.string().describe("One sentence on what this block's weights learn from this example, or that it has none"),
      }),
    )
    .describe("One step per block that has a node, from the loss back to the inputs. Never write gradients or losses"),
  outro: z.string().describe("One sentence to remember"),
});
export type TrainingArtifact = z.infer<typeof TrainingArtifact>;

const MAX_NODES = 32;
const MAX_SIDE = 8;
const MAX_VOCAB = 8;
const MAX_HEADS = 2;
const MIN_EPS = 1e-12;
/** A learning rate is kept when one step lowers the loss by at least this fraction. */
const MIN_DROP = 0.01;
const KINDS = new Set<string>(TOY_KINDS);

const clampMatrix = (m: number[][]) => m.slice(0, MAX_SIDE).map((r) => r.slice(0, MAX_SIDE));
const clamp = (x: number, lo: number, hi: number, fallback: number) => (Number.isFinite(x) ? Math.min(hi, Math.max(lo, x)) : fallback);

/** Every node sits on a block of the graph. */
export function trainingFits(a: TrainingArtifact, graph: Architecture): boolean {
  const ids = new Set(graph.blocks.map((b) => b.id));
  return a.toy.nodes.every((n) => ids.has(n.block_id));
}

function lossOrNull(a: TrainingArtifact): { before: number; after: number } | null {
  try {
    const t = train(a);
    return { before: t.lossBefore, after: t.lossAfter };
  } catch {
    return null;
  }
}

const drops = (l: { before: number; after: number } | null) => l !== null && l.before - l.after >= MIN_DROP * Math.abs(l.before);

/**
 * The gate before anyone sees a training artifact: clamps sizes, drops nodes off the graph, turns unknown kinds into
 * pass-throughs, maps the optimizer to one the engine runs and picks a toy learning rate that trains. Returns `null`
 * when the toy model can't run one finite step (cycle, dangling input, shape mismatch, bad target, non-finite numbers)
 * fewer than two graph blocks carry a node, or the model's output (the last node) is dropped or becomes a pass-through.
 */
export function cleanTraining(raw: TrainingArtifact, graph: Architecture): TrainingArtifact | null {
  const blockIds = new Set(graph.blocks.map((b) => b.id));

  // Structure: ids unique, every input names an earlier node (this also rules out cycles).
  const seen = new Set<string>();
  for (const n of raw.toy.nodes.slice(0, MAX_NODES)) {
    if (seen.has(n.id)) return null;
    for (const i of n.inputs) if (!seen.has(i)) return null;
    seen.add(n.id);
  }

  // Nodes off the graph are dropped; nothing is rewired, so a node that read one keeps only the inputs that remain
  // and becomes a pass-through (or goes too, when nothing remains).
  const kept = new Set<string>();
  const nodes: ToyNode[] = [];
  for (const n of raw.toy.nodes.slice(0, MAX_NODES)) {
    if (!blockIds.has(n.block_id)) continue;
    const inputs = n.inputs.filter((i) => kept.has(i));
    const lost = inputs.length < n.inputs.length;
    if (lost && inputs.length === 0) continue;
    const passthrough = lost || !KINDS.has(n.kind) || n.fn.trim().toLowerCase() === "passthrough";
    nodes.push(
      passthrough
        ? { id: n.id, block_id: n.block_id, kind: "dropout", fn: "passthrough", inputs: inputs.slice(0, 1), params: [], count: 1 }
        : {
            ...n,
            params: n.params.map((p) => ({ name: p.name, values: clampMatrix(p.values) })),
            count: n.kind === "attention" ? clamp(Math.round(n.count), 1, MAX_HEADS, 1) : 1,
          },
    );
    kept.add(n.id);
  }
  const carried = new Set(nodes.map((n) => n.block_id));
  if (carried.size < 2) return null;
  // The model's output is what the loss reads: it must still be the last node, and a real one.
  const output = nodes.at(-1);
  if (!output || output.id !== raw.toy.nodes.at(-1)?.id || output.fn === "passthrough") return null;

  // Target: exactly one of classes / values; class ids inside the vocabulary.
  const vocab = raw.vocab.slice(0, MAX_VOCAB);
  const classes = raw.target.classes;
  const values = clampMatrix(raw.target.values);
  if ((classes.length > 0) === (values.length > 0)) return null;
  if (classes.some((c) => !Number.isInteger(c) || c < 0 || c >= vocab.length)) return null;

  const o = raw.optimizer;
  const a: TrainingArtifact = {
    ...raw,
    toy: { ...raw.toy, nodes },
    vocab,
    target: { classes, values },
    loss: { ...raw.loss, label_smoothing: clamp(raw.loss.label_smoothing, 0, 0.5, 0) },
    optimizer: {
      ...o,
      lr_toy: Number.isFinite(o.lr_toy) && o.lr_toy > 0 ? o.lr_toy : 0.05,
      momentum: clamp(o.momentum, 0, 0.999, 0),
      beta1: clamp(o.beta1, 0, 0.999, 0.9),
      beta2: clamp(o.beta2, 0, 0.9999, 0.999),
      eps: clamp(o.eps, MIN_EPS, 1, 1e-8),
      weight_decay: clamp(o.weight_decay, 0, 1, 0),
      fallback: optimizerKind(o.kind).fallback,
    },
    steps: raw.steps.filter((s) => carried.has(s.block_id)),
  };

  // It must train: one finite step, and class ids inside the output.
  let trace;
  try {
    trace = train(a);
  } catch {
    return null;
  }
  const width = trace.predictionBefore[0]?.length ?? 0;
  if (classes.some((c) => c >= width)) return null;

  // The toy learning rate: keep it if the loss drops by 1%, else a tenth, else ten times; else keep it (the tour
  // reports the rise). The paper's `lr` never changes.
  if (!drops({ before: trace.lossBefore, after: trace.lossAfter })) {
    const base = a.optimizer.lr_toy;
    for (const lr_toy of [base * 0.1, base * 10]) {
      const tried = { ...a, optimizer: { ...a.optimizer, lr_toy } };
      if (drops(lossOrNull(tried))) return tried;
    }
  }
  return a;
}
