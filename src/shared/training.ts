import { z } from "zod";

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
