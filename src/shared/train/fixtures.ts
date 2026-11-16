// The hand-written toy Transformer (1706.03762) used by the training tests: block ids from
// samples/1706.03762.json, the journey's example "the cat sat" → "die Katze saß", width 4, 2 heads, feed-forward 8, one layer each.

import type { TrainingArtifact } from "../training";

/** Deterministic starting weights: mulberry32, uniform in [−0.5, 0.5], two decimals. */
function weights(seed: number) {
  let s = seed >>> 0;
  const next = () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  return (r: number, c: number): number[][] =>
    Array.from({ length: r }, () => Array.from({ length: c }, () => Math.round((next() - 0.5) * 100) / 100));
}

const D = 4;
const VOCAB = ["<s>", "the", "cat", "sat", "die", "Katze", "saß"];
const w = weights(1706);
const ones = [Array(D).fill(1)];
const zeros = [Array(D).fill(0)];

type Node = TrainingArtifact["toy"]["nodes"][number];
const node = (id: string, block_id: string, kind: string, inputs: string[], params: Node["params"] = [], fn = "", count = 1): Node => ({
  id, block_id, kind, fn, inputs, params, count,
});
const attention = (id: string, block_id: string, fn: string, inputs: string[]) =>
  node(id, block_id, "attention", inputs, ["Wq", "Wk", "Wv", "Wo"].map((name) => ({ name, values: w(D, D) })), fn, 2);
const norm = (id: string, block_id: string, input: string) =>
  node(id, block_id, "normalization", [input], [{ name: "gamma", values: ones }, { name: "beta", values: zeros }], "layer");
const HIDDEN = 8;
/** FFN(x) = max(0, xW1 + b1)W2 + b2: three nodes sharing one block. */
const ffn = (prefix: string, block_id: string, input: string): Node[] => [
  node(`${prefix}1`, block_id, "linear", [input], [{ name: "W", values: w(D, HIDDEN) }, { name: "b", values: w(1, HIDDEN) }]),
  node(`${prefix}_relu`, block_id, "activation", [`${prefix}1`], [], "relu"),
  node(`${prefix}2`, block_id, "linear", [`${prefix}_relu`], [{ name: "W", values: w(HIDDEN, D) }, { name: "b", values: w(1, D) }]),
];

export const TRANSFORMER_TOY: TrainingArtifact = {
  toy: {
    dims:
      "d_model 512 → 4, 8 heads → 2, 6 layers → 1 encoder and 1 decoder layer, feed-forward 2048 → 8, vocabulary 37 000 → 7. Unlike the paper, the toy doesn't tie the embedding and output weights: each has its own.",
    nodes: [
      node("src", "inputs", "input", [], [{ name: "values", values: [[1, 2, 3]] }]),
      node("src_embed", "input_embedding", "embedding_lookup", ["src"], [{ name: "E", values: w(VOCAB.length, D) }]),
      node("enc_pos", "encoder_positional", "positional_encoding", ["src_embed"]),
      attention("enc_attn", "encoder_self_attention", "self", ["enc_pos"]),
      node("enc_add1", "encoder_add_norm_1", "residual_add", ["enc_attn", "enc_pos"]),
      norm("enc_norm1", "encoder_add_norm_1", "enc_add1"),
      ...ffn("enc_ffn", "encoder_ffn", "enc_norm1"),
      node("enc_add2", "encoder_add_norm_2", "residual_add", ["enc_ffn2", "enc_norm1"]),
      norm("enc_norm2", "encoder_add_norm_2", "enc_add2"),
      node("tgt", "outputs", "input", [], [{ name: "values", values: [[0, 4, 5]] }]),
      node("tgt_embed", "output_embedding", "embedding_lookup", ["tgt"], [{ name: "E", values: w(VOCAB.length, D) }]),
      node("dec_pos", "decoder_positional", "positional_encoding", ["tgt_embed"]),
      attention("dec_self", "decoder_masked_attention", "causal", ["dec_pos"]),
      node("dec_add1", "decoder_add_norm_1", "residual_add", ["dec_self", "dec_pos"]),
      norm("dec_norm1", "decoder_add_norm_1", "dec_add1"),
      attention("dec_cross", "decoder_cross_attention", "cross", ["dec_norm1", "enc_norm2"]),
      node("dec_add2", "decoder_add_norm_2", "residual_add", ["dec_norm1", "dec_cross"]),
      norm("dec_norm2", "decoder_add_norm_2", "dec_add2"),
      ...ffn("dec_ffn", "decoder_ffn", "dec_norm2"),
      node("dec_add3", "decoder_add_norm_3", "residual_add", ["dec_ffn2", "dec_norm2"]),
      norm("dec_norm3", "decoder_add_norm_3", "dec_add3"),
      node("final_linear", "final_linear", "linear", ["dec_norm3"], [
        { name: "W", values: w(D, VOCAB.length) },
        { name: "b", values: w(1, VOCAB.length) },
      ]),
      node("softmax", "softmax", "softmax", ["final_linear"]),
    ],
  },
  vocab: VOCAB,
  target: { classes: [4, 5, 6], values: [] },
  loss: {
    kind: "cross_entropy",
    label_smoothing: 0.1,
    quote:
      "During training, we employed label smoothing of value ϵls = 0.1 [36]. This hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score.",
    caption: "How surprised the model is by the right next word at each position, with 10% of the belief spread over the others.",
  },
  optimizer: {
    kind: "Adam",
    lr: 0.05,
    lr_toy: 0.05,
    momentum: 0,
    beta1: 0.9,
    beta2: 0.98,
    eps: 1e-9,
    weight_decay: 0,
    schedule_note:
      "The paper warms the learning rate up linearly for 4000 steps, then decays it with the inverse square root of the step; the toy takes a single step at one fixed rate.",
    quote: "We used the Adam optimizer [20] with β1 = 0.9, β2 = 0.98 and ϵ = 10−9.",
    fallback: false,
  },
  intro:
    "The model has just guessed the next words for “<s> die Katze”. Now it learns from that one example: measure the error, send it back through every block, nudge every weight once, and guess again.",
  steps: [
    { block_id: "softmax", title: "Compare with the answer", narration: "The probabilities meet the right next words “die Katze saß”. The gradient here is simply the probabilities minus the smoothed target: push the right word up, every other word down.", learns: "Nothing: softmax has no weights." },
    { block_id: "final_linear", title: "Blame the scores", narration: "The error reaches the final projection, which turned each decoder vector into one score per word. Its gradient says which scores were too high and which too low.", learns: "Which directions of the decoder's vectors should vote for which word." },
    { block_id: "decoder_add_norm_3", title: "Through the last add & norm", narration: "The gradient passes the layer norm and splits at the residual: one copy goes into the feed-forward layer, one skips straight past it.", learns: "How much to scale and shift each feature after the last sublayer." },
    { block_id: "decoder_ffn", title: "Feed-forward, per position", narration: "Each position's error flows back through the feed-forward weights, which act on every position alike.", learns: "A per-position transformation that makes the next word easier to read off." },
    { block_id: "decoder_add_norm_2", title: "Split at the residual again", narration: "Again the gradient reaches both the cross-attention output and its skip connection, unchanged in each.", learns: "The scale and shift after cross-attention." },
    { block_id: "decoder_cross_attention", title: "Across to the encoder", narration: "Here the error splits in two: part goes back into the decoder's queries, part crosses over into the encoder's keys and values. This is how the German side teaches the English side.", learns: "Which source words each target position should look at, and what to take from them." },
    { block_id: "decoder_add_norm_1", title: "Under the masked attention", narration: "The gradient passes the first add & norm of the decoder, through the residual and the normalization.", learns: "The scale and shift after masked self-attention." },
    { block_id: "decoder_masked_attention", title: "Looking only backward", narration: "The mask kept each position from seeing the future, so no gradient flows to later positions' keys: only what a position was allowed to see gets blamed.", learns: "How earlier target words should inform each prediction." },
    { block_id: "decoder_positional", title: "Positions add, unchanged", narration: "The sinusoids are fixed; the gradient passes straight through to the embeddings.", learns: "Nothing: the sinusoidal encodings are fixed." },
    { block_id: "output_embedding", title: "Target word vectors", narration: "Only the rows for “<s>”, “die” and “Katze” receive a gradient: the words that were actually fed in.", learns: "Better starting vectors for the target words in this sentence." },
    { block_id: "outputs", title: "The shifted target", narration: "The token ids are data, not weights: the gradient stops here.", learns: "Nothing: these are the example's token ids." },
    { block_id: "encoder_add_norm_2", title: "Into the encoder", narration: "Everything the encoder receives arrived through cross-attention. It enters at the encoder's last add & norm.", learns: "The scale and shift of the encoder's output." },
    { block_id: "encoder_ffn", title: "Encoder feed-forward", narration: "The gradient flows back through the encoder's per-position transformation.", learns: "How to reshape each source word's vector so the decoder can use it." },
    { block_id: "encoder_add_norm_1", title: "Residual and norm", narration: "At the residual the gradient arrives twice, identical: once for the attention output and once for the skip.", learns: "The scale and shift after self-attention." },
    { block_id: "encoder_self_attention", title: "Source words attending to each other", narration: "The error tells each head how to reweight the source words so the decoder gets a more useful summary.", learns: "Which English words should look at which." },
    { block_id: "encoder_positional", title: "Positions add, unchanged", narration: "The fixed sinusoids pass the gradient through untouched.", learns: "Nothing: the sinusoidal encodings are fixed." },
    { block_id: "input_embedding", title: "Source word vectors", narration: "The rows for “the”, “cat” and “sat” receive their share of the error.", learns: "Better starting vectors for the source words in this sentence." },
    { block_id: "inputs", title: "The source sentence", narration: "The token ids are data: nothing to update. Now Adam moves every weight once, and the example runs forward again.", learns: "Nothing: these are the example's token ids." },
  ],
  outro: "One example, one step: the error flows backward through every block, and every weight moves a little so the right words become more likely.",
};
