// One real training step on a toy model: forward, loss, backward, one optimizer update, forward again.
// Plain TypeScript, no DOM: the server validates with it and the browser shows its numbers.

import type { TrainingArtifact } from "../training";
import { lossOf } from "./loss";
import { TOY_KINDS, type ToyKind, runOp } from "./ops";
import { type OptimizerSpec, step } from "./optim";
import { Tensor, reshape, tensor, withTape } from "./tensor";

export interface NodeTrace {
  id: string;
  block_id: string;
  kind: ToyKind;
  /** Forward value before the update (3-D values flattened to rows). */
  value: number[][];
  /** ∂L/∂(this node's output): the gradient arriving from every consumer, summed. */
  gradOut: number[][];
  /** Per input, in order: the gradient this node sends back to that input. */
  gradIn: number[][][];
  params: { name: string; before: number[][]; grad: number[][]; after: number[][] }[];
  passthrough: boolean;
}

export interface Trace {
  /** Forward order. */
  nodes: NodeTrace[];
  lossBefore: number;
  lossAfter: number;
  predictionBefore: number[][];
  predictionAfter: number[][];
  lr: number;
  note: string;
}

type OptKind = OptimizerSpec["kind"];

/** The paper's optimizer name → one the engine runs (case-insensitive); anything else is SGD with `fallback`. */
export function optimizerKind(name: string): { kind: OptKind; fallback: boolean } {
  // Whole words only: RAdam, NAdam and Adamax are other optimizers, so they fall back.
  const n = name.toLowerCase();
  if (/\badamw\b/.test(n)) return { kind: "adamw", fallback: false };
  if (/\badam\b/.test(n)) return { kind: "adam", fallback: false };
  if (/momentum|nesterov/.test(n)) return { kind: "momentum", fallback: false };
  if (/\bsgd\b|gradient descent/.test(n)) return { kind: "sgd", fallback: false };
  return { kind: "sgd", fallback: true };
}

const unusable = (reason: string) => new Error(`unusable: ${reason}`);
const KINDS = new Set<string>(TOY_KINDS);

/** Rows of the last axis: 1-D is one row, 3-D [a, b, c] becomes a·b rows of c. */
function rows(shape: number[], data: ArrayLike<number> | null): number[][] {
  const width = shape.length === 0 ? 1 : shape[shape.length - 1]!;
  const size = shape.reduce((a, b) => a * b, 1);
  const out: number[][] = [];
  for (let i = 0; i < size; i += width) out.push(Array.from({ length: width }, (_, j) => (data ? data[i + j]! : 0)));
  return out;
}
const valueOf = (t: Tensor) => rows(t.shape, t.data);
const gradOfT = (t: Tensor) => rows(t.shape, t.grad);

function matrix(values: number[][], what: string, requiresGrad: boolean): Tensor {
  const width = values[0]?.length ?? 0;
  if (values.length === 0 || width === 0) throw unusable(`${what} is empty`);
  for (const r of values) {
    if (r.length !== width) throw unusable(`${what} is not rectangular`);
    for (const v of r) if (!Number.isFinite(v)) throw unusable(`${what} holds a non-finite number`);
  }
  return tensor(values, requiresGrad);
}

function finite(m: number[][], what: string) {
  for (const r of m) for (const v of r) if (!Number.isFinite(v)) throw unusable(`${what} is not finite`);
}

type Node = TrainingArtifact["toy"]["nodes"][number];
type Resolved = { node: Node; kind: ToyKind; fn: string; passthrough: boolean };

function resolve(nodes: Node[]): Resolved[] {
  if (nodes.length === 0) throw unusable("the toy model has no nodes");
  const seen = new Set<string>();
  return nodes.map((node) => {
    if (seen.has(node.id)) throw unusable(`node id "${node.id}" is used twice`);
    for (const i of node.inputs) if (!seen.has(i)) throw unusable(`node "${node.id}" reads "${i}", not an earlier node`);
    seen.add(node.id);
    const known = KINDS.has(node.kind);
    const passthrough = !known || node.fn.trim().toLowerCase() === "passthrough";
    return passthrough
      ? { node, kind: "dropout" as const, fn: "passthrough", passthrough }
      : { node, kind: node.kind as ToyKind, fn: node.fn, passthrough };
  });
}

type Run = { outs: Tensor[]; edges: Tensor[][]; params: Record<string, Tensor>[] };

/** Forward through every node. With `learn`, params need grad and each input edge gets its own copy, so the
 * gradient one node sends back to an input stays separate from what other consumers send. */
function forward(nodes: Resolved[], values: Record<string, number[][]>[], learn: boolean): Run {
  const index = new Map(nodes.map((r, i) => [r.node.id, i]));
  const run: Run = { outs: [], edges: [], params: [] };
  nodes.forEach((r, i) => {
    const params: Record<string, Tensor> = {};
    if (!r.passthrough)
      for (const [name, v] of Object.entries(values[i]!))
        params[name] = matrix(v, `${r.node.id}.${name}`, learn && r.kind !== "input");
    const inputs = r.node.inputs.map((id) => run.outs[index.get(id)!]!);
    const edges = learn ? inputs.map((t) => reshape(t, t.shape)) : inputs;
    let out: Tensor;
    try {
      out = runOp(r.kind, r.passthrough ? "" : r.fn, edges, params, r.node.count);
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      throw msg.startsWith("unusable:") ? e : unusable(`${r.node.id}: ${msg}`);
    }
    for (const v of out.data) if (!Number.isFinite(v)) throw unusable(`${r.node.id} produced a non-finite value`);
    run.outs.push(out);
    run.edges.push(edges);
    run.params.push(params);
  });
  return run;
}

/** Loss of the last node against the target, as a tensor on the current tape. */
function loss(a: TrainingArtifact, last: Resolved, out: Tensor): Tensor {
  const target = a.target.classes.length > 0 ? a.target.classes : a.target.values;
  if (target.length === 0) throw unusable("the target is empty");
  const fn = last.fn.trim().toLowerCase();
  const outputIsProbs = last.kind === "softmax" || (last.kind === "activation" && fn === "sigmoid");
  let l: Tensor;
  try {
    l = lossOf(a.loss.kind, out, target, { smoothing: a.loss.label_smoothing, outputIsProbs });
  } catch (e) {
    throw unusable(`loss: ${e instanceof Error ? e.message : String(e)}`);
  }
  if (!Number.isFinite(l.data[0]!)) throw unusable("the loss is not finite");
  return l;
}

/**
 * One training step on the artifact's toy model. Throws `Error("unusable: …")` on a shape mismatch, an input that
 * names no earlier node, or any non-finite number.
 */
export function train(a: TrainingArtifact): Trace {
  const nodes = resolve(a.toy.nodes);
  const last = nodes[nodes.length - 1]!;
  const start = nodes.map((r) => Object.fromEntries(r.node.params.map((p) => [p.name, p.values])));
  const spec: OptimizerSpec = {
    kind: optimizerKind(a.optimizer.kind).kind,
    lr: a.optimizer.lr_toy,
    momentum: a.optimizer.momentum,
    beta1: a.optimizer.beta1,
    beta2: a.optimizer.beta2,
    eps: a.optimizer.eps,
    weight_decay: a.optimizer.weight_decay,
  };

  let note = "";
  const { traces, after, lossBefore } = withTape((tape) => {
    const run = forward(nodes, start, true);
    const l = loss(a, last, run.outs[run.outs.length - 1]!);
    tape.backward(l);
    const after: Record<string, number[][]>[] = [];
    const traces: NodeTrace[] = nodes.map((r, i) => {
      const learned = Object.fromEntries(Object.entries(run.params[i]!).filter(([, t]) => t.requiresGrad));
      const stepped = step(spec, learned);
      if (stepped.note) note = stepped.note;
      after.push({ ...start[i]!, ...stepped.after });
      const t: NodeTrace = {
        id: r.node.id,
        block_id: r.node.block_id,
        kind: r.kind,
        value: valueOf(run.outs[i]!),
        gradOut: gradOfT(run.outs[i]!),
        gradIn: run.edges[i]!.map(gradOfT),
        params: Object.entries(learned).map(([name, p]) => ({
          name,
          before: valueOf(p),
          grad: gradOfT(p),
          after: stepped.after[name]!,
        })),
        passthrough: r.passthrough,
      };
      finite(t.gradOut, `${t.id}'s gradient`);
      t.gradIn.forEach((g) => finite(g, `${t.id}'s gradient`));
      for (const p of t.params) {
        finite(p.grad, `${t.id}.${p.name}'s gradient`);
        finite(p.after, `${t.id}.${p.name} after the step`);
      }
      return t;
    });
    return { traces, after, lossBefore: l.data[0]! };
  });

  const again = withTape(() => {
    const run = forward(nodes, after, false);
    const out = run.outs[run.outs.length - 1]!;
    return { loss: loss(a, last, out).data[0]!, prediction: valueOf(out) };
  });

  return {
    nodes: traces,
    lossBefore,
    lossAfter: again.loss,
    predictionBefore: traces[traces.length - 1]!.value,
    predictionAfter: again.prediction,
    lr: spec.lr,
    note,
  };
}
