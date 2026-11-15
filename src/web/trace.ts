import type { Architecture } from "../shared/graph";
import { escapeHtml } from "./scene";

/**
 * Shape tracing: pick one tensor dimension ("512", "n", "d_model") and find every block it flows
 * through. Hyperparameters make names and values the same dimension (d_model = 512), and pairs
 * whose product equals it explain how it splits (512 = h × d_k = 8 × 64).
 */

const TOKEN = /[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?/g;

/** The dimensions named in a shape string such as "[B, m, 512] + [B, n, 512]". */
export const dimTokens = (shape: string): string[] => shape.match(TOKEN) ?? [];

/** A shape with each dimension as a button that starts a trace. */
export function shapeHtml(shape: string, active: string | null): string {
  let html = "";
  let last = 0;
  for (const m of shape.matchAll(TOKEN)) {
    const tok = m[0];
    html += escapeHtml(shape.slice(last, m.index));
    html += `<button type="button" class="dim" data-dim="${escapeHtml(tok)}" aria-pressed="${tok === active}" title="Trace ${escapeHtml(tok)} through the model">${escapeHtml(tok)}</button>`;
    last = m.index + tok.length;
  }
  return html + escapeHtml(shape.slice(last));
}

type Hp = { name: string; value: string };

/** "d_k = d_v" → ["d_k", "d_v"]: some papers name one value twice. */
const namesOf = (hp: Hp) =>
  hp.name
    .split(/[=,]/)
    .map((n) => n.trim())
    .filter(Boolean);

function allHyperparameters(graph: Architecture): Hp[] {
  return [...graph.hyperparameters, ...graph.blocks.flatMap((b) => b.hyperparameters)];
}

/** Every spelling of a dimension: the token itself plus the names and value tied to it by hyperparameters. */
export function aliasesOf(graph: Architecture, token: string): Set<string> {
  const parent = new Map<string, string>();
  const find = (a: string): string => {
    if (!parent.has(a)) parent.set(a, a);
    while (parent.get(a) !== a) a = parent.get(a)!;
    return a;
  };
  for (const hp of allHyperparameters(graph)) {
    const value = hp.value.trim();
    // Only short values are dimensions ("512", "n"); "3x3" or "0.1" stay on their own.
    if (!/^[A-Za-z_]\w*$|^\d+$/.test(value)) continue;
    for (const name of namesOf(hp)) parent.set(find(name), find(value));
  }
  const root = find(token);
  return new Set([token, ...[...parent.keys()].filter((k) => find(k) === root)]);
}

export type Role = "through" | "created" | "consumed" | "param";

export interface Trace {
  token: string;
  aliases: string[]; // other spellings, e.g. ["d_model"] for "512"
  blocks: Map<string, Role>;
  /** How the dimension factors into named hyperparameters, e.g. "h × d_k = 8 × 64". */
  splits: string[];
}

export function traceDim(graph: Architecture, token: string): Trace {
  const names = aliasesOf(graph, token);
  const has = (shape: string) => dimTokens(shape).some((t) => names.has(t));
  const blocks = new Map<string, Role>();
  for (const b of graph.blocks) {
    const into = has(b.input_shape);
    const out = has(b.output_shape);
    if (into && out) blocks.set(b.id, "through");
    else if (out) blocks.set(b.id, "created");
    else if (into) blocks.set(b.id, "consumed");
    else if (b.hyperparameters.some((hp) => namesOf(hp).some((n) => names.has(n)) || names.has(hp.value.trim()))) {
      blocks.set(b.id, "param");
    }
  }

  const value = [...names].map(Number).find((n) => Number.isInteger(n) && n > 1);
  const splits: string[] = [];
  if (value) {
    // Named integer hyperparameters, one name per value (first wins), excluding the dimension itself.
    const factors = new Map<number, string>();
    for (const hp of allHyperparameters(graph)) {
      const v = Number(hp.value);
      const name = namesOf(hp)[0];
      if (name && Number.isInteger(v) && v > 1 && v < value && !names.has(name) && !factors.has(v)) factors.set(v, name);
    }
    for (const [a, an] of factors) {
      const b = value / a;
      const bn = factors.get(b);
      // Two different hyperparameters: 64 = 8 × 8 would only name h twice.
      if (bn && a < b && !splits.some((s) => s.includes(`${a} × ${b}`))) splits.push(`${an} × ${bn} = ${a} × ${b}`);
    }
  }
  return { token, aliases: [...names].filter((n) => n !== token), blocks, splits };
}

const ROWS: [Role, string][] = [
  ["created", "Appears in"],
  ["through", "Flows through"],
  ["consumed", "Used up by"],
  ["param", "Set in"],
];

/** The trace card: the dimension, its other names, where it appears and leaves, and how it splits. */
export function traceCardHtml(t: Trace, graph: Architecture): string {
  const label = new Map(graph.blocks.map((b) => [b.id, b.label]));
  const listed = (role: Role) => {
    // Repeated labels (six "Add & Norm") read better counted.
    const counts = new Map<string, number>();
    for (const [id, r] of t.blocks) if (r === role) counts.set(label.get(id) ?? id, (counts.get(label.get(id) ?? id) ?? 0) + 1);
    return [...counts].map(([l, n]) => escapeHtml(l) + (n > 1 ? ` ×${n}` : "")).join(", ");
  };
  const rows = ROWS.map(([role, name]) => [name, listed(role)] as const).filter(([, list]) => list);
  return `
    <p class="tr-head"><b>${escapeHtml(t.token)}</b>${t.aliases.length ? `<span>${t.aliases.map(escapeHtml).join(" = ")}</span>` : ""}
      <button type="button" data-untrace aria-label="Stop tracing">×</button></p>
    ${
      rows.length
        ? `<dl>${rows.map(([name, list]) => `<dt>${name}</dt><dd>${list}</dd>`).join("")}</dl>`
        : `<p>No block's shape mentions it.</p>`
    }
    ${t.splits.map((s) => `<p class="tr-split">${escapeHtml(t.token)} = ${escapeHtml(s)}</p>`).join("")}`;
}
