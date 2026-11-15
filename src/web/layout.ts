import type { BlockType, Edge, Group } from "../shared/graph";

/**
 * Graph → positions. Data flows bottom-to-top like the figures in papers.
 * - Branches that only meet through "cross" edges (encoder / decoder) become columns.
 * - Rank = longest path from an input, ignoring cross edges, so the two stacks of a
 *   Transformer sit side by side at the same heights.
 * - Rows are shared across columns; blocks sharing a (column, rank) spread sideways.
 */

export interface LayoutInput {
  blocks: { id: string; type: BlockType; group: string }[];
  edges: Edge[];
  groups: Group[];
}

export interface Box {
  x: number;
  y: number; // center
  w: number;
  h: number;
  d: number;
}

export interface PlacedBlock extends Box {
  id: string;
  column: number;
  rank: number;
}

export interface PlacedGroup extends Box {
  id: string;
  label: string;
  repeat: number;
}

export interface Layout {
  blocks: Map<string, PlacedBlock>;
  groups: PlacedGroup[];
  columns: number;
  bounds: { minX: number; maxX: number; minY: number; maxY: number };
}

export const BLOCK_SIZE: Record<BlockType, [w: number, h: number, d: number]> = {
  input: [3.0, 0.45, 0.8],
  output: [3.0, 0.45, 0.8],
  embedding: [3.2, 0.7, 1.1],
  positional_encoding: [3.0, 0.55, 0.9],
  multi_head_attention: [3.4, 1.1, 1.5],
  cross_attention: [3.4, 1.1, 1.5],
  attention: [3.2, 1.0, 1.3],
  feed_forward: [3.2, 1.15, 1.2],
  add_norm: [3.2, 0.32, 1.0],
  layer_norm: [3.2, 0.32, 1.0],
  normalization: [3.2, 0.32, 1.0],
  residual_add: [1.0, 0.5, 1.0],
  linear: [3.0, 0.5, 1.0],
  activation: [2.6, 0.35, 0.9],
  softmax: [3.0, 0.75, 1.0],
  convolution: [3.0, 0.9, 1.4],
  pooling: [2.4, 0.7, 1.2],
  recurrent: [3.0, 1.0, 1.2],
  dropout: [2.6, 0.3, 0.9],
  concat: [3.2, 0.45, 1.0],
  other: [3.0, 0.7, 1.0],
};

const ROW_GAP = 0.95;
const COLUMN_GAP = 3.6;
const SIDE_GAP = 0.6;
const GROUP_PAD = 0.45;

export function layoutGraph(graph: LayoutInput): Layout {
  const ids = graph.blocks.map((b) => b.id);
  const known = new Set(ids);
  const edges = graph.edges.filter((e) => known.has(e.from) && known.has(e.to) && e.from !== e.to);
  const flow = edges.filter((e) => e.kind !== "cross");

  // Columns: weakly connected components of the non-cross graph, in block order.
  const parent = new Map(ids.map((id) => [id, id]));
  const find = (a: string): string => {
    while (parent.get(a) !== a) a = parent.get(a)!;
    return a;
  };
  for (const e of flow) parent.set(find(e.from), find(e.to));
  const columnOf = new Map<string, number>();
  const rootColumn = new Map<string, number>();
  for (const id of ids) {
    const root = find(id);
    if (!rootColumn.has(root)) rootColumn.set(root, rootColumn.size);
    columnOf.set(id, rootColumn.get(root)!);
  }
  const columns = Math.max(1, rootColumn.size);

  // Ranks: longest path over flow edges. Kahn's algorithm; anything left in a cycle
  // keeps the rank it had reached, so a malformed graph still renders.
  const rank = new Map(ids.map((id) => [id, 0]));
  const indeg = new Map(ids.map((id) => [id, 0]));
  const out = new Map<string, string[]>(ids.map((id) => [id, []]));
  for (const e of flow) {
    out.get(e.from)!.push(e.to);
    indeg.set(e.to, indeg.get(e.to)! + 1);
  }
  const queue = ids.filter((id) => indeg.get(id) === 0);
  while (queue.length) {
    const id = queue.shift()!;
    for (const next of out.get(id)!) {
      rank.set(next, Math.max(rank.get(next)!, rank.get(id)! + 1));
      indeg.set(next, indeg.get(next)! - 1);
      if (indeg.get(next) === 0) queue.push(next);
    }
  }

  // Row heights shared across columns.
  const typeOf = new Map(graph.blocks.map((b) => [b.id, b.type]));
  const maxRank = Math.max(0, ...rank.values());
  const rowH = new Array(maxRank + 1).fill(0);
  for (const id of ids) rowH[rank.get(id)!] = Math.max(rowH[rank.get(id)!], BLOCK_SIZE[typeOf.get(id)!][1]);
  const rowY: number[] = [];
  let y = 0;
  for (let r = 0; r <= maxRank; r++) {
    rowY.push(y + rowH[r] / 2);
    y += rowH[r] + ROW_GAP;
  }
  const totalH = y - ROW_GAP;

  // Column widths: widest row of side-by-side blocks in that column.
  const cells = new Map<string, string[]>();
  for (const id of ids) {
    const key = `${columnOf.get(id)}:${rank.get(id)}`;
    cells.set(key, [...(cells.get(key) ?? []), id]);
  }
  const rowWidth = (members: string[]) =>
    members.reduce((s, m) => s + BLOCK_SIZE[typeOf.get(m)!][0], 0) + SIDE_GAP * (members.length - 1);
  const colW = new Array(columns).fill(0);
  for (const [key, members] of cells) {
    const c = Number(key.split(":")[0]);
    colW[c] = Math.max(colW[c], rowWidth(members));
  }
  const totalW = colW.reduce((s, w) => s + w, 0) + COLUMN_GAP * (columns - 1);
  const colX: number[] = [];
  let x = -totalW / 2;
  for (let c = 0; c < columns; c++) {
    colX.push(x + colW[c] / 2);
    x += colW[c] + COLUMN_GAP;
  }

  const blocks = new Map<string, PlacedBlock>();
  for (const [key, members] of cells) {
    const [c, r] = key.split(":").map(Number) as [number, number];
    let cursor = colX[c]! - rowWidth(members) / 2;
    for (const id of members) {
      const [w, h, d] = BLOCK_SIZE[typeOf.get(id)!];
      blocks.set(id, { id, column: c, rank: r, x: cursor + w / 2, y: rowY[r]! - totalH / 2, w, h, d });
      cursor += w + SIDE_GAP;
    }
  }

  const groups: PlacedGroup[] = [];
  for (const g of graph.groups) {
    const members = graph.blocks.filter((b) => b.group === g.id).map((b) => blocks.get(b.id)!);
    if (!members.length) continue;
    const minX = Math.min(...members.map((m) => m.x - m.w / 2)) - GROUP_PAD;
    const maxX = Math.max(...members.map((m) => m.x + m.w / 2)) + GROUP_PAD;
    const minY = Math.min(...members.map((m) => m.y - m.h / 2)) - GROUP_PAD;
    const maxY = Math.max(...members.map((m) => m.y + m.h / 2)) + GROUP_PAD;
    const d = Math.max(...members.map((m) => m.d)) + GROUP_PAD * 2;
    groups.push({ id: g.id, label: g.label, repeat: g.repeat, x: (minX + maxX) / 2, y: (minY + maxY) / 2, w: maxX - minX, h: maxY - minY, d });
  }

  const all = [...blocks.values()];
  return {
    blocks,
    groups,
    columns,
    bounds: {
      minX: Math.min(...all.map((b) => b.x - b.w / 2), ...groups.map((g) => g.x - g.w / 2)),
      maxX: Math.max(...all.map((b) => b.x + b.w / 2), ...groups.map((g) => g.x + g.w / 2)),
      minY: Math.min(...all.map((b) => b.y - b.h / 2), ...groups.map((g) => g.y - g.h / 2)),
      maxY: Math.max(...all.map((b) => b.y + b.h / 2), ...groups.map((g) => g.y + g.h / 2)),
    },
  };
}
