import type { SourceSpan } from "../shared/graph";

/**
 * Text extracted by pdf.js has odd spacing ("d model = 512 ."), line-break hyphens
 * ("position-\nwise") and ligatures. We compare on a folded form that keeps only
 * letters and digits, and map matches back to offsets in the original page text.
 */
function fold(text: string): { folded: string; map: number[] } {
  let folded = "";
  const map: number[] = [];
  for (let i = 0; i < text.length; i++) {
    // Per-char NFKC so ligatures ("ﬁ" → "fi") still map back to one source offset.
    for (const ch of text[i]!.normalize("NFKC").toLowerCase()) {
      if (/[\p{L}\p{N}]/u.test(ch)) {
        folded += ch;
        map.push(i);
      }
    }
  }
  return { folded, map };
}

const MIN_ANCHOR = 24;

export function locateQuote(pages: string[], quote: string): SourceSpan | null {
  const q = fold(quote).folded;
  if (q.length < 8) return null;
  const folded = pages.map(fold);

  // 1. Exact folded match.
  for (let p = 0; p < folded.length; p++) {
    const idx = folded[p]!.folded.indexOf(q);
    if (idx >= 0) return span(folded[p]!.map, p, idx, idx + q.length);
  }

  // 2. Anchor on the quote's head and tail (handles a garbled middle, e.g. an equation).
  const anchor = Math.min(40, Math.floor(q.length / 3));
  if (anchor >= MIN_ANCHOR) {
    const head = q.slice(0, anchor);
    const tail = q.slice(-anchor);
    for (let p = 0; p < folded.length; p++) {
      const f = folded[p]!.folded;
      const h = f.indexOf(head);
      if (h < 0) continue;
      const t = f.indexOf(tail, h);
      if (t >= 0 && t - h < q.length * 1.5) return span(folded[p]!.map, p, h, t + tail.length);
      return span(folded[p]!.map, p, h, Math.min(f.length, h + q.length));
    }
    for (let p = 0; p < folded.length; p++) {
      const f = folded[p]!.folded;
      const t = f.indexOf(tail);
      if (t >= 0) return span(folded[p]!.map, p, Math.max(0, t + tail.length - q.length), t + tail.length);
    }
  }
  return null;
}

function span(map: number[], page: number, from: number, to: number): SourceSpan {
  return { page, start: map[from]!, end: map[to - 1]! + 1 };
}
