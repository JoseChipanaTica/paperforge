/**
 * Accepts arxiv.org/abs/<id>, arxiv.org/pdf/<id>[.pdf], alphaxiv/ar5iv mirrors, "arXiv:<id>",
 * or a bare id. Returns the id without version suffix, or null.
 */
export function parseArxivId(input: string): string | null {
  const s = input.trim();
  // New-style ids: 1706.03762, 2401.12345v2
  const modern = s.match(/(?:^|[/:\s])(\d{4}\.\d{4,5})(v\d+)?(?:\.pdf)?(?:[?#/]|$)/i) ?? s.match(/^(\d{4}\.\d{4,5})(v\d+)?$/);
  if (modern) return modern[1]!;
  // Old-style ids: cs/0112017, hep-th/9901001v2
  const legacy = s.match(/([a-z-]+(?:\.[A-Z]{2})?\/\d{7})(v\d+)?/i);
  if (legacy) return legacy[1]!;
  return null;
}

export async function fetchArxivPdf(id: string): Promise<Uint8Array> {
  const res = await fetch(`https://arxiv.org/pdf/${id}`, {
    headers: { "User-Agent": "paperforge/0.1 (+https://github.com/)" },
    redirect: "follow",
  });
  if (!res.ok) throw new Error(`arXiv returned ${res.status} for ${id}`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  // "%PDF"
  if (bytes[0] !== 0x25 || bytes[1] !== 0x50 || bytes[2] !== 0x44 || bytes[3] !== 0x46) {
    throw new Error(`arXiv did not return a PDF for ${id}`);
  }
  return bytes;
}
