# Paperforge — build plan

North star: paste `arxiv.org/abs/1706.03762` → paper page flips → highlights sweep the
method section → blocks fly out of the text → snap into a 3D encoder-decoder with
equation + tensor-shape labels → a data pulse runs through it.

## Pipeline

```
arXiv URL ─▶ /api/forge (NDJSON stream)
              1. resolve id, fetch PDF          (cache: data/cache/<id>.pdf)
              2. pdf.js text per page            → "paper" event (pages for the flip)
              3. Claude (claude-opus-5-5, structured output, Zod schema)
                 → typed block graph             (cache: data/cache/<id>.json)
              4. locate each block's source quote in the page text → page + char span
              5. "graph" event
browser   ─▶ paper card flips, sweeps highlights, blocks spawn at highlight
              positions, fly to exploded layout, snap together, pulse runs.
```

## Steps

1. **v0, demo-able** (this session)
   - `src/server/` arXiv fetch, pdf.js extraction, Claude extraction, quote locator, disk cache
   - `src/shared/graph.ts` Zod schema = the contract between server, LLM, and front end
   - `src/web/` Three.js scene: layered layout, assembly timeline, hover tooltip, data pulse
   - `samples/1706.03762.json` so the north-star demo runs with no API key
2. **Polish what the camera sees**
   - Block silhouettes per type, labels (KaTeX), camera choreography, pulse glow, easing
   - Paper flip + highlight sweep timing synced to block launches
3. **Later**: share links, more block types, GIF export, multi-paper gallery
