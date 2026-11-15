# Paperforge

Paste an arXiv deep-learning paper; Claude reads it and the browser builds the
architecture as a 3D model that assembles itself block by block. Brief: `IDEA.md`.
Plan: `PLAN.md`.

**North star:** the "Attention Is All You Need" (1706.03762) demo. Every change should
make that 30-second moment look better or work more reliably. Polish what the camera sees.

## Layout

- `index.ts` — `Bun.serve()` entry (also the Vercel function): serves `src/web/index.html`,
  `POST /api/forge` (visitor's key in `x-anthropic-key`), `GET|POST /api/papers`
- `src/shared/graph.ts` — Zod schema for the block graph. **The contract** between the
  LLM, the server and the front end. Change it here and nowhere else.
- `src/server/arxiv.ts` — parse arXiv ids/URLs, download PDFs
- `src/server/pdf.ts` — pdf.js (`pdfjs-dist/legacy`) text extraction, one string per page
- `src/server/extract.ts` — Claude call (structured output via `betaZodOutputFormat`)
- `src/server/locate.ts` — find each block's verbatim `source_quote` in the page text
- `src/server/forge.ts` — pipeline; lookup order gallery → disk cache (`data/cache/`, `/tmp/paperforge`
  on Vercel) → `samples/` → Claude
- `src/server/store.ts` — the public gallery (`Bun.SQL`): Neon Postgres when `DATABASE_URL` is set,
  `data/paperforge.db` SQLite otherwise. One row per arXiv id, first save wins. Never stores paper text.
- `src/server/token.ts` — HMAC save tokens: only graphs this server forged can be published
- `src/server/api.ts` — `/api/papers` handlers and error → `{message, code}` mapping
- `src/web/` — Three.js front end: `layout.ts` (graph → positions), `scene.ts`
  (meshes, labels, pulse), `timeline.ts` (the assembly choreography), `paper.ts` (page flip + highlights), `gallery.ts` (published-paper cards)
- `samples/` — committed block graphs that seed the cache, so the demo runs without an API key

## Conventions

- Bun everywhere: `bun`, `bun test`, `bun install`, `Bun.file`, `Bun.serve` routes, HTML imports.
  No express/vite/dotenv (Bun loads `.env`).
- Bring your own key: the browser keeps the visitor's Anthropic key in `localStorage`; the server
  uses it per request and never logs or stores it. `ANTHROPIC_API_KEY` is a local-only fallback.
- Claude model: `claude-opus-5-5` via `@anthropic-ai/sdk`. Thinking is always on for this
  model; tune `output_config.effort`, never send `thinking: {type: "disabled"}` or `budget_tokens`.
  No forced `tool_choice`, no assistant prefill — use structured outputs.
- `/api/forge` streams NDJSON events (`status`, `paper`, `graph`, `error`). Keep events
  small and typed (`ForgeEvent` in `src/shared/graph.ts`).
- Source quotes must be verbatim from the paper; the locator is what lets hover show the
  real sentence. If a quote can't be located, keep the block but mark `page: null`.
- Front end is plain TypeScript + Three.js (no React). Labels are DOM via `CSS2DRenderer`,
  equations via KaTeX.

## Commands

- `bun run dev` — hot-reloading server on http://localhost:3000
- `bun test` — unit tests (arXiv parsing, quote locator, layout, schema/sample validity)
- `bun run forge <arxiv-url> [--refresh]` — run the pipeline from the CLI, write cache
- `bunx tsc --noEmit` — type-check
