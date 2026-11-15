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
- `src/shared/learn.ts` — Zod schemas for learning artifacts (PyTorch, walkthrough, example) and
  the `/api/learn` + `/api/ask` stream events. Every step carries a graph `block_id`.
- `src/server/tutor.ts` — Claude calls for learning features. All share one system prompt and one
  `[paper, graph]` prefix with a cache breakpoint; keep it byte-identical and effort fixed.
- `src/server/learn-api.ts` — `POST /api/learn` (gallery → sample → Claude; saved only for published
  papers, first save wins), `POST /api/ask` (streamed, never stored), `GET /api/learn?arxiv_id=`
- `src/web/` — Three.js front end ("Paper & ink": spec `docs/superpowers/specs/2026-11-15-ui-overhaul-design.md`):
  - `scene.ts` (printed model: matte blocks, ink outline/shadow/halo shells from `blocks.ts`; labels, pulse),
    `layout.ts` (graph → positions), `timeline.ts` (assembly choreography), `paper.ts` (paper + highlights)
  - `main.ts` wires the modules: `landing.ts` (hero, live preview, how-it-works, gallery), `toolbar.ts`,
    `studyPanel.ts` (ARIA tabs; state in `panelState.ts`), `sheet.ts` (bottom sheet < 900 px),
    `overview.ts` (summary, block detail, Ask), `stepsView.ts` + `tour.ts` (walkthrough / example),
    `codeView.ts` (PyTorch), `learn.ts` (learn/ask clients, request bookkeeping)
  - `styles/tokens.css` holds every colour, font and spacing token; `blocks.ts` mirrors ink /
    highlighter / vermilion and owns the block palette. One CSS file per component.
- `samples/` — committed block graphs (`<id>.json`) and learning artifacts (`<id>.<kind>.json`) that
  seed the cache, so the demo runs without an API key

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
  equations via KaTeX. Colours only from `styles/tokens.css` / `TYPE_COLOR`; the hard ink offset
  shadow is reserved for "printed" things (blocks, paste box, start cards). Respect
  `prefers-reduced-motion`.

## Commands

- `bun run dev` — hot-reloading server on http://localhost:3000
- `bun test` — unit tests (arXiv parsing, quote locator, layout, schema/sample validity)
- `bun run forge <arxiv-url> [--refresh]` — run the pipeline from the CLI, write cache
- `bun run learn <arxiv-url> <kind>` — generate one learning artifact, print token usage (cache check)
- `bunx tsc --noEmit` — type-check
