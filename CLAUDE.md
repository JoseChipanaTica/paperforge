# Paperforge

Paste an arXiv deep-learning paper; Claude reads it and the browser builds the
architecture as a 3D model that assembles itself block by block. Brief: `IDEA.md`.
Plan: `PLAN.md`.

**North star:** the "Attention Is All You Need" (1706.03762) demo. Every change should
make that 30-second moment look better or work more reliably. Polish what the camera sees.

## Layout

- `index.ts` — local `Bun.serve()` entry: `src/web/index.html` (HTML import) + the API routes
- `server.ts` — Vercel entry (Bun preset, `.vercelignore` drops `index.ts`): the API routes + the page
  built into `public/` by `bun run build` (Vercel doesn't support HTML imports)
- `src/server/routes.ts` — every `/api` route: `POST /api/forge` (visitor's key in `x-anthropic-key`),
  `GET|POST /api/papers`, `GET /api/graph?arxiv_id=` (the shared graph, gallery then sample, so the browser can
  check its stored copy against it), learn/ask/inside/translate
- `src/shared/graph.ts` — Zod schema for the block graph. **The contract** between the
  LLM, the server and the front end. Change it here and nowhere else.
  Changing a required field makes graphs stored in visitors' browsers fail validation, so they get re-forged.
- `src/server/arxiv.ts` — parse arXiv ids/URLs, download PDFs
- `src/server/pdf.ts` — pdf.js (`pdfjs-dist/legacy`) text extraction, one string per page
- `src/shared/models.ts` — the Claude models a visitor can pick (Opus 5.5 default, Fable 5.1, Sonnet 5.5, Haiku 5.5);
  the browser sends the pick as `x-anthropic-model` (`web/model.ts`), the server accepts only ids from this list.
  Only `shares` models (Opus, Fable) save for everyone: gallery, shared disk cache, publish tokens (`sharesResults()`)
- `src/server/model.ts` — `currentModel()`: the request's pick (set by `withModel` in `routes.ts`, an `AsyncLocalStorage`),
  else `PAPERFORGE_MODEL` or the default; and the SDK timeout under the function limit
- `src/server/extract.ts` — Claude call (structured output via `betaZodOutputFormat`)
- `src/shared/locate.ts` — find each block's verbatim `source_quote` in the page text
- `src/server/forge.ts` — pipeline; lookup order gallery → disk cache (`data/cache/`, `/tmp/paperforge`
  on Vercel) → `samples/` → Claude
- `src/server/store.ts` — the public gallery (`Bun.SQL`): Neon Postgres when `DATABASE_URL` is set,
  `data/paperforge.db` SQLite otherwise. One row per arXiv id, first save wins. Never stores paper text.
- `src/server/token.ts` — HMAC save tokens: only graphs this server forged can be published
- `src/server/api.ts` — `/api/papers` handlers and error → `{message, code}` mapping
- `src/shared/learn.ts` — Zod schemas for learning artifacts (PyTorch and the journey: one concrete input
  followed through every block, each step with typed `View`s of the data — tokens, vectors, heads, matrix,
  patches, signal, frames, probs, tensor) and the `/api/learn` + `/api/ask` stream events. Every step carries
  a graph `block_id`; `cleanJourney` clamps views so the front end can always draw them.
- `src/shared/motion.ts` — step motion: per walkthrough step, a chain of 1–3 ops (convolution, attention, linear,
  normalization… 23 kinds, or a Claude-written `custom` SVG) with small inputs only; the browser recomputes every
  number. `cleanMotion` clamps to 8×8 and turns ops it can't compute into captioned placeholders.
- `src/shared/svg.ts` — `sanitizeSvg`: rebuilds Claude's SVG from allowlisted elements/attributes (no DOMParser, so
  server and browser run the same code). The only way generated markup reaches the page.
- `src/shared/train/` — a tiny deterministic trainer: `tensor.ts` (tensors + reverse-mode tape), `gradcheck.ts`,
  `ops.ts` (toy ops), `loss.ts`, `optim.ts`, `run.ts` (`train(a) → Trace`), `fixtures.ts` (TRANSFORMER_TOY, RAISING_TOY).
  The browser and tests both run it; Claude never supplies numbers.
- `src/shared/training.ts` — `TrainingArtifact` schema (the toy model for the walkthrough's example); `cleanTraining`
  clamps and validates, retrying `lr_toy` ×0.1/×10 when the loss drop is < 1%; `trainingFits`, `optimizerKind`.
- `src/shared/palette.ts` — the ink colours, shared by the 3D scene, the motion figures and the tutor's SVG rules
- `src/server/tutor.ts` — Claude calls for learning features. All share one system prompt and one
  `[paper, graph]` prefix with a cache breakpoint; keep it byte-identical and effort fixed. The output
  format is part of the cache: learning kinds share `ANY_ARTIFACT`, inside views `INSIDE_ARTIFACT` (the
  journey and the block graph together exceed the API's grammar size limit), motion `MOTION_ARTIFACT` and training `TRAINING_ARTIFACT` (both with the
  walkthrough after the breakpoint).
- `src/shared/translate.ts` — languages, reading levels (Beginner / Engineer / Researcher), and the prose
  of each graph/artifact (`pickText` / `applyText`). Claude always writes English for an engineer; a
  "translation" rewrites only labels, descriptions and narration for another language and/or level, never
  ids, equations, shapes or source quotes. `codeSkeleton` guards PyTorch: only comments may change.
- `src/server/translate.ts` + `translate-api.ts` — `POST /api/translate`, keyed by a hash of the English
  text + language + level: gallery (`translations` table; only published/sample text) → disk cache →
  Claude. Never rewrites the same text twice. Ask gets language and level after the cache breakpoint.
- `src/server/learn-api.ts` — `POST /api/learn` (gallery → sample → Claude; saved only for published
  papers, first save wins), `POST /api/ask` (streamed, never stored), `GET /api/learn?arxiv_id=`,
  `POST /api/inside` (a block's inside view: an `Architecture` of its own, same lookup order, `insides`
  table; the tutor writes it through the shared output format so it reads the paper from the cache),
  `POST /api/inside-learn` (a block's own walkthrough and its motion, continuing the model's example from what
  enters the block to what leaves it; `inside_learn` table; saved only when built from shared inputs)
- `src/web/` — Three.js front end ("Paper & ink": spec `docs/superpowers/specs/2026-11-15-ui-overhaul-design.md`):
  - `scene.ts` (printed model: matte blocks, ink outline/shadow/halo shells from `blocks.ts`; labels, pulse),
    `layout.ts` (graph → positions), `timeline.ts` (assembly choreography), `paper.ts` (paper + highlights)
  - `main.ts` wires the modules: `landing.ts` (hero, live preview, how-it-works, gallery), `toolbar.ts`,
    `studyPanel.ts` (ARIA tabs; state in `panelState.ts`), `sheet.ts` (bottom sheet < 900 px),
    `overview.ts` (summary, block detail, Ask), `stepsView.ts` + `tour.ts` (the walkthrough, i.e. the journey),
    `specimenLayout.ts` (a `View` → cells, pure) + `specimen.ts` (the data as a printed card beside the current
    block; between steps it flies along the wire as a packet), `motion/` (step motion: `compute.ts` the math per op,
    `templates.ts` op → frames, `player.ts` tweens SVG items by id, `figure.ts` the figure under each step),
    `trainingTour.ts` (trace → `TourScript`, `traceViews`, `trainingHeat`: "Train on this example →" on the last
    step walks the model backward with vermilion heat via `scene.setHeat`; prefetch only when the kinds list already
    has training, never an unasked Claude call; `specimenLayout` shows small numbers as compact exponents),
    the browser store (what a visitor opens again costs nothing): `localStore.ts` (`LocalStore`: papers by arXiv id,
    artifacts by [graph hash, scope, kind], translations by [text hash, language, level]; Zod-checked reads,
    quota eviction of whole papers, `usage()`/`clear()` and `usageLabel` for the settings-card line),
    `storeBackend.ts` (the `Backend` interface, memory and null backends), `idbBackend.ts` (IndexedDB; a failed or
    hung open falls back to null), `paperOpen.ts` (open plan: stored paper first, shared graph still wins; `withTimeout`),
    `stored.ts` (`withStored`: a stored value is used with no request, a miss fetches then stores;
    `withStoredArtifact`: artifacts keep their origin, a stored gallery/sample copy is refreshed from the server first),
    `codeView.ts` (PyTorch), `learn.ts` (learn/ask/inside clients, request bookkeeping), `lang.ts`
    (language + level pickers, translate client), `trace.ts` (shape tracing: dimensions, aliases via
    hyperparameters, splits like 512 = h × d_k). `main.ts` keeps `graph` (English, signed, sent to the
    API), `view` (rewritten for the reader) and `inside` (a block's inside view, when open).
  - `styles/tokens.css` holds every colour, font and spacing token; `blocks.ts` mirrors ink /
    highlighter / vermilion and owns the block palette. One CSS file per component.
- `samples/` — committed block graphs (`<id>.json`), learning artifacts (`<id>.<kind>.json`, kind `pytorch` or
  `journey` or `motion` or `training`; `bun run learn <url> <kind> --write` regenerates one, motion and training from the committed journey) and inside
  views (`<id>.inside.<block_id>.json`, with `<id>.inside.<block_id>.{journey,motion}.json` for their walkthrough;
  `bun run learn <url> <journey|motion> --inside <block_id> --write`) that seed the cache, so the demo runs without an API key

## Conventions

- Bun everywhere: `bun`, `bun test`, `bun install`, `Bun.file`, `Bun.serve` routes, HTML imports.
  No express/vite/dotenv (Bun loads `.env`).
- Bring your own key: the browser keeps the visitor's Anthropic key in `localStorage`; the server
  uses it per request and never logs or stores it. `ANTHROPIC_API_KEY` is a local-only fallback.
- Claude model: `claude-opus-5-5` by default via `@anthropic-ai/sdk`; visitors can pick another from `MODELS`, so
  never hard-code the id in a request: use `currentModel()`. Thinking is always on for this
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
- `bun run learn <arxiv-url> <kind>` (`pytorch|journey|motion|training`) — generate one learning artifact, print token usage (cache check)
- `bunx tsc --noEmit` — type-check
