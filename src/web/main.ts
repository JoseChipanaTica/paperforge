import "katex/dist/katex.min.css";
import type { Architecture, ForgeEvent, GraphSource, LocatedArchitecture } from "../shared/graph";
import { LEARN_KINDS, blockEnds, type AnyArtifact, type JourneyArtifact, type LearnKind, type PytorchArtifact } from "../shared/learn";
import { motionFits, type MotionArtifact } from "../shared/motion";
import { train, type Trace } from "../shared/train/run";
import type { TrainingArtifact } from "../shared/training";
import { FigureSlot, StepFigure, figureState, type FigureStatus } from "./motion/figure";
import { locateQuote } from "../shared/locate";
import { parseArxivId } from "../server/arxiv";
import { MODELS, modelName, type ModelId } from "../shared/models";
import { isOriginal, type TranslateKind } from "../shared/translate";
import { Run } from "./anim";
import { initLanding } from "./landing";
import { initReadingPickers, readingMessage, storedReading, translate, type Reading, type TranslationStore } from "./lang";
import { chooseModel, chosenModel, claudeHeaders } from "./model";
import { initToolbar } from "./toolbar";
import {
  InFlight,
  LearnError,
  askBlock,
  askHistoryPayload,
  fetchKinds,
  inspectTarget,
  requestArtifact,
  requestInside,
  requestInsideLearn,
  fetchSharedGraph,
} from "./learn";
import { openIdbBackend } from "./idbBackend";
import { LocalStore, graphHash, sharedDecision, usageLabel, type StoredPaper } from "./localStore";
import { canForgeAgain, openPlan, replaceWithShared, sharedStillApplies, storedEvents, withTimeout } from "./paperOpen";
import { nullBackend } from "./storeBackend";
import { insideJourneyFits, withStoredArtifact } from "./stored";
import { PaperView } from "./paper";
import { ForgeScene, escapeHtml, renderTex } from "./scene";
import { isTabKind, missAction, trainingPrefetchable, walkthroughShown, type TabId, type TabKind } from "./panelState";
import { Overview } from "./overview";
import { Sheet } from "./sheet";
import { StudyPanel } from "./studyPanel";
import { assemble } from "./timeline";
import { traceCardHtml, traceDim } from "./trace";
import { renderSteps } from "./stepsView";
import { Specimen } from "./specimen";
import { TourDriver, tourSteps } from "./tour";
import { trainingHeat, trainingScript } from "./trainingTour";
import { revealSymbol, showCode } from "./codeView";

const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector<T>(sel)!;

const scene = new ForgeScene($("#stage"));
const paper = new PaperView($("#paper"));
const statusEl = $("#status");
const controls = $("#controls");
const specimen = new Specimen(scene);
// Play holds each step until its animation has played (figurePlay, below).
const driver = new TourDriver(scene, specimen, { waitFor: () => figureSlot.played });
const panel = new StudyPanel($("#panel"));
const sheet = new Sheet($("#panel"), $<HTMLButtonElement>(".sheet-handle"));
// On phones, tapping a tab opens the sheet to half.
$("#panel").addEventListener("click", (e) => {
  if ((e.target as HTMLElement).closest('[role="tab"]')) sheet.reveal();
});
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
// Labels crowd a phone-sized stage; start with them off there (the toolbar brings them back).
if (sheet.active) document.body.classList.add("labels-off");
const overview = new Overview(panel.tab("overview"), {
  onStart: (kind) => {
    panel.select(kind);
    panel.tab(kind).focus(); // the card's panel just hid; keep keyboard focus with the content
  },
  onShowInPaper: (blockId) => {
    void openPaper(blockId);
    panel.tab("paper").focus();
  },
  onTrace: (token) => trace(token === tracing ? null : token),
  onLookInside: (blockId) => void openInside(blockId),
  onOutside: () => closeInside(),
  ask: async (blockId, question, history, onDelta) => {
    if (!graph) return;
    try {
      await askBlock(graph.arxiv_id, blockId, question, askHistoryPayload(history), graph, storedKey(), reading, onDelta);
    } catch (err) {
      if (err instanceof LearnError && err.code === "needs_key") showKey(true);
      throw err;
    }
  },
});
panel.tab("paper").append($("#paper"));

let run = new Run();
// The English graph: what the server signed and what learn / ask requests send.
let graph: LocatedArchitecture | null = null;
// What the visitor sees: the same graph with its prose for their language and level.
let view: LocatedArchitecture | null = null;
let reading: Reading = storedReading();
// A block's inside view while one is open: its English (to rewrite again) and what is shown.
let inside: { blockId: string; english: LocatedArchitecture; shown: LocatedArchitecture } | null = null;
// The dimension being traced through what's on the stage, if any.
let tracing: string | null = null;
/** The graph on the stage: a block's inside view, or the whole model. */
const displayed = () => inside?.shown ?? view;
let pages: string[] = [];
let pinned: string | null = null;
// The graph exactly as the server signed it: publishing re-checks the signature, so never mutate this.
let rawGraph: LocatedArchitecture | null = null;
let saveToken: string | null = null;

// ---------- the browser store: papers the visitor opened before reopen with no request ----------

// Opened once at startup; awaited only when a paper opens, so first paint never waits on IndexedDB. A hung
// `indexedDB.open` (a WebKit bug) gives up after STORE_WAIT and the session runs without a store.
const STORE_WAIT = 1500;
const storeReady: Promise<LocalStore> = withTimeout(openIdbBackend(), STORE_WAIT, nullBackend).then((b) => new LocalStore(b));
// The hash of the English graph on screen (stored replay, network or shared replace): what artifacts are keyed by.
let graphHashNow: string | null = null;
// The open paper's record write: anything else stored for this paper waits on it, so the paper is written first.
let paperSaved: Promise<void> = Promise.resolve();
// The open run is a "forge again": the store is bypassed and the shared check never lands.
let refreshing = false;
// Each forgePaper is one opening: the shared check lands only on the opening it was made for.
let openSeq = 0;
// Where the graph on screen came from (null while a paper is opening): "Forge again" shows for claude/cache only.
let paperSource: GraphSource | null = null;

/**
 * An artifact for graph `g` (scope "main" or a block id; kind a LearnKind or "inside"): from the browser store when it
 * holds one for g's hash (no request, no key), else `fetch`, whose result is stored after the paper's own record. The
 * hash and the paper's record write are taken now, so a result that lands after the paper changed is still filed
 * under the graph it was built for. The read is bounded (a slow store is a miss); the write is never awaited.
 * `fits`: a stored value that no longer fits what's on screen is a miss (and the fetched one replaces it).
 * `shared`: the server's saved or shipped copy only (never a Claude call). Given for the model's own artifacts: a
 * stored gallery/sample copy of a kind the server lists is refreshed from it (see withStoredArtifact).
 */
type Fetched<T> = { value: T; source: string };
function fromStore<T extends AnyArtifact | Architecture>(
  g: LocatedArchitecture,
  scope: string,
  kind: string,
  fetch: () => Promise<Fetched<T>>,
  fits: (v: T) => boolean | Promise<boolean> = () => true,
  shared?: () => Promise<Fetched<T>>,
): Promise<T> {
  const hash = graph === g ? graphHashNow : null;
  if (!hash) return fetch().then((r) => r.value);
  const saved = paperSaved;
  return withStoredArtifact<T>({
    get: async () => {
      const hit = await withTimeout(storeReady.then((s) => s.getArtifact(hash, scope, kind)), STORE_WAIT, null);
      return hit && (await fits(hit.artifact as T)) ? { artifact: hit.artifact as T, source: hit.source } : null;
    },
    listed: !!shared && scope === "main" && serverKinds.has(kind as LearnKind),
    shared: shared ?? (() => Promise.reject(new LearnError("Not shared."))),
    fetch,
    put: (v, source) => saved.then(() => storeReady).then((s) => s.putArtifact(g.arxiv_id, hash, scope, kind, v, source)),
  }).then((r) => r.value);
}

/** What a learn request brought, with where it came from. */
const fetched = (r: { artifact: AnyArtifact; source: string }): Fetched<AnyArtifact> => ({ value: r.artifact, source: r.source });

/** Rewrites kept in this browser for paper `id`: bounded reads, writes after the paper's record. */
function translationStore(id: string): TranslationStore {
  const saved = paperSaved;
  return {
    get: (h, lang, level, kind) => withTimeout(storeReady.then((s) => s.getTranslation(h, lang, level, kind)), STORE_WAIT, null),
    put: (h, lang, level, kind, text) => saved.then(() => storeReady).then((s) => s.putTranslation(id, h, lang, level, kind, text)),
  };
}

// ---------- API key (bring your own; stays in this browser) ----------

const KEY_STORAGE = "paperforge.key";
const keyInput = $<HTMLInputElement>("#key");
const keyToggle = $("#key-toggle");
const keyPanel = $("#key-panel");

// The visitor's model, picked next to the key and named wherever we ask for the key.
const modelSelect = $<HTMLSelectElement>("#model");
const modelHint = $("#model-hint");
modelSelect.innerHTML = MODELS.map((m) => `<option value="${m.id}">${escapeHtml(m.name)}</option>`).join("");
modelSelect.value = chosenModel();

function syncModel() {
  const model = MODELS.find((m) => m.id === chosenModel())!;
  modelHint.textContent = model.note;
  for (const el of document.querySelectorAll(".model-name")) el.textContent = model.name;
}

modelSelect.addEventListener("change", () => {
  chooseModel(modelSelect.value as ModelId);
  syncModel();
});
syncModel();

function storedKey(): string {
  try {
    return localStorage.getItem(KEY_STORAGE) ?? "";
  } catch {
    return "";
  }
}

function syncKey() {
  keyToggle.classList.toggle("set", !!storedKey());
  keyToggle.textContent = storedKey() ? "API key ✓" : "API key";
}

function showKey(open: boolean) {
  keyPanel.hidden = !open;
  keyToggle.setAttribute("aria-expanded", String(open));
  if (!open) return;
  keyInput.value = storedKey();
  keyInput.focus();
  void refreshStoreLine();
}

/** The card's "Saved in this browser" line: usage read within STORE_WAIT, Clear asks before it wipes. */
const storeLine = $("#store-line");
function textButton(label: string, onClick: () => void): HTMLButtonElement {
  const b = document.createElement("button");
  b.type = "button";
  b.className = "key-link";
  b.textContent = label;
  b.addEventListener("click", onClick);
  return b;
}
async function refreshStoreLine(): Promise<void> {
  const read = await withTimeout(
    storeReady.then(async (s) => ({ usage: await s.usage(), available: s.available })),
    STORE_WAIT,
    null,
  );
  if (!read) return void (storeLine.textContent = "");
  const label = usageLabel(read.usage, read.available);
  storeLine.replaceChildren(label.lead);
  if (label.strong) {
    const b = document.createElement("b");
    b.textContent = label.strong;
    storeLine.append(b);
  }
  if (!label.clearable) return;
  const ask = () => {
    storeLine.replaceChildren(
      "Clear everything saved? ",
      textButton("Yes", () => void storeReady.then((s) => s.clear()).then(refreshStoreLine)),
      " · ",
      textButton("No", () => void refreshStoreLine()),
    );
  };
  storeLine.append(" · ", textButton("Clear", ask));
}

keyToggle.addEventListener("click", () => showKey(keyToggle.getAttribute("aria-expanded") !== "true"));
/** Remember the visitor's key in this browser (empty clears it). */
function saveKey(key: string) {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {}
  syncKey();
}

keyInput.addEventListener("change", () => {
  saveKey(keyInput.value.trim());
  if (!isOriginal(reading.lang, reading.level)) void applyReading(); // a rewrite may have been waiting for the key
});
keyInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    keyInput.dispatchEvent(new Event("change"));
    showKey(false);
  }
});
syncKey();

// ---------- forging ----------

function setStatus(message: string | null, error = false) {
  statusEl.hidden = !message;
  statusEl.classList.toggle("error", error);
  statusEl.querySelector(".status-text")!.textContent = message ?? "";
}

async function forgePaper(input: string, refresh = false) {
  run.cancel();
  run = new Run();
  const myRun = run;
  landing.hide();
  $<HTMLInputElement>("#url").value = input.includes("/") ? input : `https://arxiv.org/abs/${input}`;
  document.body.classList.add("has-paper");
  setStatus("Fetching paper…");
  overview.setGraph(null);
  pinned = null;
  rawGraph = saveToken = shareUrl = null;
  syncPublish();
  graph = view = inside = null; // results for the previous paper must not land on this one
  tracing = null;
  paperSource = null; // "Forge again" stays hidden during a forge
  syncInside();
  controls.hidden = true; // Replay / Publish would act on the previous paper until this one loads
  resetLearn();
  graphHashNow = null;
  refreshing = refresh;
  const seq = ++openSeq;

  // A slow store falls through to the network.
  const id = parseArxivId(input);
  const hit =
    id && !refresh
      ? await withTimeout(
          storeReady.then((s) => s.getPaper(id)),
          STORE_WAIT,
          null,
        )
      : null;
  if (myRun !== run) return;
  if (hit && openPlan(hit, refresh) === "replay") {
    void storeReady.then((s) => s.setOpen(hit.arxiv_id));
    for (const e of storedEvents(hit)) {
      if (myRun !== run) return;
      await handle(e, myRun, hit.graph_hash);
    }
    void checkShared(hit, seq);
    return;
  }

  const key = storedKey();
  let res: Response;
  try {
    res = await fetch("/api/forge", {
      method: "POST",
      headers: claudeHeaders(key),
      body: JSON.stringify({ url: input, refresh }),
    });
  } catch {
    setStatus("Can't reach the Paperforge server.", true);
    return;
  }

  const reader = res.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let nl: number;
    while ((nl = buffer.indexOf("\n")) >= 0) {
      const line = buffer.slice(0, nl);
      buffer = buffer.slice(nl + 1);
      if (line.trim() && myRun === run) await handle(JSON.parse(line) as ForgeEvent, myRun);
    }
  }
}

/**
 * In the background, after a stored paper replays: a gallery or sample graph that differs wins. It is stored with the
 * paper's pages and shown exactly as a fresh graph would be.
 */
async function checkShared(p: StoredPaper, seq: number) {
  const shared = await fetchSharedGraph(p.arxiv_id);
  const hash = shared ? await graphHash(shared.graph) : null;
  const still = () => sharedStillApplies({ id: p.arxiv_id, seq }, { id: graph?.arxiv_id ?? null, seq: openSeq, refreshing });
  if (!shared || !hash || !still() || sharedDecision(p, shared, hash) !== "replace") return;
  const next = replaceWithShared(p, shared, hash);
  paperSaved = storeReady.then((s) => s.putPaper(next));
  // Rebuild as for a fresh graph: the old graph's assembly, learning and inside view are put away.
  run.cancel();
  run = new Run();
  const myRun = run;
  pinned = null;
  inside = null;
  tracing = null;
  syncInside();
  resetLearn();
  await handle({ type: "graph", graph: next.graph, source: next.source }, myRun, next.graph_hash);
  if (still() && myRun === run) setStatus("Showing the published version of this paper.");
}

/** `storedHash`: the graph came from the browser store (already saved, its hash known). */
async function handle(e: ForgeEvent, myRun: Run, storedHash?: string) {
  switch (e.type) {
    case "status":
      setStatus(e.stage === "extract" ? "Claude is reading the paper — this takes a minute or two…" : e.message);
      break;
    case "paper":
      pages = e.pages;
      scene.clear();
      $("#title").hidden = true;
      paper.setPaper(e.arxiv_id, e.pages);
      panel.select("paper");
      sheet.snapTo("half");
      paper.show(true);
      paper.startReading();
      history.replaceState(null, "", `?paper=${encodeURIComponent(e.arxiv_id)}`);
      break;
    case "graph": {
      rawGraph = structuredClone(e.graph);
      saveToken = e.save_token ?? null;
      syncPublish();
      graph = e.graph;
      paperSource = e.source;
      syncInside();
      setStatus(null); // a failed rewrite below leaves its message up
      const hash = storedHash ?? (await graphHash(e.graph));
      if (myRun !== run) return;
      graphHashNow = hash;
      if (storedHash === undefined) {
        // Store the paper before anything else for it is written (paperSaved); nothing on screen waits for it.
        const record = {
          arxiv_id: e.graph.arxiv_id,
          pages,
          graph: e.graph,
          graph_hash: hash,
          save_token: e.save_token ?? null,
          source: e.source,
        };
        paperSaved = storeReady.then((s) => {
          if (graph?.arxiv_id === record.arxiv_id) s.setOpen(record.arxiv_id);
          return s.putPaper(record);
        });
      }
      const asked = reading;
      const shown = await localized("graph", e.graph);
      if (myRun !== run) return;
      view = shown;
      overview.setGraph(shown);
      paper.stopReading();
      showTitle();
      void play(myRun);
      if (reading !== asked) void applyReading(); // changed while that rewrite was on its way
      break;
    }
    case "error":
      paper.stopReading();
      setStatus(e.message, true);
      if (e.code === "needs_key") showKey(true);
      break;
  }
}

/**
 * Assemble what's on the stage: the model (first forge, Replay) or a block's inside view. The camera follows the
 * director's cut unless the visitor prefers reduced motion; `direct` forces it (a recording).
 */
async function play(myRun = (run = new Run()), opts: { direct?: boolean } = {}) {
  const g = displayed();
  if (!g) return;
  trace(null);
  document.body.classList.add("assembling");
  controls.hidden = false;
  paper.root.classList.remove("inspect");
  panel.select("paper"); // the sweep plays in the Paper tab (first forge and Replay)
  sheet.snapTo("half"); // phones: the sweep happens in the sheet
  scene.reduceMotion = reducedMotion();
  const direct = opts.direct || !reducedMotion();
  if (!direct) myRun.skip(); // straight to the finished model
  await assemble(scene, paper, g, myRun, direct);
  if (myRun === run) {
    document.body.classList.remove("assembling");
    panel.select("overview");
    sheet.snapTo("tabs"); // phones: hand the screen to the model
  }
  toolbar.sync();
  void refreshKinds();
}

function showTitle() {
  const g = displayed();
  if (!g || !view) return;
  $("#title").hidden = false;
  $(".tc-paper").textContent = inside ? `Inside the ${view.model_name} · arXiv:${g.arxiv_id}` : `${g.paper_title} · arXiv:${g.arxiv_id}`;
  $(".tc-model").textContent = g.model_name;
  $(".tc-summary").textContent = g.summary;
  // A taste on the stage; the full list lives in Overview.
  $(".tc-hparams").innerHTML = g.hyperparameters
    .slice(0, 4)
    .map((h) => `<span class="chip">${escapeHtml(h.name)} = <b>${escapeHtml(h.value)}</b></span>`)
    .join("");
}

// ---------- inspecting ----------

/** Overview follows the mouse, except while a question about the pinned block is in progress. */
function inspect(id: string | null) {
  overview.showBlock(id, pinned);
}

scene.onHover = (id) => {
  if (!document.body.classList.contains("assembling")) inspect(inspectTarget(id, pinned, overview.askBusy));
};

scene.onClick = (id) => {
  if (document.body.classList.contains("assembling")) return;
  pinned = id;
  inspect(id);
  if (!id) return;
  const touring = panel.state.active === "journey" && tourKind !== null;
  if (sheet.active && !touring) panel.select("overview"); // phones: a tapped block opens Overview
  sheet.reveal();
  const active = panel.state.active;
  if (active === "pytorch") revealSymbol(panel.tab("pytorch"), id);
  else if (active === "paper") void openPaper(id);
  else if (active === "journey" && tourKind && driver.script) {
    // Clicking a block during a tour jumps to its step.
    const step = driver.script.steps.findIndex((s) => s.block_id === id);
    if (step >= 0) driver.goTo(driver.script, step);
  }
};

scene.onDoubleClick = (id) => {
  if (!document.body.classList.contains("assembling")) void openInside(id);
};

/** Show the paper at a block's source sentence (the Paper tab). */
async function openPaper(blockId?: string) {
  const g = displayed();
  if (!g) return;
  panel.select("paper");
  const block = g.blocks.find((b) => b.id === blockId) ?? g.blocks.find((b) => b.source);
  if (block?.source) await paper.reveal(block, new Run());
}

// ---------- publishing ----------

const publishBtn = controls.querySelector<HTMLButtonElement>('[data-action="publish"]')!; // owned by the toolbar
let shareUrl: string | null = null;

function syncPublish(label = "Publish to gallery") {
  publishBtn.hidden = !saveToken && !shareUrl;
  publishBtn.textContent = shareUrl ? "Copy link" : label;
}

async function publish() {
  if (shareUrl) {
    await navigator.clipboard?.writeText(shareUrl).catch(() => {});
    syncPublish();
    publishBtn.textContent = "Link copied";
    return;
  }
  if (!rawGraph || !saveToken) return;
  // Everything about this paper is read now: the visitor may open another before the answer comes back.
  const sent = rawGraph;
  const id = sent.arxiv_id;
  const token = saveToken;
  const saved = paperSaved; // this paper's record write, which forgetToken must follow
  publishBtn.textContent = "Publishing…";
  const res = await fetch("/api/papers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph: sent, save_token: token }),
  }).catch(() => null);
  const body = ((await res?.json().catch(() => null)) ?? {}) as { arxiv_id?: string; error?: string };
  const stillOpen = rawGraph === sent;
  if (res && (res.status === 201 || res.status === 409)) {
    // Spent: Publish must not come back when this paper reopens from the store, even if the visitor moved on.
    const forgot = saved.then(() => storeReady).then((s) => s.forgetToken(id));
    if (!stillOpen) return;
    paperSaved = forgot;
    saveToken = null;
    shareUrl = `${location.origin}/?paper=${encodeURIComponent(body.arxiv_id ?? id)}`;
    syncPublish();
    setStatus(res.status === 201 ? "Published to the gallery." : "Already in the gallery — Copy link shares that version.");
    setTimeout(() => setStatus(null), 3500);
  } else {
    if (!stillOpen) return;
    syncPublish();
    setStatus(body.error ?? "Couldn't publish right now.", true);
  }
}

// ---------- learning: PyTorch, walkthrough (the journey), ask ----------

// Artifacts opened for this paper, so revisiting a tab never pays for a second generation.
const artifacts = new Map<LearnKind, AnyArtifact>();
// Kinds the server says exist for this paper (gallery or sample): opening one is free even when the browser lacks it.
const serverKinds = new Set<LearnKind>();
const isKind = isTabKind;
// The walkthrough, or the training step that follows it; both play in the Walkthrough tab.
type TourKind = "journey" | "training";
// Where each tour was left, so switching tabs mid-tour picks up at the same step.
// Keyed by scope: "main" for the model's walkthrough, a block id for that block's inside walkthrough,
// "training" for the training step.
const tourAt: Record<string, number> = {};
let tourKind: TourKind | null = null;
let tourScope = "main";
/** Which walkthrough the visitor is in: the model's, or the open inside view's. */
const scopeNow = () => inside?.blockId ?? "main";
const progress: Partial<Record<TabKind, string>> = {};

const NAMES: Record<TabKind, string> = { journey: "walkthrough", pytorch: "PyTorch code" };

let openTab: TabId | null = null;
panel.onSelect = (tab) => {
  if (tourKind && tab !== "journey" && driver.active) {
    tourAt[tourScope] = driver.index;
    driver.pause();
    specimen.hide(); // it comes back on the same step when the tab does
    stopFigure();
    scene.setHeat(null); // so does the training heat
  }
  const again = tab === openTab;
  openTab = tab;
  if (!isKind(tab) || !graph) return;
  // Inside a block, the walkthrough is that block's own; code still points at the model's blocks.
  if (tab === "journey" && inside) return void openInsideJourney(inside.blockId);
  closeInside();
  if (again && artifacts.has(tab)) return; // already showing (e.g. a tap to open the phone sheet)
  if (artifacts.has(tab) || panel.state.status[tab] === "ready") void openLearn(tab, false);
  else renderState(tab);
};

/** A learning tab's body before its content exists: generate, generating, key needed, or error. */
function renderState(kind: TabKind) {
  const root = panel.tab(kind);
  const status = panel.state.status[kind];
  if (status === "generating") {
    root.innerHTML = `
      <div class="learn-state" aria-busy="true">
        <div class="skeleton"><i></i><i></i><i></i><i></i><i></i></div>
        <p class="learn-progress">${escapeHtml(progress[kind] ?? `Claude is writing the ${NAMES[kind]}…`)}</p>
      </div>`;
  } else if (status === "needs_key") {
    root.innerHTML = `
      <form class="learn-state key-card">
        <h3>Add your Anthropic API key</h3>
        <p><b class="model-name">${escapeHtml(modelName(chosenModel()))}</b> writes the ${NAMES[kind]} for this paper using your key. It stays in this browser: Paperforge's server uses it for this request only and never saves or logs it.</p>
        <label class="sr-only" for="key-${kind}">Anthropic API key</label>
        <input id="key-${kind}" type="password" autocomplete="off" spellcheck="false" placeholder="sk-ant-…" required />
        <button class="btn btn-primary">Save key and write the ${NAMES[kind]}</button>
      </form>`;
    root.querySelector("form")!.addEventListener("submit", (e) => {
      e.preventDefault();
      const key = root.querySelector<HTMLInputElement>("input")!.value.trim();
      if (!key) return;
      saveKey(key);
      void openLearn(kind);
    });
  } else if (status === "error") {
    root.innerHTML = `
      <div class="learn-state">
        <h3>The ${NAMES[kind]} didn't come through</h3>
        <p>${escapeHtml(panel.state.error[kind] ?? "Something went wrong.")}</p>
        <button type="button" class="btn btn-primary" data-retry>Try again</button>
      </div>`;
    root.querySelector("[data-retry]")!.addEventListener("click", () => void openLearn(kind));
  } else {
    root.innerHTML = `
      <div class="learn-state">
        <h3>${kind === "pytorch" ? "PyTorch code for this model" : "An example, followed through the model"}</h3>
        <p><b class="model-name">${escapeHtml(modelName(chosenModel()))}</b> writes this from the paper, usually in under a minute, using your API key.</p>
        <button type="button" class="btn btn-primary" data-generate>Write the ${NAMES[kind]}</button>
      </div>`;
    root.querySelector("[data-generate]")!.addEventListener("click", () => void openLearn(kind));
  }
}

// One request per kind at a time; a repeat click joins it instead of paying again.
const learning = new InFlight<AnyArtifact>();

function resetLearn() {
  learning.reset();
  driver.forget();
  tourKind = null;
  openTab = null;
  for (const k of ["journey", "pytorch"] as const) {
    panel.tab(k).replaceChildren();
    delete progress[k];
  }
  for (const k of Object.keys(tourAt)) delete tourAt[k];
  tourScope = "main";
  trainingReq = trainingArtifact = null;
  trainingFailed = false;
  trainingOn = false;
  scene.setHeat(null);
  artifacts.clear();
  serverKinds.clear();
  insideJourneys.clear();
  translations.clear();
  motions.clear();
  stopFigure();
  insides.clear();
  panel.dispatch({ type: "reset" });
}

async function refreshKinds() {
  const id = graph?.arxiv_id;
  if (!id) return;
  const hash = graphHashNow;
  const [kinds, local] = await Promise.all([fetchKinds(id), storedKinds(hash)]);
  if (graph?.arxiv_id !== id || graphHashNow !== hash) return;
  for (const k of kinds) serverKinds.add(k);
  panel.dispatch({ type: "kinds", ready: [...new Set([...kinds, ...local])] });
}

/** The learning kinds this browser already holds for the graph with this hash (they open with no request). */
async function storedKinds(hash: string | null): Promise<LearnKind[]> {
  if (!hash) return [];
  const has = await Promise.all(
    LEARN_KINDS.map((k) => withTimeout(storeReady.then((s) => s.getArtifact(hash, "main", k)).then((a) => !!a), STORE_WAIT, false)),
  );
  return LEARN_KINDS.filter((_, i) => has[i]);
}

/** `explicit`: the visitor clicked for it (not a "ready" tab opening on its own), so a store miss may call Claude. */
async function openLearn(kind: TabKind, explicit = true) {
  if (!graph) return;
  const g = graph;
  const cached = artifacts.get(kind);
  // Stored in this browser: shown with no request; otherwise the tab shows its progress while Claude writes.
  const request = async () => {
    // Opened on its own because the browser seemed to hold it, and the server has none: never a Claude call unasked.
    if (missAction({ explicit, serverReady: serverKinds.has(kind) }) === "unwritten") throw new LearnError("Not in this browser.", "not_stored");
    if (graph === g) {
      panel.dispatch({ type: "generate", kind });
      if (panel.state.active === kind) renderState(kind);
    }
    const r = await requestArtifact(g.arxiv_id, kind, g, storedKey(), (m) => {
      progress[kind] = m;
      panel.tab(kind).querySelector(".learn-progress")?.replaceChildren(m);
    });
    return fetched(r);
  };
  // The server's copy, when it lists this kind: a stored shared copy may have been replaced since (free, no Claude).
  const shared = () => requestArtifact(g.arxiv_id, kind, g, storedKey(), () => {}, undefined, true).then(fetched);
  try {
    const artifact = cached ?? (await learning.run(kind, () => fromStore(g, "main", kind, request, undefined, shared))).value;
    if (graph !== g) return; // the visitor moved on to another paper meanwhile
    artifacts.set(kind, artifact);
    // Translate while the tab still shows its progress, then mark it ready.
    if (panel.state.active === kind) await showArtifact(kind, artifact, g);
    if (graph !== g) return;
    panel.dispatch({ type: "loaded", kind });
  } catch (err) {
    if (graph !== g) return;
    const e = err instanceof LearnError ? err : new LearnError(String(err));
    if (e.code === "not_stored") panel.dispatch({ type: "unready", kind });
    else panel.dispatch({ type: "failed", kind, message: e.message, needsKey: e.code === "needs_key" });
    renderState(kind);
  }
}

/** Render an artifact for the visitor's language and level (the original if the rewrite can't be had). */
async function showArtifact(kind: TabKind, english: AnyArtifact, g: LocatedArchitecture) {
  if (kind === "journey" && trainingOn) return openTraining(); // the visitor went on to train: stay there
  const asked = reading;
  if (!isOriginal(asked.lang, asked.level)) panel.tab(kind).querySelector(".learn-progress")?.replaceChildren(readingMessage(asked));
  const artifact = await localized(kind, english);
  // Stale: another paper, another reading since, or the visitor left the tab (it renders on return).
  if (graph !== g || reading !== asked || panel.state.active !== kind || !view) return;
  if (kind === "pytorch") {
    showCode(panel.tab("pytorch"), artifact as PytorchArtifact, (blockId) => {
      scene.highlight(blockId);
      void scene.focusOn(blockId, new Run());
    });
    return;
  }
  renderJourney(artifact as JourneyArtifact, "main", view);
}

/** The step list for a walkthrough (already in the reader's language) against the graph it walks. */
function renderJourney(journey: JourneyArtifact, scope: string, shownGraph: LocatedArchitecture) {
  const script = tourSteps(journey);
  // A language or level change mid-tour keeps the visitor on the same step.
  const at = driver.active && tourKind === "journey" && tourScope === scope ? driver.index : (tourAt[scope] ?? -1);
  tourKind = "journey";
  tourScope = scope;
  scene.setHeat(null);
  renderSteps(
    panel.tab("journey"),
    script,
    shownGraph,
    driver,
    at,
    (token) => trace(token === tracing ? null : token),
    tracing,
    (el, blockId) => {
      figureAt = el && blockId ? { el, blockId, scope } : null;
      refreshFigure();
      // The last step is in view: have an existing training step ready by the time the visitor asks for it.
      if (scope === "main" && driver.index >= script.steps.length - 1) prefetchTraining();
    },
    // Training follows the model's own example; an inside walkthrough has none.
    scope === "main" ? { label: "Train on this example →", run: startTraining } : undefined,
  );
  loadMotion(scope);
}

// ---------- training: one real step on the walkthrough's example ----------

// The training artifact for this paper (English): fetched in the background once the walkthrough's last step shows.
let trainingReq: Promise<TrainingArtifact> | null = null;
// The background fetch failed (e.g. no key): don't keep asking on every step; the button still asks again.
let trainingFailed = false;
// The Walkthrough tab shows the training tour (it comes back there after a tab switch or a language change).
let trainingOn = false;
let trainingProgress = "";
// What trainingReq brought, once it has: the tour then opens without a loading state.
let trainingArtifact: TrainingArtifact | null = null;
// One training run per artifact: the numbers never change, only the words do.
const trainingRuns = new WeakMap<TrainingArtifact, Trace>();

/** `existingOnly`: the background prefetch, which takes only a saved or shipped training step (never a Claude call). */
function fetchTraining(existingOnly = false): Promise<TrainingArtifact> | null {
  const g = graph;
  const journey = artifacts.get("journey") as JourneyArtifact | undefined;
  if (!g || !journey) return null;
  if (!trainingReq) {
    trainingFailed = false;
    trainingProgress = "";
    // Always with the graph and the walkthrough: the server fits the toy to them.
    const p = fromStore(g, "main", "training", () =>
      requestArtifact(
        g.arxiv_id,
        "training",
        g,
        storedKey(),
        (m) => {
          trainingProgress = m;
          if (trainingOn) panel.tab("journey").querySelector(".learn-progress")?.replaceChildren(m);
        },
        journey,
        existingOnly,
      ).then(fetched),
      undefined,
      () => requestArtifact(g.arxiv_id, "training", g, storedKey(), () => {}, journey, true).then(fetched),
    ).then((artifact) => {
      if (trainingReq === p) trainingArtifact = artifact as TrainingArtifact;
      return artifact as TrainingArtifact;
    });
    trainingReq = p;
    p.catch(() => {
      if (trainingReq !== p) return;
      trainingReq = null;
      trainingFailed = true;
    });
  }
  return trainingReq;
}

/** Background fetch, only of a training step that already exists (never a Claude call unasked). */
function prefetchTraining() {
  // Nothing built yet (`not_ready`) just means not prefetched: the button asks for real, without existing_only.
  if (!trainingFailed && trainingPrefetchable(panel.state)) fetchTraining(true)?.catch(() => {});
}

/** "Train on this example →": leave the walkthrough where it is and start the training tour at its beginning. */
function startTraining() {
  if (!graph || inside) return;
  if (driver.active && tourKind === "journey") tourAt[tourScope] = driver.index;
  delete tourAt.training;
  driver.stop();
  trainingOn = true;
  void openTraining();
}

/** Back from the training tour to the walkthrough, at the step it was left on. */
function leaveTraining() {
  trainingOn = false;
  driver.stop();
  scene.setHeat(null);
  tourKind = null;
  tourScope = "main";
  void showWalkthrough();
}

/** Show the training tour in the Walkthrough tab (fetching it if it isn't here yet), in the reader's language. */
async function openTraining() {
  const g = graph;
  const root = panel.tab("journey");
  const pending = fetchTraining();
  if (!g || !pending) return;
  const still = () => graph === g && trainingOn && !inside && panel.state.active === "journey";
  if (!trainingArtifact && still()) {
    root.innerHTML = `
      <div class="learn-state" aria-busy="true">
        <div class="skeleton"><i></i><i></i><i></i><i></i><i></i></div>
        <p class="learn-progress">${escapeHtml(trainingProgress || "Claude is building a toy version of the model to train…")}</p>
      </div>`;
  }
  try {
    const english = await pending;
    if (!still()) return;
    const asked = reading;
    const shown = await localized("training", english);
    if (!still() || reading !== asked || !view) return;
    let run = trainingRuns.get(english);
    if (!run) trainingRuns.set(english, (run = train(english)));
    const t = run;
    const script = trainingScript(shown, t);
    // A language or level change mid-tour keeps the visitor on the same step.
    const at = driver.active && tourKind === "training" ? driver.index : (tourAt.training ?? -1);
    stopFigure();
    tourKind = "training";
    tourScope = "training";
    renderSteps(
      root,
      script,
      view,
      driver,
      at,
      (token) => trace(token === tracing ? null : token),
      tracing,
      // Where the error has landed so far: the blocks visited, glowing with their gradient.
      () => scene.setHeat(trainingHeat(script, t, driver.index)),
      { label: "← Back to the walkthrough", run: leaveTraining },
    );
  } catch (err) {
    if (!still()) return;
    const e = err instanceof LearnError ? err : new LearnError(String(err));
    // The button caught the prefetch, which found nothing built: now ask for real (its request is already cleared).
    if (e.code === "not_ready" && !trainingReq) return void openTraining();
    if (e.code === "needs_key") {
      root.innerHTML = `
        <form class="learn-state key-card">
          <h3>Add your Anthropic API key</h3>
          <p><b class="model-name">${escapeHtml(modelName(chosenModel()))}</b> builds a toy version of this model to train, using your key. It stays in this browser: Paperforge's server uses it for this request only and never saves or logs it.</p>
          <label class="sr-only" for="key-training">Anthropic API key</label>
          <input id="key-training" type="password" autocomplete="off" spellcheck="false" placeholder="sk-ant-…" required />
          <button class="btn btn-primary">Save key and train</button>
          <button type="button" class="btn" data-back>← Back to the walkthrough</button>
        </form>`;
      root.querySelector("form")!.addEventListener("submit", (ev) => {
        ev.preventDefault();
        const key = root.querySelector<HTMLInputElement>("input")!.value.trim();
        if (!key) return;
        saveKey(key);
        void openTraining();
      });
    } else {
      root.innerHTML = `
        <div class="learn-state">
          <h3>The training step didn't come through</h3>
          <p>${escapeHtml(e.message)}</p>
          <button type="button" class="btn btn-primary" data-retry>Try again</button>
          <button type="button" class="btn" data-back>← Back to the walkthrough</button>
        </div>`;
      root.querySelector("[data-retry]")!.addEventListener("click", () => void openTraining());
    }
    root.querySelector("[data-back]")!.addEventListener("click", leaveTraining);
  }
}

// ---------- the walkthrough inside a block ----------

// Inside walkthroughs for this paper (English), by block id, so reopening one is free.
const insideJourneys = new Map<string, Promise<JourneyArtifact>>();

/** Show the open inside view's own walkthrough in the Walkthrough tab (generating it the first time). */
async function openInsideJourney(blockId: string) {
  const g = graph;
  const at = inside;
  if (!g || !at || at.blockId !== blockId) return;
  const root = panel.tab("journey");
  const label = view?.blocks.find((b) => b.id === blockId)?.label ?? blockId;
  const still = () => graph === g && inside?.blockId === blockId && panel.state.active === "journey";
  let pending = insideJourneys.get(blockId);
  if (!pending) {
    const request = () => {
      if (still())
        root.innerHTML = `
          <div class="learn-state" aria-busy="true">
            <div class="skeleton"><i></i><i></i><i></i><i></i><i></i></div>
            <p class="learn-progress">${escapeHtml(`Claude is following the input inside ${label}…`)}</p>
          </div>`;
      return requestInsideLearn(g.arxiv_id, blockId, "journey", { graph: g, inside: at.english, ...blockEnds(artifacts.get("journey") as JourneyArtifact | undefined, blockId) }, storedKey(), (m) =>
        root.querySelector(".learn-progress")?.replaceChildren(m),
      ).then((r) => ({ value: r.artifact as JourneyArtifact, source: r.source }));
    };
    // A stored one must start from the example the model's walkthrough brings into this block.
    const ends = blockEnds(artifacts.get("journey") as JourneyArtifact | undefined, blockId);
    const p = fromStore(g, blockId, "journey", request, (j) => insideJourneyFits(j, ends));
    insideJourneys.set(blockId, (pending = p));
    p.catch(() => insideJourneys.get(blockId) === p && insideJourneys.delete(blockId));
  }
  try {
    const english = await pending;
    if (!still()) return;
    const asked = reading;
    const shown = await localized("journey", english, blockId);
    if (!still() || reading !== asked) return;
    renderJourney(shown, blockId, inside!.shown);
  } catch (err) {
    if (!still()) return;
    const e = err instanceof LearnError ? err : new LearnError(String(err));
    root.innerHTML = `
      <div class="learn-state">
        <h3>${e.code === "needs_key" ? "Add your Anthropic API key" : `The walkthrough inside ${escapeHtml(label)} didn't come through`}</h3>
        <p>${escapeHtml(e.code === "needs_key" ? `${modelName(chosenModel())} writes this walkthrough using your key. Add it in the top bar, then try again.` : e.message)}</p>
        <button type="button" class="btn btn-primary" data-retry>Try again</button>
      </div>`;
    root.querySelector("[data-retry]")!.addEventListener("click", () => void openInsideJourney(blockId));
    if (e.code === "needs_key") showKey(true);
  }
}

// ---------- step motion: each walkthrough step animates its block's work ----------

// Motion for this paper (English) by scope: "main", or a block id for its inside walkthrough.
const motions = new Map<string, { motion: MotionArtifact | null; status: FigureStatus; asked: boolean }>();
let figure: StepFigure | null = null;
let figureAt: { el: HTMLElement; blockId: string; scope: string } | null = null;
const figureSlot = new FigureSlot();

/** Fetch a walkthrough's motion once, in the background; the walkthrough never waits on it. */
function loadMotion(scope: string) {
  const g = graph;
  if (!g || motions.get(scope)?.asked) return;
  const state = { motion: null as MotionArtifact | null, status: "loading" as FigureStatus, asked: true };
  motions.set(scope, state);
  const request =
    scope === "main"
      ? fromStore(
          g,
          "main",
          "motion",
          () => requestArtifact(g.arxiv_id, "motion", g, storedKey(), () => {}, artifacts.get("journey") as JourneyArtifact | undefined).then(fetched),
          // A stored motion must animate the walkthrough on screen.
          (m) => {
            const journey = artifacts.get("journey") as JourneyArtifact | undefined;
            return !!journey && motionFits(m as MotionArtifact, journey);
          },
          () => requestArtifact(g.arxiv_id, "motion", g, storedKey(), () => {}, artifacts.get("journey") as JourneyArtifact | undefined, true).then(fetched),
        )
      : fromStore(g, scope, "motion", () =>
          insideJourneys
            .get(scope)!
            .then((journey) =>
              inside?.blockId === scope
                ? requestInsideLearn(g.arxiv_id, scope, "motion", { graph: g, inside: inside.english, journey }, storedKey(), () => {})
                : Promise.reject(new LearnError("closed")),
            )
            .then(fetched),
          async (m) => motionFits(m as MotionArtifact, await insideJourneys.get(scope)!),
        );
  request
    .then((artifact) => {
      if (graph !== g || motions.get(scope) !== state) return;
      state.motion = artifact as MotionArtifact;
      state.status = "ready";
      if (figureAt?.scope === scope) refreshFigure();
    })
    .catch((err: unknown) => {
      if (graph !== g || motions.get(scope) !== state) return;
      state.status = err instanceof LearnError && err.code === "needs_key" ? "needs_key" : "error";
      state.asked = false; // Retry asks again
      if (figureAt?.scope === scope) refreshFigure();
    });
}

/** Show the current step's figure; Play waits on all of it (figureSlot), set before any await. */
function refreshFigure() {
  figure?.stop();
  figure = null;
  const at = figureAt;
  figureSlot.set(at ? () => showFigure(at) : null);
}

/** The figure's animation in the visitor's language, or why there isn't one. */
async function showFigure(at: { el: HTMLElement; blockId: string; scope: string }) {
  const m = motions.get(at.scope);
  const english = m?.motion ?? null;
  const shown = english ? await localized("motion", english, at.scope === "main" ? undefined : at.scope) : null;
  if (figureAt !== at) return; // the visitor moved to another step meanwhile
  const f = new StepFigure(at.el, reducedMotion());
  figure = f;
  const state = figureState(shown, m?.status ?? "loading", at.blockId);
  if (state.kind === "ops") return f.show(state.ops);
  const retry = () => loadMotion(at.scope);
  f.message(state.kind, retry, (key) => (saveKey(key), retry()));
}

function stopFigure() {
  figureAt = null;
  refreshFigure();
}

// ---------- language and level ----------

// Rewrites for this paper by kind, block, language and level, so switching back and forth never asks twice.
const translations = new Map<string, Promise<unknown>>();

/**
 * Content for the visitor's language and level. The server keeps every rewrite it makes, so this
 * costs a Claude call only the first time anyone asks for this text this way. On failure the
 * original stays up and the status line says why.
 */
async function localized<C extends object>(kind: TranslateKind, english: C, blockId?: string): Promise<C> {
  const r = reading;
  if (isOriginal(r.lang, r.level) || !graph) return english;
  const cacheKey = `${kind}:${blockId ?? ""}:${r.lang}:${r.level}`;
  let pending = translations.get(cacheKey) as Promise<C> | undefined;
  if (!pending) {
    const message = readingMessage(r);
    // Kept in this browser: no request, so no "Translating…" either.
    const p = translate(graph.arxiv_id, kind, r, english, storedKey(), blockId, {
      store: translationStore(graph.arxiv_id),
      onRequest: () => setStatus(message),
    });
    translations.set(cacheKey, (pending = p));
    p.catch(() => translations.get(cacheKey) === p && translations.delete(cacheKey)).finally(() => {
      if (statusEl.querySelector(".status-text")!.textContent === message) setStatus(null);
    });
  }
  try {
    return await pending;
  } catch (err) {
    const e = err instanceof LearnError ? err : new LearnError(String(err));
    if (e.code === "needs_key") {
      setStatus("Add your Anthropic API key to have Claude rewrite this for you.", true);
      showKey(true);
    } else setStatus(`Couldn't rewrite this: ${e.message}`, true);
    return english;
  }
}

/** Re-show the open paper for the current reading: labels, title, Overview, trace and the open learning tab. */
async function applyReading() {
  const g = graph;
  if (!g || !view) return;
  const asked = reading;
  const shown = await localized("graph", g);
  const open = inside;
  const shownInside = open && (await localized("expansion", open.english, open.blockId));
  if (graph !== g || reading !== asked || inside !== open) return;
  view = shown;
  if (open && shownInside) inside = { ...open, shown: shownInside };
  const d = displayed()!;
  scene.relabel(d);
  showTitle();
  overview.relabel(d);
  refreshTrace();
  if (!overview.askBusy) inspect(pinned);
  const active = panel.state.active;
  if (active === "journey") return showWalkthrough();
  const artifact = isKind(active) ? artifacts.get(active) : undefined;
  if (isKind(active) && artifact) await showArtifact(active, artifact, g);
}

initReadingPickers($<HTMLSelectElement>("#lang"), $<HTMLSelectElement>("#level"), (next) => {
  reading = next;
  if (isOriginal(next.lang, next.level)) setStatus(null); // drop a stale "couldn't rewrite"
  void applyReading();
});

// ---------- tracing a dimension ----------

const traceCard = $("#trace");
traceCard.addEventListener("click", (e) => {
  if ((e.target as HTMLElement).closest("[data-untrace]")) trace(null);
});

/** Trace a dimension through what's on the stage, or stop with null. */
function trace(token: string | null) {
  tracing = token;
  refreshTrace();
  if (!overview.askBusy) inspect(pinned);
}

function refreshTrace() {
  const g = displayed();
  const t = tracing && g ? traceDim(g, tracing) : null;
  scene.setTrace(t ? new Set(t.blocks.keys()) : null);
  overview.setTraced(t ? t.token : null);
  for (const el of panel.tab("journey").querySelectorAll<HTMLElement>(".dim")) el.setAttribute("aria-pressed", String(el.dataset.dim === t?.token));
  traceCard.hidden = !t;
  traceCard.innerHTML = t && g ? traceCardHtml(t, g) : "";
}

// ---------- looking inside a block ----------

// Inside views opened for this paper (English, located in the paper), so reopening is free.
const insides = new Map<string, Promise<LocatedArchitecture>>();

/** Open a block's inside view: its internal structure, assembled from the paper like the model. */
async function openInside(blockId: string) {
  const g = graph;
  const block = view?.blocks.find((b) => b.id === blockId);
  if (!g || !view || !block || inside) return; // one level deep
  if (tourKind && driver.active) tourAt[tourScope] = driver.index; // "← Whole model" comes back here
  driver.stop();
  scene.setHeat(null);
  let pending = insides.get(blockId);
  if (!pending) {
    const id = g.arxiv_id;
    // The English inside graph is what's stored; it's located in this paper's pages on the way out either way.
    const request = () => {
      if (graph === g) setStatus(`Looking inside ${block.label}…`);
      return requestInside(id, blockId, g, storedKey(), (m) => setStatus(m)).then((r) => ({ value: r.graph, source: r.source }));
    };
    const p = fromStore(g, blockId, "inside", request).then((a) => ({
      ...a,
      arxiv_id: id,
      blocks: a.blocks.map((b) => ({ ...b, source: locateQuote(pages, b.source_quote) })),
    }));
    insides.set(blockId, (pending = p));
    p.catch(() => insides.get(blockId) === p && insides.delete(blockId));
  }
  try {
    const english = await pending;
    if (graph !== g || inside) return;
    setStatus(null); // a failed rewrite below leaves its message up
    const asked = reading;
    const shown = await localized("expansion", english, blockId);
    if (graph !== g || inside || !view) return;
    inside = { blockId, english, shown };
    trainingOn = false; // "← Whole model" comes back to the walkthrough
    pinned = null;
    overview.setInside(shown, view.model_name);
    syncInside();
    showTitle();
    if (reading !== asked) void applyReading();
    await play();
  } catch (err) {
    if (graph !== g) return;
    const e = err instanceof LearnError ? err : new LearnError(String(err));
    setStatus(e.message, true);
    if (e.code === "needs_key") showKey(true);
  }
}

/** Back from a block's inside view to the whole model, assembled. */
function closeInside() {
  if (!inside || !view) return;
  // The inside walkthrough's tour points at inside blocks: remember its step and put it away.
  if (tourScope !== "main") {
    if (driver.active) tourAt[tourScope] = driver.index;
    driver.stop();
    stopFigure();
    tourKind = null;
    tourScope = "main";
  }
  run.cancel(); // an inside assembly may still be playing
  run = new Run();
  document.body.classList.remove("assembling");
  inside = null;
  pinned = null;
  trace(null);
  overview.setInside(view, null);
  paper.setBlocks(view.blocks);
  scene.showAssembled(view);
  syncInside();
  showTitle();
  inspect(null);
  toolbar.sync();
  // Back on the model with its walkthrough open: show the model's own, at the step it was left on.
  if (panel.state.active === "journey") void showWalkthrough();
}

/** Fill the Walkthrough tab with the walkthrough that belongs here (see walkthroughShown). */
async function showWalkthrough() {
  const shown = walkthroughShown(inside?.blockId ?? null, artifacts.has("journey"));
  if (shown.kind === "inside") return openInsideJourney(shown.blockId);
  if (shown.kind === "main") return openLearn("journey");
  renderState("journey");
}

// ---------- controls ----------

let explodeRun = new Run();
const toolbar = initToolbar(controls, {
  skip: () => run.skip(),
  outside: () => closeInside(),
  replay() {
    if (!graph || !displayed()) return;
    driver.stop();
    scene.setHeat(null);
    pinned = null;
    inspect(null);
    paper.setPaper(graph.arxiv_id, pages);
    void play();
  },
  explode() {
    explodeRun.cancel();
    explodeRun = new Run();
    const from = scene.explode;
    const to = from > 0.5 ? 0 : 1;
    const d0 = scene.fitDistance;
    const d1 = scene.distanceFor(to);
    void explodeRun.tween(900, (t) => {
      scene.setExplode(from + (to - from) * t);
      scene.fitDistance = d0 + (d1 - d0) * t;
    });
    scene.explode = to; // so the toggle state reads right immediately
  },
  // Bypass the browser store and the server's cache: the new graph overwrites the stored record. Uses the model picked now.
  again() {
    if (graph && !inside && canForgeAgain(paperSource)) void forgePaper(graph.arxiv_id, true);
  },
  labels: () => document.body.classList.toggle("labels-off"),
  pulse: () => (scene.pulsing ? (scene.pulsing = false) : scene.startPulse()),
  publish: () => void publish(),
  state: () => ({
    explode: scene.explode > 0.5,
    labels: !document.body.classList.contains("labels-off"),
    pulse: scene.pulsing,
  }),
});

/** The toolbar's back button shows only inside a block. */
function syncInside() {
  toolbar.outsideButton.hidden = !inside;
  toolbar.againButton.hidden = !!inside || !canForgeAgain(paperSource);
}

$<HTMLFormElement>("#forge-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const value = $<HTMLInputElement>("#url").value.trim();
  if (value) void forgePaper(value);
});

const landing = initLanding($("#landing"), { scene, stage: $("#stage"), onForge: (id) => void forgePaper(id) });

window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.key === " " && document.body.classList.contains("assembling")) run.skip();
  if (e.key === "Escape") {
    // Esc stops a trace, then the tour you can see, then leaves a block's inside; otherwise it collapses the phone sheet.
    const tourShowing = driver.active && tourKind !== null && panel.state.active === "journey";
    if (tracing) trace(null);
    else if (tourShowing) {
      driver.stop();
      scene.setHeat(null); // Next / Play bring the training heat back with the tour
    }
    else if (inside) closeInside();
    else sheet.snapTo("tabs");
  }
  // Arrow keys step through a tour, unless they're moving between tabs.
  const inTour = driver.active && panel.state.active === "journey";
  if (inTour && !(e.target as HTMLElement).closest?.('[role="tablist"]')) {
    if (e.key === "ArrowRight") driver.next();
    if (e.key === "ArrowLeft") driver.back();
  }
});

const initial = new URLSearchParams(location.search).get("paper");
if (initial) void forgePaper(initial);
else landing.preview();
