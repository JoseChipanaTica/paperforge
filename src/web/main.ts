import "katex/dist/katex.min.css";
import type { ForgeEvent, LocatedArchitecture } from "../shared/graph";
import type { AnyArtifact, ExampleArtifact, LearnKind, PytorchArtifact, WalkthroughArtifact } from "../shared/learn";
import { Run } from "./anim";
import { loadGallery } from "./gallery";
import {
  InFlight,
  LearnError,
  askBlock,
  askHistoryPayload,
  fetchKinds,
  inspectTarget,
  requestArtifact,
} from "./learn";
import { PaperView } from "./paper";
import { ForgeScene, escapeHtml, renderTex } from "./scene";
import type { TabId } from "./panelState";
import { Overview } from "./overview";
import { StudyPanel } from "./studyPanel";
import { assemble } from "./timeline";
import { renderSteps } from "./stepsView";
import { TourDriver, tourSteps } from "./tour";
import { revealSymbol, showCode } from "./codeView";

const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector<T>(sel)!;

const scene = new ForgeScene($("#stage"));
const paper = new PaperView($("#paper"));
const statusEl = $("#status");
const controls = $("#controls");
const driver = new TourDriver(scene);
const panel = new StudyPanel($("#panel"));
const overview = new Overview(panel.tab("overview"), {
  onStart: (kind) => panel.select(kind),
  onShowInPaper: (blockId) => void openPaper(blockId),
  ask: async (blockId, question, history, onDelta) => {
    if (!graph) return;
    try {
      await askBlock(graph.arxiv_id, blockId, question, askHistoryPayload(history), graph, storedKey(), onDelta);
    } catch (err) {
      if (err instanceof LearnError && err.code === "needs_key") openKey();
      throw err;
    }
  },
});
panel.tab("paper").append($("#paper"));

let run = new Run();
let graph: LocatedArchitecture | null = null;
let pages: string[] = [];
let pinned: string | null = null;
// The graph exactly as the server signed it: publishing re-checks the signature, so never mutate this.
let rawGraph: LocatedArchitecture | null = null;
let saveToken: string | null = null;

// ---------- API key (bring your own; stays in this browser) ----------

const KEY_STORAGE = "paperforge.key";
const keyInput = $<HTMLInputElement>("#key");
const keyToggle = $("#key-toggle");

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

function openKey() {
  keyInput.hidden = false;
  keyInput.value = storedKey();
  keyInput.focus();
}

keyToggle.addEventListener("click", () => (keyInput.hidden ? openKey() : (keyInput.hidden = true)));
/** Remember the visitor's key in this browser (empty clears it). */
function saveKey(key: string) {
  try {
    if (key) localStorage.setItem(KEY_STORAGE, key);
    else localStorage.removeItem(KEY_STORAGE);
  } catch {}
  syncKey();
}

keyInput.addEventListener("change", () => saveKey(keyInput.value.trim()));
keyInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") {
    keyInput.dispatchEvent(new Event("change"));
    keyInput.hidden = true;
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
  document.body.classList.add("has-paper");
  setStatus("Fetching paper…");
  overview.setGraph(null);
  pinned = null;
  rawGraph = saveToken = shareUrl = null;
  syncPublish();
  resetLearn();

  const key = storedKey();
  let res: Response;
  try {
    res = await fetch("/api/forge", {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(key && { "x-anthropic-key": key }) },
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

async function handle(e: ForgeEvent, myRun: Run) {
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
      paper.show(true);
      paper.startReading();
      history.replaceState(null, "", `?paper=${encodeURIComponent(e.arxiv_id)}`);
      break;
    case "graph":
      rawGraph = structuredClone(e.graph);
      saveToken = e.save_token ?? null;
      syncPublish();
      graph = e.graph;
      overview.setGraph(e.graph);
      paper.stopReading();
      setStatus(null);
      showTitle(e.graph);
      void play(myRun);
      break;
    case "error":
      paper.stopReading();
      setStatus(e.message, true);
      if (e.code === "needs_key") openKey();
      break;
  }
}

async function play(myRun = (run = new Run())) {
  if (!graph) return;
  document.body.classList.add("assembling");
  controls.hidden = false;
  paper.root.classList.remove("inspect");
  scene.sway = true;
  await assemble(scene, paper, graph, myRun);
  if (myRun === run) {
    document.body.classList.remove("assembling");
    panel.select("overview");
  }
  syncToggles();
  void refreshKinds();
}

function showTitle(g: LocatedArchitecture) {
  $("#title").hidden = false;
  $(".tc-paper").textContent = `${g.paper_title} · arXiv:${g.arxiv_id}`;
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
  const active = panel.state.active;
  if (active === "pytorch") revealSymbol(panel.tab("pytorch"), id);
  else if (active === "paper") void openPaper(id);
  else if ((active === "walkthrough" || active === "example") && driver.script) {
    // Clicking a block during a tour jumps to its step.
    const step = driver.script.steps.findIndex((s) => s.block_id === id);
    if (step >= 0) driver.goTo(driver.script, step);
  }
};

/** Show the paper at a block's source sentence (the Paper tab). */
async function openPaper(blockId?: string) {
  if (!graph) return;
  panel.select("paper");
  const block = graph.blocks.find((b) => b.id === blockId) ?? graph.blocks.find((b) => b.source);
  if (block?.source) await paper.reveal(block, new Run());
}

// ---------- publishing ----------

const publishBtn = controls.querySelector<HTMLButtonElement>('[data-action="publish"]')!;
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
  publishBtn.textContent = "Publishing…";
  const res = await fetch("/api/papers", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ graph: rawGraph, save_token: saveToken }),
  }).catch(() => null);
  const body = ((await res?.json().catch(() => null)) ?? {}) as { arxiv_id?: string; error?: string };
  if (res && (res.status === 201 || res.status === 409)) {
    saveToken = null;
    shareUrl = `${location.origin}/?paper=${encodeURIComponent(body.arxiv_id ?? rawGraph.arxiv_id)}`;
    syncPublish();
    setStatus(res.status === 201 ? "Published to the gallery." : "Already in the gallery — Copy link shares that version.");
    setTimeout(() => setStatus(null), 3500);
  } else {
    syncPublish();
    setStatus(body.error ?? "Couldn't publish right now.", true);
  }
}

// ---------- learning: PyTorch, walkthrough, easy example, ask ----------

// Artifacts opened for this paper, so revisiting a tab never pays for a second generation.
const artifacts = new Map<LearnKind, AnyArtifact>();
const isKind = (t: TabId): t is LearnKind => t === "walkthrough" || t === "example" || t === "pytorch";
type TourKind = "walkthrough" | "example";
// Where each tour was left, so switching tabs mid-tour picks up at the same step.
const tourAt: Partial<Record<TourKind, number>> = {};
let tourKind: TourKind | null = null;
const progress: Partial<Record<LearnKind, string>> = {};

const NAMES: Record<LearnKind, string> = { walkthrough: "walkthrough", example: "easy example", pytorch: "PyTorch code" };

panel.onSelect = (tab) => {
  if (tourKind && tab !== tourKind && driver.active) {
    tourAt[tourKind] = driver.index;
    driver.pause();
  }
  if (!isKind(tab) || !graph) return;
  if (artifacts.has(tab) || panel.state.status[tab] === "ready") void openLearn(tab);
  else renderState(tab);
};

/** A learning tab's body before its content exists: generate, generating, key needed, or error. */
function renderState(kind: LearnKind) {
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
        <p>Claude writes the ${NAMES[kind]} for this paper using your key. It stays in this browser and is only sent to Paperforge's server for this request.</p>
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
        <h3>${kind === "pytorch" ? "PyTorch code for this model" : kind === "walkthrough" ? "A guided walkthrough" : "An easy example"}</h3>
        <p>Claude writes this from the paper, usually in under a minute, using your API key.</p>
        <button type="button" class="btn btn-primary" data-generate>Write the ${NAMES[kind]}</button>
      </div>`;
    root.querySelector("[data-generate]")!.addEventListener("click", () => void openLearn(kind));
  }
}

// One request per kind at a time; a repeat click joins it instead of paying again.
const learning = new InFlight<Awaited<ReturnType<typeof requestArtifact>>>();

function resetLearn() {
  learning.reset();
  driver.stop();
  tourKind = null;
  for (const k of ["walkthrough", "example", "pytorch"] as const) {
    panel.tab(k).replaceChildren();
    delete progress[k];
  }
  delete tourAt.walkthrough;
  delete tourAt.example;
  artifacts.clear();
  panel.dispatch({ type: "reset" });
}

async function refreshKinds() {
  const id = graph?.arxiv_id;
  if (!id) return;
  const kinds = await fetchKinds(id);
  if (graph?.arxiv_id !== id) return;
  panel.dispatch({ type: "kinds", ready: kinds });
}

async function openLearn(kind: LearnKind) {
  if (!graph) return;
  const g = graph;
  const cached = artifacts.get(kind);
  if (!cached) {
    panel.dispatch({ type: "generate", kind });
    if (panel.state.active === kind) renderState(kind);
  }
  try {
    const { value, current } = cached
      ? { value: { artifact: cached }, current: true }
      : await learning.run(kind, () =>
          requestArtifact(g.arxiv_id, kind, g, storedKey(), (m) => {
            progress[kind] = m;
            panel.tab(kind).querySelector(".learn-progress")?.replaceChildren(m);
          }),
        );
    if (graph !== g) return; // the visitor moved on to another paper meanwhile
    const { artifact } = value;
    artifacts.set(kind, artifact);
    panel.dispatch({ type: "loaded", kind });
    if (!current || panel.state.active !== kind) return; // the visitor is looking at something else
    showArtifact(kind, artifact, g);
  } catch (err) {
    if (graph !== g) return;
    const e = err instanceof LearnError ? err : new LearnError(String(err));
    panel.dispatch({ type: "failed", kind, message: e.message, needsKey: e.code === "needs_key" });
    renderState(kind);
  }
}

function showArtifact(kind: LearnKind, artifact: AnyArtifact, g: LocatedArchitecture) {
  if (kind === "pytorch") {
    showCode(panel.tab("pytorch"), artifact as PytorchArtifact, (blockId) => {
      scene.highlight(blockId);
      void scene.focusOn(blockId, new Run());
    });
    return;
  }
  const script =
    kind === "walkthrough" ? tourSteps(kind, artifact as WalkthroughArtifact) : tourSteps(kind, artifact as ExampleArtifact);
  tourKind = kind;
  renderSteps(panel.tab(kind), script, g, driver, tourAt[kind] ?? -1);
}

// ---------- controls ----------

function syncToggles() {
  controls.querySelector('[data-action="explode"]')!.classList.toggle("on", scene.explode > 0.5);
  controls.querySelector('[data-action="labels"]')!.classList.toggle("on", !document.body.classList.contains("labels-off"));
  controls.querySelector('[data-action="pulse"]')!.classList.toggle("on", scene.pulsing);
}

let explodeRun = new Run();
controls.addEventListener("click", (e) => {
  const action = (e.target as HTMLElement).closest<HTMLElement>("button")?.dataset.action;
  switch (action) {
    case "skip":
      run.skip();
      break;
    case "replay":
      pinned = null;
      inspect(null);
      if (graph) {
        paper.setPaper(graph.arxiv_id, pages);
        void play();
      }
      break;
    case "explode": {
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
      break;
    }
    case "labels":
      document.body.classList.toggle("labels-off");
      break;
    case "pulse":
      if (scene.pulsing) scene.pulsing = false;
      else scene.startPulse();
      break;
    case "publish":
      void publish();
      break;
  }
  syncToggles();
});

$<HTMLFormElement>("#forge-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const value = $<HTMLInputElement>("#url").value.trim();
  if (value) void forgePaper(value);
});

document.querySelectorAll<HTMLButtonElement>(".examples button").forEach((btn) =>
  btn.addEventListener("click", () => {
    $<HTMLInputElement>("#url").value = `https://arxiv.org/abs/${btn.dataset.paper}`;
    void forgePaper(btn.dataset.paper!);
  }),
);

window.addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLInputElement) return;
  if (e.key === " " && document.body.classList.contains("assembling")) run.skip();
  if (e.key === "Escape") {
    if (driver.active) driver.stop();
  }
  // Arrow keys step through a tour, unless they're moving between tabs.
  const inTour = driver.active && (panel.state.active === "walkthrough" || panel.state.active === "example");
  if (inTour && !(e.target as HTMLElement).closest?.('[role="tablist"]')) {
    if (e.key === "ArrowRight") driver.next();
    if (e.key === "ArrowLeft") driver.back();
  }
});

if (!document.body.classList.contains("has-paper")) {
  void loadGallery($("#gallery"), (id) => {
    $<HTMLInputElement>("#url").value = id.includes("/") ? id : `https://arxiv.org/abs/${id}`;
    void forgePaper(id);
  });
}

const initial = new URLSearchParams(location.search).get("paper");
if (initial) {
  $<HTMLInputElement>("#url").value = initial.includes("/") ? initial : `https://arxiv.org/abs/${initial}`;
  void forgePaper(initial);
}
