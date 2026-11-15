import "katex/dist/katex.min.css";
import type { ForgeEvent, LocatedArchitecture } from "../shared/graph";
import { Run, ease } from "./anim";
import { TYPE_COLOR } from "./blocks";
import { PaperView } from "./paper";
import { ForgeScene, escapeHtml, renderTex } from "./scene";
import { assemble } from "./timeline";

const $ = <T extends HTMLElement = HTMLElement>(sel: string) => document.querySelector<T>(sel)!;

const scene = new ForgeScene($("#stage"));
const paper = new PaperView($("#paper"));
const statusEl = $("#status");
const inspector = $("#inspector");
const controls = $("#controls");

let run = new Run();
let graph: LocatedArchitecture | null = null;
let pages: string[] = [];
let pinned: string | null = null;

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
  document.body.classList.remove("empty");
  setStatus("Fetching paper…");
  inspector.hidden = true;
  pinned = null;

  let res: Response;
  try {
    res = await fetch("/api/forge", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
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
      paper.show(true);
      paper.startReading();
      history.replaceState(null, "", `?paper=${encodeURIComponent(e.arxiv_id)}`);
      break;
    case "graph":
      graph = e.graph;
      paper.stopReading();
      setStatus(null);
      showTitle(e.graph);
      void play(myRun);
      break;
    case "error":
      paper.stopReading();
      setStatus(e.message, true);
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
  if (myRun === run) document.body.classList.remove("assembling");
  syncToggles();
}

function showTitle(g: LocatedArchitecture) {
  $("#title").hidden = false;
  $(".tc-paper").textContent = `${g.paper_title} · arXiv:${g.arxiv_id}`;
  $(".tc-model").textContent = g.model_name;
  $(".tc-summary").textContent = g.summary;
  $(".tc-hparams").innerHTML = g.hyperparameters
    .map((h) => `<span class="chip">${escapeHtml(h.name)} = <b>${escapeHtml(h.value)}</b></span>`)
    .join("");
}

// ---------- inspecting ----------

function inspect(id: string | null) {
  const sb = id ? scene.blocks.get(id) : null;
  if (!sb) {
    inspector.hidden = true;
    return;
  }
  const b = sb.block;
  const page = b.source ? `p. ${b.source.page + 1}` : null;
  inspector.hidden = false;
  inspector.innerHTML = `
    <div class="in-type">${escapeHtml(b.type.replace(/_/g, " "))}${b.group ? ` · ${escapeHtml(b.group)}` : ""}</div>
    <div class="in-title"><span class="dot" style="background:${TYPE_COLOR[b.type]}"></span>${escapeHtml(b.label)}</div>
    <div class="in-shape">${escapeHtml(b.input_shape)} → ${escapeHtml(b.output_shape)}</div>
    ${b.hyperparameters.length ? `<div class="in-hp">${b.hyperparameters.map((h) => `<span class="chip">${escapeHtml(h.name)} = <b>${escapeHtml(h.value)}</b></span>`).join("")}</div>` : ""}
    ${b.equation ? `<div class="in-eq">${renderTex(b.equation, true)}</div>` : ""}
    <p class="in-desc">${escapeHtml(b.description)}</p>
    <blockquote class="in-quote">“${escapeHtml(b.source_quote)}”</blockquote>
    <div class="in-cite ${page ? "" : "missing"}">${page ? `From the paper, ${page} · click the block to open it` : "Quote not found verbatim in the extracted text"}</div>`;
}

scene.onHover = (id) => {
  if (!document.body.classList.contains("assembling")) inspect(id ?? pinned);
};

scene.onClick = (id) => {
  if (document.body.classList.contains("assembling")) return;
  pinned = id;
  inspect(id);
  const sb = id ? scene.blocks.get(id) : null;
  if (sb?.block.source) void openPaper(sb.block.id);
  else closePaper();
};

let shift = new Run();
function shiftView(to: number) {
  shift.cancel();
  shift = new Run();
  const from = scene.viewShift;
  void shift.tween(700, (t) => scene.setViewShift(from + (to - from) * t), ease.inOutCubic);
}

async function openPaper(blockId?: string) {
  if (!graph) return;
  paper.root.classList.add("inspect");
  shiftView(0.17);
  const block = graph.blocks.find((b) => b.id === blockId) ?? graph.blocks.find((b) => b.source);
  if (block) await paper.reveal(block, new Run());
  else paper.show(true);
  syncToggles();
}

function closePaper() {
  if (!paper.visible) return;
  paper.show(false);
  paper.root.classList.remove("inspect");
  shiftView(0);
  syncToggles();
}

// ---------- controls ----------

function syncToggles() {
  controls.querySelector('[data-action="explode"]')!.classList.toggle("on", scene.explode > 0.5);
  controls.querySelector('[data-action="labels"]')!.classList.toggle("on", !document.body.classList.contains("labels-off"));
  controls.querySelector('[data-action="pulse"]')!.classList.toggle("on", scene.pulsing);
  controls.querySelector('[data-action="paper"]')!.classList.toggle("on", paper.visible);
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
      inspector.hidden = true;
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
    case "paper":
      if (paper.visible) closePaper();
      else void openPaper(pinned ?? undefined);
      break;
  }
  syncToggles();
});

$("#paper .paper-close").addEventListener("click", closePaper);

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
  if (e.key === "Escape") closePaper();
});

const initial = new URLSearchParams(location.search).get("paper");
if (initial) {
  $<HTMLInputElement>("#url").value = initial.includes("/") ? initial : `https://arxiv.org/abs/${initial}`;
  void forgePaper(initial);
}
