// Generate one learning artifact from the terminal and print its token usage:
//   bun run learn <arxiv-url-or-id> <pytorch|journey|motion|training> [--inside <block_id>] [--write]
// motion and training are built on the committed journey; training also warns about loss/optimizer quotes the
// locator can't find in the paper.
// Run two kinds on the same paper within 5 minutes: the second should show a cache read.
// --inside writes a block's inside walkthrough (journey) or its motion, from the committed inside view.
// --write saves it as the committed sample (samples/<id>.<kind>.json, or <id>.inside.<block>.<kind>.json).
import { Architecture } from "../src/shared/graph";
import { join } from "node:path";
import { artifactBlockCount, blockEnds, cleanJourney, JourneyArtifact, LearnKind, pruneArtifact, type AnyArtifact } from "../src/shared/learn";
import { locateQuote } from "../src/shared/locate";
import { cleanMotion, type MotionArtifact } from "../src/shared/motion";
import { cleanTraining, type TrainingArtifact } from "../src/shared/training";
import { forge, loadPaperPages } from "../src/server/forge";
import { generateArtifact, generateInsideLearn } from "../src/server/tutor";

const args = process.argv.slice(2);
const write = args.includes("--write");
const insideAt = args.indexOf("--inside");
const blockId = insideAt >= 0 ? args[insideAt + 1] : undefined;
const [input, kindArg] = args.filter((a, i) => a !== "--write" && (insideAt < 0 || (i !== insideAt && i !== insideAt + 1)));
const kind = LearnKind.safeParse(kindArg);
if (!input || !kind.success || (insideAt >= 0 && (!blockId || kind.data === "pytorch" || kind.data === "training"))) {
  console.error("usage: bun run learn <arxiv-url-or-id> <pytorch|journey|motion|training> [--inside <block_id>] [--write]");
  process.exit(1);
}

const located = await forge(input, (e) => e.type === "status" && console.log(`· ${e.message}`));
const graph = Architecture.parse(located); // same shape the server sends Claude
const pages = await loadPaperPages(located.arxiv_id);
const started = performance.now();
const samples = join(import.meta.dir, "..", "samples");
const file = (k: string) => join(samples, `${located.arxiv_id.replace(/\//g, "_")}.${k}.json`);
const need = async <T>(k: string, parse: (v: unknown) => T, hint: string): Promise<T> => {
  const f = Bun.file(file(k));
  if (!(await f.exists())) {
    console.error(`Missing ${file(k)}: ${hint}`);
    process.exit(1);
  }
  return parse(await f.json());
};

let artifact: AnyArtifact;
let name: string = kind.data;
if (blockId) {
  // A block's inside walkthrough continues the main one: from what enters the block to what leaves.
  const block = graph.blocks.find((b) => b.id === blockId);
  if (!block) throw new Error(`No block ${blockId} in ${graph.model_name}`);
  const inside = await need(`inside.${blockId}`, (v) => Architecture.parse(v), "open the block's inside view first");
  const main = await need("journey", (v) => JourneyArtifact.parse(v), "generate the main journey first");
  const { entering, leaving } = blockEnds(main, blockId);
  const journey = kind.data === "motion" ? await need(`inside.${blockId}.journey`, (v) => JourneyArtifact.parse(v), "generate the inside journey first") : undefined;
  const made = await generateInsideLearn(kind.data as "journey" | "motion", { pages, graph, block, inside, entering, leaving, journey });
  artifact = kind.data === "motion" ? cleanMotion(made as MotionArtifact, inside, journey!) : cleanJourney(made as JourneyArtifact, inside);
  name = `inside.${blockId}.${kind.data}`;
} else {
  // Motion animates (and training trains on) the committed walkthrough, so the samples line up with each other.
  const onJourney = kind.data === "motion" || kind.data === "training";
  const journey = onJourney ? await need("journey", (v) => JourneyArtifact.parse(v), "generate the journey first") : undefined;
  const generated = await generateArtifact(kind.data, { pages, graph, journey });
  if (kind.data === "training") {
    const toy = generated as TrainingArtifact;
    for (const [what, quote] of [["loss", toy.loss.quote], ["optimizer", toy.optimizer.quote]] as const) {
      if (quote.trim() && !locateQuote(pages, quote)) console.warn(`! ${what} quote not found in the paper: "${quote}"`);
    }
    const clean = cleanTraining(toy, graph);
    if (!clean) {
      console.error("Couldn't build a toy model of this paper to train (cleanTraining rejected it).");
      process.exit(1);
    }
    artifact = clean;
  } else {
    artifact = kind.data === "motion" ? cleanMotion(generated as MotionArtifact, graph, journey!) : pruneArtifact(kind.data, generated, graph);
  }
}
console.log(`\n${name} for ${graph.model_name}: ${artifactBlockCount(kind.data, artifact)} blocks mapped in ${((performance.now() - started) / 1000).toFixed(0)} s`);
if (write) {
  const path = file(name);
  await Bun.write(path, JSON.stringify(artifact, null, 2) + "\n");
  console.log(`wrote ${path}`);
}
