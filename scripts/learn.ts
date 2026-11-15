// Generate one learning artifact from the terminal and print its token usage:
//   bun run learn <arxiv-url-or-id> <pytorch|journey> [--write]
// Run two kinds on the same paper within 5 minutes: the second should show a cache read.
// --write saves it as the committed sample (samples/<id>.<kind>.json).
import { Architecture } from "../src/shared/graph";
import { join } from "node:path";
import { artifactBlockCount, LearnKind, pruneArtifact } from "../src/shared/learn";
import { forge, loadPaperPages } from "../src/server/forge";
import { generateArtifact } from "../src/server/tutor";

const args = process.argv.slice(2);
const write = args.includes("--write");
const [input, kindArg] = args.filter((a) => a !== "--write");
const kind = LearnKind.safeParse(kindArg);
if (!input || !kind.success) {
  console.error("usage: bun run learn <arxiv-url-or-id> <pytorch|journey> [--write]");
  process.exit(1);
}

const located = await forge(input, (e) => e.type === "status" && console.log(`· ${e.message}`));
const graph = Architecture.parse(located); // same shape the server sends Claude
const pages = await loadPaperPages(located.arxiv_id);
const started = performance.now();
const artifact = pruneArtifact(kind.data, await generateArtifact(kind.data, { pages, graph }), graph);
console.log(`\n${kind.data} for ${graph.model_name}: ${artifactBlockCount(kind.data, artifact)} blocks mapped in ${((performance.now() - started) / 1000).toFixed(0)} s`);
if (write) {
  const path = join(import.meta.dir, "..", "samples", `${located.arxiv_id.replace(/\//g, "_")}.${kind.data}.json`);
  await Bun.write(path, JSON.stringify(artifact, null, 2) + "\n");
  console.log(`wrote ${path}`);
}
