// Generate one learning artifact from the terminal and print its token usage:
//   bun run learn <arxiv-url-or-id> <pytorch|walkthrough|example>
// Run two kinds on the same paper within 5 minutes: the second should show a cache read.
import { Architecture } from "../src/shared/graph";
import { artifactBlockCount, LearnKind } from "../src/shared/learn";
import { forge, loadPaperPages } from "../src/server/forge";
import { generateArtifact } from "../src/server/tutor";

const [input, kindArg] = process.argv.slice(2);
const kind = LearnKind.safeParse(kindArg);
if (!input || !kind.success) {
  console.error("usage: bun run learn <arxiv-url-or-id> <pytorch|walkthrough|example>");
  process.exit(1);
}

const located = await forge(input, (e) => e.type === "status" && console.log(`· ${e.message}`));
const graph = Architecture.parse(located); // same shape the server sends Claude
const pages = await loadPaperPages(located.arxiv_id);
const started = performance.now();
const artifact = await generateArtifact(kind.data, { pages, graph });
console.log(`\n${kind.data} for ${graph.model_name}: ${artifactBlockCount(kind.data, artifact)} blocks mapped in ${((performance.now() - started) / 1000).toFixed(0)} s`);
