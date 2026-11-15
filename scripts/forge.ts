// Run the pipeline from the terminal: bun run forge <arxiv-url> [--refresh]
import { forge } from "../src/server/forge";

const args = process.argv.slice(2);
const input = args.find((a) => !a.startsWith("--"));
if (!input) {
  console.error("usage: bun run forge <arxiv-url-or-id> [--refresh]");
  process.exit(1);
}

const graph = await forge(input, (e) => {
  if (e.type === "status") console.log(`· ${e.message}`);
  if (e.type === "paper") console.log(`· ${e.pages.length} pages`);
}, { refresh: args.includes("--refresh") });

console.log(`\n${graph.paper_title} → ${graph.model_name}: ${graph.blocks.length} blocks, ${graph.edges.length} edges`);
for (const b of graph.blocks) {
  const where = b.source ? `p.${b.source.page + 1}` : "quote not found";
  console.log(`  ${b.label.padEnd(30)} ${b.output_shape.padEnd(16)} ${where}`);
}
