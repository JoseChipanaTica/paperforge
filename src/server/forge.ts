import { join } from "node:path";
import { Architecture, type ForgeEvent, type LocatedArchitecture } from "../shared/graph";
import { fetchArxivPdf, parseArxivId } from "./arxiv";
import { extractArchitecture } from "./extract";
import { locateQuote } from "./locate";
import { extractPages } from "./pdf";

const ROOT = join(import.meta.dir, "..", "..");
const CACHE = join(ROOT, "data", "cache");
const SAMPLES = join(ROOT, "samples");

const fileId = (id: string) => id.replace(/\//g, "_");

async function readJson<T>(path: string): Promise<T | null> {
  const f = Bun.file(path);
  return (await f.exists()) ? ((await f.json()) as T) : null;
}

async function loadPages(id: string, emit: (e: ForgeEvent) => void): Promise<string[]> {
  const pagesPath = join(CACHE, `${fileId(id)}.pages.json`);
  const cached = await readJson<string[]>(pagesPath);
  if (cached) return cached;

  const pdfPath = join(CACHE, `${fileId(id)}.pdf`);
  let pdf: Uint8Array;
  if (await Bun.file(pdfPath).exists()) {
    pdf = await Bun.file(pdfPath).bytes();
  } else {
    emit({ type: "status", stage: "fetch", message: `Downloading arXiv:${id}` });
    pdf = await fetchArxivPdf(id);
    await Bun.write(pdfPath, pdf);
  }
  emit({ type: "status", stage: "read", message: "Extracting text with pdf.js" });
  const pages = await extractPages(pdf);
  await Bun.write(pagesPath, JSON.stringify(pages));
  return pages;
}

async function loadArchitecture(
  id: string,
  pages: string[],
  refresh: boolean,
  emit: (e: ForgeEvent) => void,
): Promise<{ arch: Architecture; cached: boolean }> {
  const cachePath = join(CACHE, `${fileId(id)}.json`);
  if (!refresh) {
    const hit = (await readJson(cachePath)) ?? (await readJson(join(SAMPLES, `${fileId(id)}.json`)));
    if (hit) return { arch: Architecture.parse(hit), cached: true };
  }
  emit({ type: "status", stage: "extract", message: "Claude is reading the paper" });
  const arch = await extractArchitecture(pages);
  await Bun.write(cachePath, JSON.stringify(arch, null, 2));
  return { arch, cached: false };
}

export function locate(id: string, arch: Architecture, pages: string[]): LocatedArchitecture {
  return {
    ...arch,
    arxiv_id: id,
    blocks: arch.blocks.map((b) => ({ ...b, source: locateQuote(pages, b.source_quote) })),
  };
}

export async function forge(
  input: string,
  emit: (e: ForgeEvent) => void,
  { refresh = false } = {},
): Promise<LocatedArchitecture> {
  const id = parseArxivId(input);
  if (!id) throw new Error(`That doesn't look like an arXiv link or id: "${input}"`);

  const pages = await loadPages(id, emit);
  emit({ type: "paper", arxiv_id: id, pages });

  const { arch, cached } = await loadArchitecture(id, pages, refresh, emit);
  emit({ type: "status", stage: "locate", message: "Finding each block in the text" });
  const graph = locate(id, arch, pages);
  emit({ type: "graph", graph, cached });
  return graph;
}
