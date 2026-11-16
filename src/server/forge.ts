import { join } from "node:path";
import { Architecture, type ForgeEvent, type GraphSource, type LocatedArchitecture } from "../shared/graph";
import { fetchArxivPdf, parseArxivId } from "./arxiv";
import { extractArchitecture } from "./extract";
import { locateQuote } from "../shared/locate";
import { extractPages } from "./pdf";
import { sharesResults } from "./model";
import { getStore, type Store } from "./store";
import { getSecret, signGraph } from "./token";

const ROOT = join(import.meta.dir, "..", "..");
// Vercel functions can only write to /tmp; the cache there is per-instance and temporary.
export const CACHE_DIR = process.env.VERCEL ? "/tmp/paperforge" : join(ROOT, "data", "cache");
const SAMPLES = join(ROOT, "samples");

const fileId = (id: string) => id.replace(/\//g, "_");

async function readJson<T>(path: string): Promise<T | null> {
  const f = Bun.file(path);
  return (await f.exists()) ? ((await f.json()) as T) : null;
}

/** The committed sample graph for an arXiv id, if there is one. */
export async function readSampleGraph(id: string): Promise<Architecture | null> {
  const sample = await readJson(join(SAMPLES, `${fileId(id)}.json`));
  return sample ? Architecture.parse(sample) : null;
}

async function loadPages(id: string, emit: (e: ForgeEvent) => void, cacheDir: string): Promise<string[]> {
  const pagesPath = join(cacheDir, `${fileId(id)}.pages.json`);
  const cached = await readJson<string[]>(pagesPath);
  if (cached) return cached;

  const pdfPath = join(cacheDir, `${fileId(id)}.pdf`);
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

/** The paper's text, one string per page (disk cache → arXiv). Also used by the tutor. */
export function loadPaperPages(id: string, emit: (e: ForgeEvent) => void = () => {}): Promise<string[]> {
  return loadPages(id, emit, CACHE_DIR);
}

type Extract = typeof extractArchitecture;

async function loadArchitecture(
  id: string,
  pages: string[],
  { refresh, apiKey, store, extract, cacheDir }: Required<Omit<ForgeOptions, "apiKey" | "loadPages">> & { apiKey?: string },
  emit: (e: ForgeEvent) => void,
): Promise<{ arch: Architecture; source: GraphSource }> {
  const cachePath = join(cacheDir, `${fileId(id)}.json`);
  if (!refresh) {
    // The gallery is optional infrastructure: if the database is down, keep forging.
    const published = await store.getPaper(id).catch((err) => (console.error("gallery lookup failed:", err), null));
    if (published) return { arch: published, source: "gallery" };
    const cached = await readJson(cachePath);
    if (cached) return { arch: Architecture.parse(cached), source: "cache" };
    const sample = await readSampleGraph(id);
    if (sample) return { arch: sample, source: "sample" };
  }
  getSecret(); // fail before the visitor pays for a Claude call we couldn't let them publish
  emit({ type: "status", stage: "extract", message: "Claude is reading the paper" });
  const arch = await extract(pages, { apiKey });
  // The disk cache serves every later visitor, so only a model that shares results writes it.
  if (sharesResults()) await Bun.write(cachePath, JSON.stringify(arch, null, 2));
  return { arch, source: "claude" };
}

export function locate(id: string, arch: Architecture, pages: string[]): LocatedArchitecture {
  return {
    ...arch,
    arxiv_id: id,
    blocks: arch.blocks.map((b) => ({ ...b, source: locateQuote(pages, b.source_quote) })),
  };
}

export interface ForgeOptions {
  refresh?: boolean;
  /** The visitor's own Anthropic key; falls back to ANTHROPIC_API_KEY when absent. */
  apiKey?: string;
  store?: Store;
  extract?: Extract;
  loadPages?: (id: string, emit: (e: ForgeEvent) => void) => Promise<string[]>;
  cacheDir?: string;
}

export async function forge(
  input: string,
  emit: (e: ForgeEvent) => void,
  opts: ForgeOptions = {},
): Promise<LocatedArchitecture> {
  const id = parseArxivId(input);
  if (!id) throw new Error(`That doesn't look like an arXiv link or id: "${input}"`);
  const cacheDir = opts.cacheDir ?? CACHE_DIR;

  const pages = await (opts.loadPages ?? ((id, emit) => loadPages(id, emit, cacheDir)))(id, emit);
  emit({ type: "paper", arxiv_id: id, pages });

  const { arch, source } = await loadArchitecture(
    id,
    pages,
    {
      refresh: opts.refresh ?? false,
      apiKey: opts.apiKey?.trim() || undefined,
      store: opts.store ?? getStore(),
      extract: opts.extract ?? extractArchitecture,
      cacheDir,
    },
    emit,
  );
  emit({ type: "status", stage: "locate", message: "Finding each block in the text" });
  const graph = locate(id, arch, pages);
  // Fresh or disk-cached Claude output is this server's own and publishable, so a reload
  // before clicking Publish keeps the option. Gallery and sample graphs are not, nor graphs
  // from a model whose results stay with the visitor.
  const publishable = source === "cache" || (source === "claude" && sharesResults());
  emit({ type: "graph", graph, source, ...(publishable && { save_token: signGraph(graph) }) });
  return graph;
}
