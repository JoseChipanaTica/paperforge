import type { LocatedArchitecture } from "../shared/graph";
import type { AnyArtifact, AskEvent, AskTurn, LearnEvent, LearnKind } from "../shared/learn";

/** Browser side of the learning features: the /api/learn and /api/ask clients and the code panel. */

export class LearnError extends Error {
  constructor(
    message: string,
    readonly code?: string,
  ) {
    super(message);
  }
}

const headers = (key: string) => ({ "Content-Type": "application/json", ...(key && { "x-anthropic-key": key }) });

async function* lines<T>(res: Response): AsyncGenerator<T> {
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
      if (line.trim()) yield JSON.parse(line) as T;
    }
  }
}

/** Kinds that open instantly, with no key: saved to the gallery or shipped as samples. */
export async function fetchKinds(id: string): Promise<LearnKind[]> {
  try {
    const res = await fetch(`/api/learn?arxiv_id=${encodeURIComponent(id)}`);
    return res.ok ? await res.json() : [];
  } catch {
    return [];
  }
}

export async function requestArtifact(
  id: string,
  kind: LearnKind,
  graph: LocatedArchitecture,
  key: string,
  onStatus: (message: string) => void,
): Promise<{ artifact: AnyArtifact; source: string; saved: boolean }> {
  const res = await fetch("/api/learn", {
    method: "POST",
    headers: headers(key),
    body: JSON.stringify({ arxiv_id: id, kind, graph }),
  }).catch(() => null);
  if (!res?.body) throw new LearnError("Can't reach the Paperforge server.");
  for await (const e of lines<LearnEvent>(res)) {
    if (e.type === "status") onStatus(e.message);
    else if (e.type === "learn") return e;
    else throw new LearnError(e.message, e.code);
  }
  throw new LearnError("The server stopped before finishing.");
}

export async function askBlock(
  id: string,
  blockId: string,
  question: string,
  history: AskTurn[],
  graph: LocatedArchitecture,
  key: string,
  onDelta: (text: string) => void,
): Promise<void> {
  const res = await fetch("/api/ask", {
    method: "POST",
    headers: headers(key),
    body: JSON.stringify({ arxiv_id: id, block_id: blockId, question, history, graph }),
  }).catch(() => null);
  if (!res?.body) throw new LearnError("Can't reach the Paperforge server.");
  if (!res.ok) throw new LearnError(((await res.json().catch(() => ({}))) as { error?: string }).error ?? "Couldn't ask that.");
  for await (const e of lines<AskEvent>(res)) {
    if (e.type === "delta") onDelta(e.text);
    else if (e.type === "error") throw new LearnError(e.message, e.code);
  }
}

// ---------- request bookkeeping ----------

/**
 * The ask history to send: the server takes at most 10 turns of 4000 chars, so send the
 * recent window (starting on a question) and clip long answers, instead of failing forever.
 */
export function askHistoryPayload(turns: AskTurn[], maxTurns = 10, maxChars = 4000): AskTurn[] {
  let recent = turns.slice(-maxTurns);
  if (recent[0]?.role === "assistant") recent = recent.slice(1);
  return recent.map((t) => ({ role: t.role, text: t.text.slice(0, maxChars) }));
}

/** Which block the inspector shows: hover previews, except while an Ask is being typed or answered. */
export function inspectTarget(hovered: string | null, pinned: string | null, askBusy: boolean): string | null {
  return askBusy ? pinned : (hovered ?? pinned);
}

/**
 * At most one request per kind in flight (a repeat click reuses it, so it never pays twice),
 * and only the most recently requested kind is "current" when its result arrives.
 */
export class InFlight<T> {
  private pending = new Map<string, Promise<T>>();
  private latest: string | null = null;

  get busy() {
    return this.pending.size > 0;
  }

  async run(key: string, work: () => Promise<T>): Promise<{ value: T; current: boolean }> {
    this.latest = key;
    let p = this.pending.get(key);
    if (!p) {
      const q: Promise<T> = work().finally(() => {
        if (this.pending.get(key) === q) this.pending.delete(key);
      });
      this.pending.set(key, (p = q));
    }
    const value = await p;
    return { value, current: this.latest === key };
  }

  /** Forget everything (a new paper): late results come back as not current. */
  reset() {
    this.pending.clear();
    this.latest = null;
  }
}
