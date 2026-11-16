/** How often a quiet stream sends a blank line, so proxies don't drop it while Claude works. */
export const HEARTBEAT_MS = 20_000;

/**
 * An NDJSON response fed by `run`. Clients skip blank lines, so while `run` is quiet the stream
 * sends one every `heartbeatMs`: a long Claude call never leaves the connection idle.
 */
export function ndjson<E>(run: (emit: (e: E) => void) => Promise<void>, heartbeatMs = HEARTBEAT_MS): Response {
  const encoder = new TextEncoder();
  const stream = new ReadableStream({
    async start(controller) {
      const send = (text: string) => {
        try {
          controller.enqueue(encoder.encode(text));
        } catch {} // the visitor left; the work finishes without them
      };
      const beat = setInterval(() => send("\n"), heartbeatMs);
      try {
        await run((e) => send(JSON.stringify(e) + "\n"));
      } finally {
        clearInterval(beat);
        try {
          controller.close();
        } catch {}
      }
    },
  });
  return new Response(stream, { headers: { "Content-Type": "application/x-ndjson" } });
}
