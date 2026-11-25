/**
 * Record the assembly as a video: a capture of this tab, cropped to the stage and study panel, so the labels,
 * equations and the paper's highlight sweep (all DOM) are in it. Chromium only (Region Capture).
 */

// Region Capture isn't in the DOM typings yet.
declare const CropTarget: { fromElement(el: Element): Promise<unknown> };
type CroppableTrack = MediaStreamTrack & { cropTo?(target: unknown): Promise<void> };

const MIMES = ["video/mp4;codecs=avc1", "video/webm;codecs=vp9", "video/webm"];

export function canRecord(): boolean {
  return (
    typeof navigator !== "undefined" &&
    typeof navigator.mediaDevices?.getDisplayMedia === "function" &&
    "CropTarget" in globalThis &&
    "MediaRecorder" in globalThis
  );
}

/** The best format this browser records: MP4 posts everywhere, WebM is the fallback. */
export function pickMime(isTypeSupported: (type: string) => boolean): string {
  return MIMES.find((m) => isTypeSupported(m)) ?? "video/webm";
}

export function fileName(arxivId: string, mime: string): string {
  return `paperforge-${arxivId}.${mime.startsWith("video/mp4") ? "mp4" : "webm"}`;
}

export type RecordResult = "saved" | "cancelled" | "denied";

/**
 * Capture `el` while `play` runs; download the video when it finishes. Aborting `signal`, or the visitor ending the
 * share from the browser's bar, discards it. Call from the click itself: the capture prompt needs the user's gesture.
 */
export async function record(
  el: HTMLElement,
  play: () => Promise<void>,
  name: string,
  signal: AbortSignal,
  deps: { media?: MediaDevices; download?: (blob: Blob, name: string) => void } = {},
): Promise<RecordResult> {
  let stream: MediaStream;
  try {
    stream = await (deps.media ?? navigator.mediaDevices).getDisplayMedia({
      video: { frameRate: 60 },
      audio: false,
      preferCurrentTab: true,
    } as DisplayMediaStreamOptions);
  } catch {
    return "denied";
  }
  const track = stream.getVideoTracks()[0] as CroppableTrack | undefined;
  try {
    // Without the crop the whole tab is recorded, which still beats no video.
    try {
      if (track?.cropTo) await track.cropTo(await CropTarget.fromElement(el));
    } catch {}
    const mime = pickMime((t) => MediaRecorder.isTypeSupported(t));
    const recorder = new MediaRecorder(stream, { mimeType: mime });
    const chunks: Blob[] = [];
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    const stopped = new Promise<void>((resolve) => (recorder.onstop = () => resolve()));
    recorder.start(1000);
    const outcome = await new Promise<"done" | "cancelled">((resolve) => {
      if (signal.aborted) return resolve("cancelled");
      signal.addEventListener("abort", () => resolve("cancelled"), { once: true });
      track?.addEventListener("ended", () => resolve("cancelled"), { once: true });
      play().then(
        () => resolve("done"),
        () => resolve("cancelled"),
      );
    });
    if (recorder.state !== "inactive") recorder.stop();
    await stopped;
    if (outcome === "cancelled") return "cancelled";
    (deps.download ?? download)(new Blob(chunks, { type: mime }), fileName(name, mime));
    return "saved";
  } finally {
    for (const t of stream.getTracks()) t.stop();
  }
}

function download(blob: Blob, name: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
