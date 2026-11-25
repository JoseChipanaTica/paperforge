import { afterEach, beforeEach, expect, test } from "bun:test";
import { fileName, pickMime, record } from "./recorder";

test("pickMime prefers mp4", () => {
  expect(pickMime(() => true)).toBe("video/mp4;codecs=avc1");
  expect(pickMime((t) => t.startsWith("video/webm"))).toBe("video/webm;codecs=vp9");
  expect(pickMime(() => false)).toBe("video/webm");
});

test("fileName", () => {
  expect(fileName("1706.03762", "video/mp4;codecs=avc1")).toBe("paperforge-1706.03762.mp4");
  expect(fileName("1706.03762", "video/webm;codecs=vp9")).toBe("paperforge-1706.03762.webm");
});

// A tab capture that records nothing real: a track that can end, a recorder that emits one chunk on stop.
class FakeTrack extends EventTarget {
  stopped = false;
  stop() {
    this.stopped = true;
  }
  async cropTo() {}
}
class FakeRecorder {
  static isTypeSupported = () => true;
  state = "inactive";
  ondataavailable: ((e: { data: Blob }) => void) | null = null;
  onstop: (() => void) | null = null;
  start() {
    this.state = "recording";
  }
  stop() {
    this.state = "inactive";
    this.ondataavailable?.({ data: new Blob(["frames"]) });
    this.onstop?.();
  }
}
const saved = globalThis as Record<string, unknown>;
const originals = { MediaRecorder: saved.MediaRecorder, CropTarget: saved.CropTarget };
beforeEach(() => {
  saved.MediaRecorder = FakeRecorder;
  saved.CropTarget = { fromElement: async () => ({}) };
});
afterEach(() => Object.assign(saved, originals));

const capture = (track: FakeTrack) =>
  ({
    getDisplayMedia: async () => ({ getVideoTracks: () => [track], getTracks: () => [track] }),
  }) as unknown as MediaDevices;
const el = {} as HTMLElement;

test("a finished play downloads the video", async () => {
  const track = new FakeTrack();
  const downloads: string[] = [];
  const result = await record(el, async () => {}, "1706.03762", new AbortController().signal, {
    media: capture(track),
    download: (_, name) => downloads.push(name),
  });
  expect(result).toBe("saved");
  expect(downloads).toEqual(["paperforge-1706.03762.mp4"]);
  expect(track.stopped).toBe(true);
});

test("a denied prompt records nothing", async () => {
  let played = false;
  const downloads: string[] = [];
  const media = { getDisplayMedia: () => Promise.reject(new DOMException("denied", "NotAllowedError")) } as unknown as MediaDevices;
  const result = await record(el, async () => void (played = true), "x", new AbortController().signal, {
    media,
    download: (_, name) => downloads.push(name),
  });
  expect(result).toBe("denied");
  expect(played).toBe(false);
  expect(downloads).toEqual([]);
});

test("a stopped track discards", async () => {
  const track = new FakeTrack();
  const downloads: string[] = [];
  const result = await record(
    el,
    () => {
      setTimeout(() => track.dispatchEvent(new Event("ended")), 5);
      return new Promise(() => {}); // the build never gets to finish
    },
    "x",
    new AbortController().signal,
    { media: capture(track), download: (_, name) => downloads.push(name) },
  );
  expect(result).toBe("cancelled");
  expect(downloads).toEqual([]);
  expect(track.stopped).toBe(true);
});

test("an abort discards", async () => {
  const track = new FakeTrack();
  const ctrl = new AbortController();
  const downloads: string[] = [];
  const result = await record(
    el,
    () => {
      setTimeout(() => ctrl.abort(), 5);
      return new Promise(() => {});
    },
    "x",
    ctrl.signal,
    { media: capture(track), download: (_, name) => downloads.push(name) },
  );
  expect(result).toBe("cancelled");
  expect(downloads).toEqual([]);
});

test("a failed crop still records the whole tab", async () => {
  saved.CropTarget = { fromElement: () => Promise.reject(new Error("not croppable")) };
  const downloads: string[] = [];
  const result = await record(el, async () => {}, "x", new AbortController().signal, {
    media: capture(new FakeTrack()),
    download: (_, name) => downloads.push(name),
  });
  expect(result).toBe("saved");
  expect(downloads).toEqual(["paperforge-x.mp4"]);
});
