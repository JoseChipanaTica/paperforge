import { expect, test } from "bun:test";
import { ndjson } from "./stream";

test("ndjson sends blank lines while the work is quiet, then the events", async () => {
  const res = ndjson<{ type: string }>(async (emit) => {
    await Bun.sleep(35);
    emit({ type: "done" });
  }, 10);
  expect(res.headers.get("Content-Type")).toBe("application/x-ndjson");
  const text = await res.text();
  expect(text.startsWith("\n\n")).toBe(true);
  expect(text.split("\n").filter((l) => l.trim()).map((l) => JSON.parse(l))).toEqual([{ type: "done" }]);
});

test("ndjson closes the stream when the work throws", async () => {
  const res = ndjson(async () => {
    throw new Error("boom");
  }, 10);
  expect(await res.text()).toBe("");
});
