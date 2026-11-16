import { expect, test } from "bun:test";
import { currentModel, withModel } from "./model";
import { ndjson } from "./stream";

const req = (model?: string) => new Request("http://x/api", { headers: model ? { "x-anthropic-model": model } : {} });

test("withModel uses the visitor's pick, including inside a streamed response", async () => {
  expect(withModel(req("claude-sonnet-5-5"), currentModel)).toBe("claude-sonnet-5-5");
  const res = withModel(req("claude-haiku-5-5"), () =>
    ndjson<{ model: string }>(async (emit) => {
      await Bun.sleep(5);
      emit({ model: currentModel() });
    }),
  );
  expect(JSON.parse((await res.text()).trim())).toEqual({ model: "claude-haiku-5-5" });
});

test("withModel ignores a model that isn't offered", () => {
  const fallback = currentModel();
  expect(withModel(req("claude-opus-4-1"), currentModel)).toBe(fallback);
  expect(withModel(req(), currentModel)).toBe(fallback);
});
