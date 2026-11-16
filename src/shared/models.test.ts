import { expect, test } from "bun:test";
import { DEFAULT_MODEL, isModel, MODELS, modelName } from "./models";

test("the default model is one visitors can pick", () => {
  expect(isModel(DEFAULT_MODEL)).toBe(true);
  expect(MODELS[0].id as string).toBe(DEFAULT_MODEL); // listed first in the picker
});

test("isModel accepts only listed ids", () => {
  expect(isModel("claude-sonnet-5-5")).toBe(true);
  expect(isModel("claude-opus-4-1")).toBe(false);
  expect(isModel("")).toBe(false);
  expect(isModel(undefined)).toBe(false);
});

test("modelName prints a model id the way visitors read it", () => {
  expect(modelName("claude-opus-5-5")).toBe("Claude Opus 5.5");
  expect(modelName("claude-fable-5-1")).toBe("Claude Fable 5.1");
  expect(modelName("claude-opus-5")).toBe("Claude Opus 5");
  expect(modelName("claude-haiku-4-5-20251001")).toBe("Claude Haiku 4.5");
  expect(modelName("some-other-model")).toBe("some-other-model");
});
