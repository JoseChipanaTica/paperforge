import { describe, expect, test } from "bun:test";
import type { View } from "../shared/learn";
import { stepHtml } from "./stepsView";
import type { TourStep } from "./tour";

const view = (caption: string): View => ({ kind: "tensor", caption, rows: [], cols: [], values: [], dims: "[3, 512]", split: 1 });
const step: TourStep = {
  block_id: "encoder_self_attention",
  title: "",
  what: "Every word looks at every other word.",
  body: "Scores <script>alert(1)</script> become weights.",
  shape: "[B, 3, 512] → [B, 3, 512]",
  views: [view("attention weights for “sat”"), view("values")],
};

describe("stepHtml", () => {
  const html = stepHtml(step, null);

  test("leads with what happens, in bold", () => {
    expect(html).toContain("<b>Every word looks at every other word.</b>");
  });

  test("makes each dimension of the shape traceable", () => {
    expect(html).toContain('data-dim="512"');
    expect(html).toContain('class="dim"');
  });

  test("marks the traced dimension", () => {
    expect(stepHtml(step, "512")).toContain('aria-pressed="true"');
  });

  test("lists every view caption", () => {
    expect(html).toContain("attention weights for “sat”");
    expect(html).toContain("<li>values</li>");
  });

  test("escapes the narration", () => {
    expect(html).not.toContain("<script>");
    expect(html).toContain("&lt;script&gt;");
  });
});
