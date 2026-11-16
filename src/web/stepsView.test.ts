import { describe, expect, test } from "bun:test";
import type { View } from "../shared/learn";
import type { LocatedArchitecture } from "../shared/graph";
import { stepHtml, stepsClick, stepsHtml } from "./stepsView";
import type { TourDriver, TourScript, TourStep } from "./tour";

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

describe("the outro", () => {
  const script: TourScript = {
    analogy: { title: "A", text: "a" },
    intro: { title: "Walkthrough", body: "in", label: "x", views: [] },
    steps: [step],
    output: { label: "y", views: [] },
    outro: "That's the model.",
  };
  const graph = { blocks: [] } as unknown as LocatedArchitecture;

  test("the outro shows the continue button only when onContinue is given", () => {
    expect(stepsHtml(script, graph, null)).not.toContain("data-continue");
    const html = stepsHtml(script, graph, null, "Train on this example →");
    const outro = html.slice(html.indexOf("step-outro"));
    expect(outro).toContain("data-continue");
    expect(outro).toContain("Train on this example →");
  });

  test("the input's note says what the numbers are: illustrative unless the script says otherwise", () => {
    expect(stepsHtml(script, graph, null)).toContain("Values are illustrative");
    const computed = stepsHtml({ ...script, intro: { ...script.intro, note: "Computed in your browser on a toy model" } }, graph, null);
    expect(computed).toContain("Computed in your browser on a toy model");
    expect(computed).not.toContain("Values are illustrative");
  });

  test("clicking it calls run", () => {
    let ran = 0;
    const calls: string[] = [];
    const driver = { next: () => calls.push("next"), back: () => calls.push("back"), togglePlay: () => calls.push("play"), goTo: () => calls.push("go") } as unknown as TourDriver;
    // A click on the button's label: closest() finds the button itself.
    const target = { closest: (sel: string) => (sel === "[data-continue]" ? { dataset: {} } : null) } as unknown as HTMLElement;
    stepsClick(target, script, driver, () => {}, { label: "Train", run: () => ran++ });
    expect(ran).toBe(1);
    expect(calls).toEqual([]);
    stepsClick(target, script, driver, () => {}, undefined); // no button, nothing to run
    expect(ran).toBe(1);
  });
});
