import { describe, expect, test } from "bun:test";
import { SVG_MAX, sanitizeSvg } from "./svg";

const wrap = (inner: string) => `<svg viewBox="0 0 360 220">${inner}</svg>`;

describe("sanitizeSvg", () => {
  test("keeps shapes, text and SMIL animation", () => {
    const out = sanitizeSvg(
      wrap(
        '<rect x="1" y="2" width="3" height="4" fill="#fff"><animate attributeName="x" from="0" to="10" dur="1s" repeatCount="indefinite"/></rect><text x="5" y="9">a &lt; b</text>',
      ),
    );
    expect(out).toContain('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 360 220">');
    expect(out).toContain('<rect x="1" y="2" width="3" height="4" fill="#fff">');
    expect(out).toContain('<animate attributeName="x" from="0" to="10" dur="1s" repeatCount="indefinite"/>');
    expect(out).toContain("a &lt; b</text>");
    expect(out.endsWith("</svg>")).toBe(true);
  });

  test("drops scripts and their content", () => {
    const out = sanitizeSvg(wrap("<script>alert(1)</script><rect/>"));
    expect(out).not.toContain("script");
    expect(out).not.toContain("alert");
    expect(out).toContain("<rect/>");
  });

  test("drops event handlers and javascript: urls, in any case", () => {
    const out = sanitizeSvg(`<svg OnLoad="alert(1)"><rect onclick="x()" fill="red"/><animate href="JavaScript:alert(1)"/></svg>`);
    expect(out.toLowerCase()).not.toContain("onload");
    expect(out.toLowerCase()).not.toContain("onclick");
    expect(out.toLowerCase()).not.toContain("javascript");
    expect(out).toContain('fill="red"');
  });

  test("keeps internal references only", () => {
    expect(sanitizeSvg(wrap('<animate attributeName="x" href="#a"/>'))).toContain('href="#a"');
    expect(sanitizeSvg(wrap('<linearGradient href="#g" id="h"/>'))).toContain('href="#g"');
    expect(sanitizeSvg(wrap('<animate attributeName="x" href="https://x"/>'))).not.toContain("https");
    expect(sanitizeSvg(wrap('<animate attributeName="x" xlink:href="data:text/html,x"/>'))).not.toContain("data:");
  });

  test("strips url() and expression() except internal url(#id)", () => {
    expect(sanitizeSvg(wrap('<rect style="fill:url(https://x)"/>'))).not.toContain("https");
    expect(sanitizeSvg(wrap('<rect fill="url(https://x)"/>'))).not.toContain("https");
    expect(sanitizeSvg(wrap('<rect style="width:expression(alert(1))"/>'))).not.toContain("expression");
    expect(sanitizeSvg(wrap('<rect fill="url(#g)"/>'))).toContain('fill="url(#g)"');
  });

  test("drops foreignObject, iframe, image and unknown elements with their subtrees", () => {
    const out = sanitizeSvg(
      wrap('<foreignObject><div>hi</div></foreignObject><image href="#a"/><iframe src="x"></iframe><blink><rect/></blink><circle r="1"/>'),
    );
    for (const bad of ["foreignObject", "div", "hi", "image", "iframe", "blink"]) expect(out).not.toContain(bad);
    expect(out).toContain('<circle r="1"/>');
    expect(out).not.toContain("<rect/>"); // inside the dropped <blink>
  });

  test("escapes attribute values and text it rebuilds", () => {
    const out = sanitizeSvg(wrap('<text x="1&quot; onload=&quot;x">a<b</text>'));
    expect(out).toBe("");
    const ok = sanitizeSvg(wrap('<text font-family="a&quot;b">x &amp; y</text>'));
    expect(ok).toContain('font-family="a&quot;b"');
    expect(ok).toContain("x &amp; y");
  });

  test("rejects oversize and unparseable input", () => {
    expect(sanitizeSvg(wrap(`<rect/>`.repeat(SVG_MAX / 6)))).toBe("");
    expect(sanitizeSvg("<svg><rect")).toBe("");
    expect(sanitizeSvg("<svg><g></svg>")).toBe("");
    expect(sanitizeSvg("<g><rect/></g>")).toBe("");
    expect(sanitizeSvg("")).toBe("");
  });

  test("drops comments, CDATA, doctype and processing instructions", () => {
    const out = sanitizeSvg(`<?xml version="1.0"?><!DOCTYPE svg><svg><!-- note --><![CDATA[<script>x</script>]]><rect/></svg>`);
    expect(out).toBe('<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>');
  });
});

describe("sanitizeSvg against the review's attacks", () => {
  test("drops style entirely (image-set beacons, page overlays)", () => {
    const out = sanitizeSvg(`<svg style="background-image:image-set('https://evil/x' 1x)"><rect style="position:fixed;inset:0"/></svg>`);
    expect(out).not.toContain("style");
    expect(out).not.toContain("evil");
  });

  test("CSS escapes can't smuggle url() through presentation attributes or SMIL", () => {
    const out = sanitizeSvg(wrap('<rect fill="u\\72 l(https://evil/x)"/><set attributeName="fill" to="u\\72 l(https://evil/x)"/>'));
    expect(out).not.toContain("evil");
    expect(out).not.toContain("\\");
  });

  test("functions other than url(#id) and transforms are refused", () => {
    expect(sanitizeSvg(wrap('<rect fill="image(https://evil/x)"/>'))).not.toContain("evil");
    expect(sanitizeSvg(wrap('<g transform="translate(10, 20) rotate(45)"/>'))).toContain('transform="translate(10, 20) rotate(45)"');
    expect(sanitizeSvg(wrap('<g transform="translate(10) evil(1)"/>'))).not.toContain("transform");
  });

  test("SMIL can't animate href, style or ids", () => {
    for (const name of ["href", "xlink:href", "style", "id"]) {
      const out = sanitizeSvg(wrap(`<linearGradient id="g"><set attributeName="${name}" to="https://evil/a.svg#g"/></linearGradient>`));
      expect(out).not.toContain("evil");
      expect(out).not.toContain(`attributeName="${name}"`);
    }
  });

  test("the root always gets the SVG namespace, whatever the input says", () => {
    expect(sanitizeSvg("<svg><rect/></svg>")).toBe('<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>');
    expect(sanitizeSvg('<svg xmlns="http://evil"><rect/></svg>')).toBe('<svg xmlns="http://www.w3.org/2000/svg"><rect/></svg>');
  });

  test("duplicate attributes (href and xlink:href) keep the first, so the output stays valid XML", () => {
    const out = sanitizeSvg(wrap('<animate attributeName="x" href="#a" xlink:href="#b"/>'));
    expect(out.match(/href=/g)?.length).toBe(1);
  });

  test("control characters and out-of-range entities are dropped, never thrown", () => {
    expect(() => sanitizeSvg(wrap("<text>a&#99999999;b&#0;c</text>"))).not.toThrow();
    const out = sanitizeSvg(wrap("<text>a&#99999999;b&#0;c\u0001d</text>"));
    expect(out).toContain(">abcd</text>");
  });
});
