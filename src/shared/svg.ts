/**
 * The only way a Claude-written SVG reaches the page. It never passes markup through: it
 * tokenizes the input and rebuilds it from allowlisted elements and attributes, re-escaping every
 * value. No DOMParser, so the server (Bun) and the browser run the same code.
 */

export const SVG_MAX = 30_000;

const ELEMENTS = new Set(
  "svg g rect circle ellipse line polyline polygon path text tspan defs linearGradient radialGradient stop clipPath mask animate animateTransform animateMotion set title desc".split(
    " ",
  ),
);

const ATTRIBUTES = new Set(
  "x y x1 y1 x2 y2 cx cy r rx ry fx fy width height d points transform fill stroke stroke-width stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin opacity fill-opacity stroke-opacity font-size font-family font-weight font-style text-anchor dominant-baseline letter-spacing viewBox preserveAspectRatio id offset stop-color stop-opacity gradientUnits gradientTransform clip-path clipPathUnits mask attributeName attributeType from to by values keyTimes keySplines keyPoints calcMode dur begin end repeatCount restart fill-rule clip-rule path rotate type additive accumulate href".split(
    " ",
  ),
);

const TOKEN = /<\/?([A-Za-z][\w:.-]*)((?:\s+[^\s=/>]+(?:\s*=\s*(?:"[^"]*"|'[^']*'))?)*)\s*(\/?)>|([^<]+)|(<)/g;
const ATTRIBUTE = /([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'))?/g;

const SVG_NS = "http://www.w3.org/2000/svg";
const TARGETED = new Set(["animate", "set", "animateTransform"]);
// Attributes SMIL may animate: geometry and paint, never references, styles or ids.
const ANIMATABLE = new Set([...ATTRIBUTES].filter((a) => !["href", "id", "attributeName", "begin", "end"].includes(a)));
// Transform lists are the only place a function other than url(#id) may appear.
const TRANSFORM = /^(\s*(matrix|translate|scale|rotate|skewX|skewY)\(\s*[-+\d.eE,\s]*\)\s*,?)+\s*$/;

/** Code points XML 1.0 allows (no NUL or other control characters). */
const allowed = (c: number) => c === 9 || c === 10 || c === 13 || (c >= 0x20 && c <= 0xd7ff) || (c >= 0xe000 && c <= 0xfffd) || (c >= 0x10000 && c <= 0x10ffff);
const char = (c: number) => (allowed(c) ? String.fromCodePoint(c) : "");

const decode = (s: string) =>
  s
    .replace(/&(#x[\da-f]+|#\d+|quot|amp|lt|gt|apos);/gi, (_, e: string) => {
      const k = e.toLowerCase();
      if (k.startsWith("#x")) return char(parseInt(k.slice(2), 16));
      if (k.startsWith("#")) return char(Number(k.slice(1)));
      return ({ quot: '"', amp: "&", lt: "<", gt: ">", apos: "'" } as Record<string, string>)[k]!;
    })
    .replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f\ufffe\uffff]/g, "");
const escape = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** An attribute worth keeping, with its value decoded, or null. */
function keep(rawName: string, value: string): [string, string] | null {
  const name = rawName === "xlink:href" ? "href" : rawName;
  if (/^on/i.test(name) || !ATTRIBUTES.has(name)) return null;
  const v = decode(value);
  // CSS escapes (u\72 l) and comments could rebuild anything a later check looks for.
  if (/[\\]|\/\*/.test(v)) return null;
  if (/javascript:|data:|@import/i.test(v.replace(/\s+/g, ""))) return null;
  if (name === "href" && !v.trim().startsWith("#")) return null;
  if (name === "attributeName" && !ANIMATABLE.has(v.trim())) return null;
  // No functions (image-set, image, expression…) except url(#id) and, on transforms, transform lists.
  if (v.includes("(")) {
    const transformList = (name === "transform" || name === "gradientTransform" || name === "type") && TRANSFORM.test(v);
    const onlyLocalUrls = v.replace(/url\(\s*#[\w-]+\s*\)/g, "").indexOf("(") === -1;
    if (!transformList && !onlyLocalUrls) return null;
  }
  return [name, v];
}

export function sanitizeSvg(input: string): string {
  if (!input || input.length > SVG_MAX) return "";
  const src = input.replace(/<\?[\s\S]*?\?>|<!--[\s\S]*?-->|<!\[CDATA\[[\s\S]*?\]\]>|<!DOCTYPE[^>]*>/gi, "");
  const stack: { name: string; kept: boolean }[] = [];
  let out = "";
  let rooted = false;
  let closed = false;
  for (const m of src.matchAll(TOKEN)) {
    const [whole, name, attrs = "", selfClose, text, stray] = m;
    if (stray) return ""; // a "<" that starts no tag: not well-formed
    if (text !== undefined) {
      if (!stack.length) {
        if (text.trim()) return "";
        continue;
      }
      if (stack.at(-1)!.kept) out += escape(decode(text));
      continue;
    }
    if (closed) return ""; // anything after the root closed
    if (whole.startsWith("</")) {
      const top = stack.pop();
      if (!top || top.name !== name) return "";
      if (top.kept) out += `</${name}>`;
      if (!stack.length) closed = true;
      continue;
    }
    if (!stack.length) {
      if (rooted || name !== "svg") return "";
      rooted = true;
    }
    let kept = ELEMENTS.has(name!) && (stack.length === 0 || stack.at(-1)!.kept);
    if (kept) {
      // The root is always an SVG in the SVG namespace, whatever the input declared.
      let a = stack.length === 0 ? ` xmlns="${SVG_NS}"` : "";
      const seen = new Set<string>();
      for (const at of attrs.matchAll(ATTRIBUTE)) {
        const k = keep(at[1]!, at[2] ?? at[3] ?? "");
        if (k && !seen.has(k[0])) {
          seen.add(k[0]); // href and xlink:href both become href: keep the first
          a += ` ${k[0]}="${escape(k[1])}"`;
        }
      }
      // An animation whose target attribute was refused (href, style, id…) is dropped whole.
      if (TARGETED.has(name!) && !seen.has("attributeName")) kept = false;
      else out += `<${name}${a}${selfClose ? "/" : ""}>`;
    }
    if (!selfClose) stack.push({ name: name!, kept });
    else if (!stack.length) closed = true;
  }
  return rooted && closed && !stack.length ? out : "";
}
