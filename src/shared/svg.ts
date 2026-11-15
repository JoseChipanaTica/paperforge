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
  "x y x1 y1 x2 y2 cx cy r rx ry fx fy width height d points transform fill stroke stroke-width stroke-dasharray stroke-dashoffset stroke-linecap stroke-linejoin opacity fill-opacity stroke-opacity font-size font-family font-weight font-style text-anchor dominant-baseline letter-spacing viewBox preserveAspectRatio id offset stop-color stop-opacity gradientUnits gradientTransform clip-path clipPathUnits mask attributeName attributeType from to by values keyTimes keySplines keyPoints calcMode dur begin end repeatCount restart fill-rule clip-rule path rotate type additive accumulate style href xmlns".split(
    " ",
  ),
);

const TOKEN = /<\/?([A-Za-z][\w:.-]*)((?:\s+[^\s=/>]+(?:\s*=\s*(?:"[^"]*"|'[^']*'))?)*)\s*(\/?)>|([^<]+)|(<)/g;
const ATTRIBUTE = /([^\s=/>]+)(?:\s*=\s*(?:"([^"]*)"|'([^']*)'))?/g;

const decode = (s: string) =>
  s.replace(/&(#x[\da-f]+|#\d+|quot|amp|lt|gt|apos);/gi, (_, e: string) => {
    const k = e.toLowerCase();
    if (k.startsWith("#x")) return String.fromCodePoint(parseInt(k.slice(2), 16));
    if (k.startsWith("#")) return String.fromCodePoint(Number(k.slice(1)));
    return ({ quot: '"', amp: "&", lt: "<", gt: ">", apos: "'" } as Record<string, string>)[k]!;
  });
const escape = (s: string) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");

/** An attribute worth keeping, with its value decoded, or null. */
function keep(rawName: string, value: string): [string, string] | null {
  const name = rawName === "xlink:href" ? "href" : rawName;
  if (/^on/i.test(name) || !ATTRIBUTES.has(name)) return null;
  const v = decode(value);
  const squashed = v.replace(/\s+/g, "").toLowerCase();
  if (squashed.includes("javascript:") || squashed.includes("expression(") || squashed.includes("@import")) return null;
  if (name === "href" && !v.trim().startsWith("#")) return null;
  // url() only to something inside this SVG (a gradient, a clip path).
  if (/url\(\s*(?!['"]?#)/i.test(v)) return null;
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
    const kept = ELEMENTS.has(name!) && (stack.length === 0 || stack.at(-1)!.kept);
    if (kept) {
      let a = "";
      for (const at of attrs.matchAll(ATTRIBUTE)) {
        const k = keep(at[1]!, at[2] ?? at[3] ?? "");
        if (k) a += ` ${k[0]}="${escape(k[1])}"`;
      }
      out += `<${name}${a}${selfClose ? "/" : ""}>`;
    }
    if (!selfClose) stack.push({ name: name!, kept });
    else if (!stack.length) closed = true;
  }
  return rooted && closed && !stack.length ? out : "";
}
