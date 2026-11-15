import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";

/** Extract plain text from a PDF, one string per page, with line breaks preserved. */
export async function extractPages(data: Uint8Array): Promise<string[]> {
  const doc = await getDocument({
    data,
    useSystemFonts: true,
    disableFontFace: true,
    verbosity: 0,
  }).promise;

  const pages: string[] = [];
  for (let i = 1; i <= doc.numPages; i++) {
    const page = await doc.getPage(i);
    const content = await page.getTextContent();
    let text = "";
    for (const item of content.items) {
      if (!("str" in item)) continue;
      text += item.str;
      if (item.hasEOL) text += "\n";
      else if (item.str && !item.str.endsWith(" ")) text += " ";
    }
    pages.push(normalize(text));
    page.cleanup();
  }
  await doc.cleanup();
  return pages;
}

function normalize(text: string): string {
  return text
    .replace(/\u00ad/g, "") // soft hyphens
    .replace(/[ \t]+/g, " ")
    .replace(/ *\n */g, "\n")
    .replace(/\n{3,}/g, "\n\n")
    .trim();
}
