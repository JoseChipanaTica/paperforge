import { join } from "node:path";

/**
 * The Paperforge icons, served at the site root under stable names so other sites can link them
 * (`/favicon.ico`, `/icon.svg`, `/apple-touch-icon.png`…). Sources live in `src/web/icons/`;
 * `bun run build` copies them into `public/` for Vercel.
 */
export const ICON_FILES = ["favicon.ico", "icon.svg", "app-icon.svg", "apple-touch-icon.png", "icon-192.png", "icon-512.png"];

const ICON_DIR = join(import.meta.dir, "../web/icons");

/** Names aren't hashed, so a day's cache; open to any origin, since other sites embed them. */
export function iconHeaders(_name: string): Record<string, string> {
  return { "Cache-Control": "public, max-age=86400", "Access-Control-Allow-Origin": "*" };
}

const TYPES: Record<string, string> = { ico: "image/x-icon", svg: "image/svg+xml", png: "image/png" };

export const iconRoutes: Record<string, () => Response> = Object.fromEntries(
  ICON_FILES.map((name) => [
    `/${name}`,
    () =>
      new Response(Bun.file(join(ICON_DIR, name)), {
        headers: { ...iconHeaders(name), "Content-Type": TYPES[name.split(".").pop()!]! },
      }),
  ]),
);
