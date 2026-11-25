import { join } from "node:path";
import { ICON_FILES, iconHeaders } from "./src/server/icons";
import { apiRoutes } from "./src/server/routes";

// Vercel entry (Bun preset): the API, plus the page built into `public/` by `bun run build`
// (HTML imports aren't supported there). Hashed assets are immutable, so the CDN keeps them.
const PUBLIC = join(import.meta.dir, "public");

Bun.serve({
  idleTimeout: 255, // Claude can take a couple of minutes on a long paper
  routes: apiRoutes,
  async fetch(req) {
    const path = new URL(req.url).pathname;
    const name = path === "/" ? "index.html" : path.slice(1);
    const file = Bun.file(join(PUBLIC, name));
    if (name.includes("/") || name.startsWith(".") || !(await file.exists())) {
      return new Response("Not found", { status: 404 });
    }
    // The icons keep stable names (other sites link them), so they can't be cached as immutable.
    if (ICON_FILES.includes(name)) return new Response(file, { headers: iconHeaders(name) });
    const cache = name === "index.html" ? "public, max-age=0, s-maxage=60" : "public, max-age=31536000, immutable";
    return new Response(file, { headers: { "Cache-Control": cache } });
  },
});
