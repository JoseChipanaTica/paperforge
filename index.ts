import index from "./src/web/index.html";
import { iconRoutes } from "./src/server/icons";
import { apiRoutes } from "./src/server/routes";

// Local entry: the page through Bun's HTML import, plus the API. On Vercel, `server.ts` serves
// the API and the page is built into `public/` (HTML imports aren't supported there).
const server = Bun.serve({
  port: Number(process.env.PORT ?? 3000),
  idleTimeout: 255, // Claude can take a couple of minutes on a long paper
  routes: { "/": index, ...iconRoutes, ...apiRoutes },
  development: process.env.NODE_ENV !== "production" && { hmr: true, console: true },
});

console.log(`Paperforge running at ${server.url}`);
