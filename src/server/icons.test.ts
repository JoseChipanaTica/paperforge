import { expect, test } from "bun:test";
import { ICON_FILES, iconHeaders, iconRoutes } from "./icons";

test("every icon is served at the site root, linkable from other sites", async () => {
  expect(ICON_FILES).toEqual(["favicon.ico", "icon.svg", "app-icon.svg", "apple-touch-icon.png", "icon-192.png", "icon-512.png"]);
  for (const name of ICON_FILES) {
    const res = await iconRoutes[`/${name}`]!();
    expect(res.status).toBe(200);
    expect((await res.arrayBuffer()).byteLength).toBeGreaterThan(100);
    expect(res.headers.get("Access-Control-Allow-Origin")).toBe("*");
    expect(res.headers.get("Cache-Control")).toBe("public, max-age=86400");
  }
});

test("content types", async () => {
  expect((await iconRoutes["/icon.svg"]!()).headers.get("Content-Type")).toStartWith("image/svg+xml");
  expect((await iconRoutes["/icon-512.png"]!()).headers.get("Content-Type")).toBe("image/png");
  expect((await iconRoutes["/favicon.ico"]!()).headers.get("Content-Type")).toBe("image/x-icon");
});

test("favicon.ico holds 16, 32 and 48 px images", async () => {
  const ico = new DataView(await (await iconRoutes["/favicon.ico"]!()).arrayBuffer());
  expect(ico.getUint16(2, true)).toBe(1); // icon
  expect(ico.getUint16(4, true)).toBe(3);
  expect([0, 1, 2].map((i) => ico.getUint8(6 + 16 * i))).toEqual([16, 32, 48]);
});

test("icon headers: short cache (unhashed names), open to other origins", () => {
  expect(iconHeaders("x.png")).toEqual({ "Cache-Control": "public, max-age=86400", "Access-Control-Allow-Origin": "*" });
});
