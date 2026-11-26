import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";

// Processed scenes live next to the viewer (../scenes/<name>/scene.json, points.ply, splat.ply).
const SCENES = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../scenes");
const TYPES = {
  ".json": "application/json; charset=utf-8",
  ".ply": "application/octet-stream",
  ".spz": "application/octet-stream",
  ".svg": "image/svg+xml",
  ".jpg": "image/jpeg",
};

function listScenes() {
  if (!fs.existsSync(SCENES)) return [];
  return fs
    .readdirSync(SCENES, { withFileTypes: true })
    .filter((d) => d.isDirectory() && fs.existsSync(path.join(SCENES, d.name, "scene.json")))
    .map((d) => {
      try {
        const s = JSON.parse(fs.readFileSync(path.join(SCENES, d.name, "scene.json"), "utf8"));
        return { name: d.name, room: s.room, splat: Boolean(s.files?.splat), created: s.created };
      } catch {
        return null;
      }
    })
    .filter(Boolean)
    .sort((a, b) => String(b.created).localeCompare(String(a.created)));
}

function scenes(req, res, next) {
  const url = decodeURIComponent((req.url || "").split("?")[0]);
  if (!url.startsWith("/scenes/")) return next();
  const rel = url.slice("/scenes/".length);
  res.setHeader("Cache-Control", "no-store");
  if (rel === "") {
    res.setHeader("Content-Type", TYPES[".json"]);
    res.end(JSON.stringify(listScenes()));
    return;
  }
  // only <scene>/<file> at the top of a scene folder (never the work/ internals)
  const parts = rel.split("/");
  const file = path.resolve(SCENES, rel);
  if (parts.length !== 2 || !file.startsWith(SCENES + path.sep) || !fs.existsSync(file) || !fs.statSync(file).isFile()) {
    res.statusCode = 404;
    res.end("not found");
    return;
  }
  res.setHeader("Content-Type", TYPES[path.extname(file).toLowerCase()] || "application/octet-stream");
  fs.createReadStream(file).pipe(res);
}

export default defineConfig({
  plugins: [
    {
      name: "spacecapture-scenes",
      // block bodies on purpose: a function returned from these hooks would run as a post hook
      configureServer(server) {
        server.middlewares.use(scenes);
      },
      configurePreviewServer(server) {
        server.middlewares.use(scenes);
      },
    },
  ],
  server: { port: 5173, proxy: { "/api": "http://127.0.0.1:8765" } },
});
