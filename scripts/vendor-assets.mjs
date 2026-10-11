/**
 * Vendors the browser libraries the studio UI depends on so the app never
 * fetches anything from a CDN at runtime (works offline / behind a firewall).
 *
 * Run with:  npm run vendor
 */
import { mkdir, writeFile } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const projectRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const outputDir = resolve(projectRoot, "app/static/vendor");

// `markers` are the global symbols each bundle must expose — a far more
// reliable validity check than guessing at a byte size.
const ASSETS = [
  {
    name: "alpine.min.js",
    url: "https://cdn.jsdelivr.net/npm/alpinejs@3/dist/cdn.min.js",
    markers: ["Alpine"],
  },
  {
    name: "lucide.min.js",
    url: "https://cdn.jsdelivr.net/npm/lucide@latest/dist/umd/lucide.min.js",
    markers: ["createIcons", "createElement"],
  },
  {
    name: "wavesurfer.min.js",
    url: "https://cdn.jsdelivr.net/npm/wavesurfer.js@7/dist/wavesurfer.min.js",
    markers: ["WaveSurfer"],
  },
];

const MIN_BYTES = 10_000;

async function fetchAsset({ name, url, markers }) {
  const response = await fetch(url, { redirect: "follow" });
  if (!response.ok) {
    throw new Error(`${name}: HTTP ${response.status} for ${url}`);
  }
  const body = await response.text();
  if (body.length < MIN_BYTES) {
    throw new Error(
      `${name}: response was only ${body.length} bytes; the CDN probably returned an error page.`,
    );
  }
  if (/^\s*<(!doctype|html)/i.test(body)) {
    throw new Error(`${name}: received HTML instead of JavaScript.`);
  }
  const missing = markers.filter((marker) => !body.includes(marker));
  if (missing.length) {
    throw new Error(
      `${name}: bundle is missing expected symbol(s) ${missing.join(", ")} — refusing to vendor it.`,
    );
  }
  const destination = resolve(outputDir, name);
  await writeFile(destination, body, "utf8");
  const version = response.headers.get("x-jsd-version") || "unknown";
  console.log(`  ${name.padEnd(20)} ${(body.length / 1024).toFixed(0).padStart(4)} KB  (version ${version})`);
}

async function main() {
  await mkdir(outputDir, { recursive: true });
  console.log("Vendoring browser libraries into app/static/vendor …");
  for (const asset of ASSETS) {
    await fetchAsset(asset);
  }
  console.log("Done.");
}

main().catch((error) => {
  console.error(`Vendoring failed: ${error.message}`);
  process.exitCode = 1;
});
