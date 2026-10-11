/**
 * Boot the real studio page in a DOM engine (jsdom) against a running server
 * and assert that it actually mounts.
 *
 * This is the check that matters most for this frontend. The Alpine CDN build
 * starts itself from a `queueMicrotask` that drains as soon as its own <script>
 * finishes — before the *next* deferred script runs. If studio.js were loaded
 * after Alpine, `x-data="studio()"` would evaluate against an undefined factory
 * and the page would render dead. No syntax check or unit test can catch that;
 * only executing the document can.
 *
 * Usage:
 *   node tests/boot_check.mjs [baseUrl]
 *   VOXSTUDIO_BASE_URL=http://127.0.0.1:8000 node tests/boot_check.mjs
 *
 * Exit codes: 0 = passed, 1 = failed, 3 = no server reachable (treated as a
 * skip by the pytest wrapper).
 */
import jsdom from "jsdom";

const { JSDOM, VirtualConsole } = jsdom;

const EXPLICIT_URL = process.argv[2] || process.env.VOXSTUDIO_BASE_URL || "";
const CANDIDATES = EXPLICIT_URL
  ? [EXPLICIT_URL]
  : ["http://127.0.0.1:49574", "http://127.0.0.1:8000", "http://127.0.0.1:8080"];

async function reachable(baseUrl) {
  try {
    const response = await fetch(`${baseUrl}/api/health`, { signal: AbortSignal.timeout(2500) });
    return response.ok;
  } catch {
    return false;
  }
}

let baseUrl = "";
for (const candidate of CANDIDATES) {
  if (await reachable(candidate)) {
    baseUrl = candidate;
    break;
  }
}

if (!baseUrl) {
  console.error(`SKIPPED: no VoxStudio server reachable at ${CANDIDATES.join(", ")}`);
  process.exit(3);
}

console.log(`Booting ${baseUrl} in jsdom…`);

const failures = [];
const consoleErrors = [];

function check(label, condition, actual) {
  if (condition) {
    console.log(`  ok    ${label}`);
  } else {
    failures.push(label);
    console.error(`  FAIL  ${label} — got ${JSON.stringify(actual)}`);
  }
}

const virtualConsole = new VirtualConsole();
virtualConsole.on("jsdomError", (error) => consoleErrors.push(error.message));
virtualConsole.on("error", (...args) => consoleErrors.push(args.map(String).join(" ")));

const dom = await JSDOM.fromURL(baseUrl, {
  runScripts: "dangerously",
  resources: "usable",
  pretendToBeVisual: true,
  virtualConsole,
  beforeParse(window) {
    window.confirm = () => true;
    // Canvas is only touched while recording, which never happens at boot.
    window.HTMLCanvasElement.prototype.getContext = () => null;
    // jsdom may not expose fetch; borrow Node's, resolving relative paths.
    if (typeof window.fetch !== "function") {
      window.fetch = (input, init) =>
        fetch(input instanceof URL ? input.href : new URL(String(input), window.location.href), init);
    }
  },
});

const { window } = dom;

await new Promise((resolveLoad) => {
  if (window.document.readyState === "complete") resolveLoad();
  else window.addEventListener("load", resolveLoad);
});
// Let the bootstrap fetches and Alpine's reactivity settle.
await new Promise((r) => setTimeout(r, 1200));

// --------------------------------------------------------------------------- //
// Bundles
// --------------------------------------------------------------------------- //
check("Alpine registered itself", Boolean(window.Alpine), typeof window.Alpine);
check("Alpine exposes start()", typeof window.Alpine?.start === "function");
check("studio factory is a global", typeof window.studio === "function", typeof window.studio);
check("Wavesurfer is loaded", typeof window.WaveSurfer?.create === "function");
check("Lucide is loaded", typeof window.lucide?.createIcons === "function");

// --------------------------------------------------------------------------- //
// The decisive assertion: did `x-data="studio()"` actually evaluate?
// --------------------------------------------------------------------------- //
let mounted = null;
try {
  mounted = window.Alpine.$data(window.document.body);
} catch (error) {
  mounted = null;
}
check("component mounted on <body>", Boolean(mounted), mounted);
check("mounts on the Voice Studio tab by default", mounted?.tab === "voice", mounted?.tab);
check("mounted state exposes generate()", typeof mounted?.generate === "function");
check("mounted state exposes renderProject()", typeof mounted?.renderProject === "function");

// --------------------------------------------------------------------------- //
// Live backend wiring — these requests hit the real FastAPI app
// --------------------------------------------------------------------------- //
check("capabilities loaded from the API", mounted?.capabilities !== null, mounted?.capabilities);
check("capabilities request did not fail", mounted?.capabilitiesFailed === false, mounted?.capabilitiesFailed);
check("backend is reachable for synthesis", mounted?.blockedReason?.includes("XTTS-v2"), mounted?.blockedReason);
check(
  "language catalogue loaded",
  Array.isArray(mounted?.languages) && mounted.languages.some((item) => item.code === "en"),
  mounted?.languages?.length,
);
check(
  "unsupported languages are flagged (Uzbek)",
  Array.isArray(mounted?.unsupportedLanguages) &&
    mounted.unsupportedLanguages.some((item) => item.code === "uz"),
  mounted?.unsupportedLanguages?.length,
);

// --------------------------------------------------------------------------- //
// Lucide must have replaced every <i data-lucide> placeholder
// --------------------------------------------------------------------------- //
// Lucide replaces each <i data-lucide> with an <svg> that *keeps* the attribute,
// so count placeholders by tag rather than by attribute.
const iconPlaceholders = window.document.querySelectorAll("i[data-lucide]").length;
const icons = window.document.querySelectorAll("svg[data-lucide]").length;
check("no unconverted <i> icon placeholders remain", iconPlaceholders === 0, iconPlaceholders);
check("Lucide rendered SVG icons", icons > 0, icons);
check(
  "every rendered icon is an SVG with a lucide class",
  window.document.querySelectorAll("svg.lucide[data-lucide]").length === icons,
  { icons, lucideClass: window.document.querySelectorAll("svg.lucide[data-lucide]").length },
);

// --------------------------------------------------------------------------- //
// Structure
// --------------------------------------------------------------------------- //
check("three tab buttons rendered", window.document.querySelectorAll("nav button.tab").length === 3);
check("microphone visualizer canvas present", window.document.querySelectorAll("canvas[data-viz]").length > 0);
check(
  "wavesurfer container present",
  Boolean(window.document.querySelector('[x-ref="waveform"]')),
);
check(
  "dropzone wired with drag handlers",
  window.document.querySelector(".dropzone")?.getAttribute("@drop")?.includes("handleDrop") ||
    window.document.querySelector(".dropzone")?.outerHTML.includes("@drop"),
);

// --------------------------------------------------------------------------- //
// Nothing may explode while the page boots
// --------------------------------------------------------------------------- //
const fatal = consoleErrors.filter((message) =>
  /ReferenceError|is not defined|is not a function|SyntaxError/i.test(message),
);
check("no fatal script errors during boot", fatal.length === 0, fatal);

dom.window.close();

if (failures.length) {
  console.error(`\n${failures.length} boot check(s) failed: ${failures.join(", ")}`);
  process.exit(1);
}
console.log("\nBoot check passed — the studio mounts and talks to the backend.");
