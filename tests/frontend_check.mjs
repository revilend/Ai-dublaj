/**
 * Headless verification of app/static/js/studio.js.
 *
 * The browser-only APIs the component touches (window, document, URL, Blob)
 * are stubbed, so the pure logic — computed getters, state transitions and the
 * Tailwind class mappings — can be asserted without a real browser.
 *
 * Run with:  node tests/frontend_check.mjs
 */
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import vm from "node:vm";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const source = readFileSync(resolve(root, "app/static/js/studio.js"), "utf8");

const sandbox = {
  console,
  setTimeout,
  clearTimeout,
  URL,
  Blob,
  FormData,
  Audio: function Audio() {},
  MediaRecorder: undefined,
  navigator: {},
  requestAnimationFrame: () => 0,
  cancelAnimationFrame: () => {},
  fetch: async () => {
    throw new Error("network is not available in the harness");
  },
};
sandbox.window = sandbox;
sandbox.document = { querySelectorAll: () => [], querySelector: () => null };
sandbox.location = { protocol: "http:" };

vm.createContext(sandbox);
vm.runInContext(source, sandbox);

let failures = 0;

function check(label, condition, actual) {
  if (condition) {
    console.log(`  ok    ${label}`);
  } else {
    failures += 1;
    console.error(`  FAIL  ${label} — got ${JSON.stringify(actual)}`);
  }
}

// --------------------------------------------------------------------------- //
// Component bootstraps
// --------------------------------------------------------------------------- //
check("window.studio is exposed", typeof sandbox.window.studio === "function");
const component = sandbox.studio();
check("component is an object", component && typeof component === "object");

const requiredMethods = [
  "init",
  "startRecording",
  "stopRecording",
  "finishRecording",
  "generate",
  "loadWaveform",
  "togglePlayback",
  "loadVoices",
  "saveVoice",
  "deleteVoice",
  "createProject",
  "renderProject",
  "uploadSubtitles",
  "saveSegmentText",
  "assignSegmentVoice",
  "previewLine",
  "playPreview",
  "autoTranslate",
  "seekToSegment",
  "stepSegment",
  "syncActiveSegment",
  "togglePlay",
  "onEditorKeydown",
  "onVideoTimeUpdate",
  "onVideoLoaded",
  "formatRange",
  "formatTimestamp",
  "stageClass",
  "resetPlayerState",
  "toast",
  "dismissToast",
  "drawBars",
];
requiredMethods.forEach((name) => {
  check(`method ${name}()`, typeof component[name] === "function", typeof component[name]);
});

// --------------------------------------------------------------------------- //
// Defaults
// --------------------------------------------------------------------------- //
check("default tab is voice", component.tab === "voice", component.tab);
check("default speed is 1", component.speed === 1, component.speed);
check("default pitch is 0", component.pitch === 0, component.pitch);
check("default format is wav", component.format === "wav", component.format);
check("no reference at boot", component.reference.blob === null);
check("analysis runs on upload by default", component.newProject.analyze === true, component.newProject.analyze);

// --------------------------------------------------------------------------- //
// The dubbing UI must call the current endpoint surface, not the retired one.
// --------------------------------------------------------------------------- //
check("upload posts to /api/dubbing/upload", source.includes('/api/dubbing/upload'));
check("render posts to /api/dubbing/render/{id}", source.includes('/api/dubbing/render/${project.id}'));
check("segment edits PUT /api/dubbing/segments/{id}", source.includes('/api/dubbing/segments/${segment.id}'));
check(
  "per-line preview posts to /api/dubbing/segments/{id}/preview",
  source.includes('/api/dubbing/segments/${segment.id}/preview'),
);
check(
  "auto-translate posts to /api/dubbing/{id}/translate",
  source.includes('/api/dubbing/${project.id}/translate'),
);
check(
  "the retired POST /api/dubbing create route is gone",
  !source.includes('apiRequest("/api/dubbing", { method: "POST"'),
  source.match(/apiRequest\("\/api\/dubbing"[^)]*\)/)?.[0],
);
check(
  "the retired /{id}/render route is gone",
  !/\/api\/dubbing\/\$\{[^}]*\}\/render/.test(source),
  source.match(/\/api\/dubbing\/\$\{[^}]*\}\/render/)?.[0],
);

// --------------------------------------------------------------------------- //
// Word counter
// --------------------------------------------------------------------------- //
component.text = "";
check("wordCount of empty text is 0", component.wordCount === 0, component.wordCount);
component.text = "Salom dunyo, bu studio";
check("wordCount counts words", component.wordCount === 4, component.wordCount);
component.text = "   spaced   out   ";
check("wordCount ignores padding", component.wordCount === 2, component.wordCount);

// --------------------------------------------------------------------------- //
// Guard rails (blockedReason / canSynthesize)
// --------------------------------------------------------------------------- //
const blocked = () => component.blockedReason;

check("blocks until engines are known", blocked().includes("Checking local engines"), blocked());
component.capabilities = { tts: { available: false } };
check("blocks without XTTS", blocked().includes("XTTS-v2 is not installed"), blocked());

component.capabilities = { tts: { available: true } };
component.text = "";
check("blocks without text", blocked() === "Type a script to synthesise.", blocked());

component.text = "Salom";
component.voiceSource = "library";
component.selectedVoiceId = "";
check("blocks without a voice", blocked().includes("saved voice"), blocked());

component.selectedVoiceId = "3";
check("allows a complete library request", component.canSynthesize === true, blocked());

component.voiceSource = "recording";
check(
  "blocks without a recorded sample",
  blocked().toLowerCase().includes("record or upload"),
  blocked(),
);
component.reference = { kind: "recording", blob: new Blob(["x"]), name: "r.webm", url: "blob:x" };
check("allows a recorded sample", component.canSynthesize === true, blocked());

// A backend that cannot report capabilities must not dead-lock the button.
component.capabilitiesFailed = true;
component.capabilities = null;
check("failure to load capabilities still allows attempts", component.canSynthesize === true, blocked());
component.capabilitiesFailed = false;

// --------------------------------------------------------------------------- //
// Formatters
// --------------------------------------------------------------------------- //
check("formatClock(0) is safe", component.formatClock(0) === "0:00.0", component.formatClock(0));
check("formatClock(null) is safe", component.formatClock(null) === "0:00.0", component.formatClock(null));
check("formatClock(75.4)", component.formatClock(75.4) === "1:15.4", component.formatClock(75.4));
// The `Time` column contract from the spec: `00:03 - 00:07`.
check("formatTimestamp pads to MM:SS", component.formatTimestamp(3) === "00:03", component.formatTimestamp(3));
check("formatTimestamp floors seconds", component.formatTimestamp(75.9) === "01:15", component.formatTimestamp(75.9));
check("formatTimestamp over an hour", component.formatTimestamp(3661) === "61:01", component.formatTimestamp(3661));
check("formatTimestamp is safe on null", component.formatTimestamp(null) === "00:00", component.formatTimestamp(null));
check("formatTimestamp clamps negatives", component.formatTimestamp(-4) === "00:00", component.formatTimestamp(-4));
check(
  "formatRange matches `00:03 - 00:07`",
  component.formatRange({ start_time: 3, end_time: 7 }) === "00:03 - 00:07",
  component.formatRange({ start_time: 3, end_time: 7 }),
);
check("initials two words", component.initials("Ada Lovelace") === "AL", component.initials("Ada Lovelace"));
check("initials one word", component.initials("Narrator") === "N", component.initials("Narrator"));
check("initials fallback", component.initials("") === "V", component.initials(""));

// --------------------------------------------------------------------------- //
// Tailwind class mappings must be literal strings (the JIT scans them)
// --------------------------------------------------------------------------- //
check("progressClass running", component.progressClass("running") === "progress-bar");
check("progressClass failed", component.progressClass("failed") === "progress-bar-failed");
check("progressClass completed", component.progressClass("completed") === "progress-bar-done");
check("statusChipClass completed", component.statusChipClass("completed") === "chip-ok");
check("statusChipClass failed", component.statusChipClass("failed") === "chip-danger");
check("statusChipClass pending", component.statusChipClass("pending") === "chip-neutral");
check("statusChipClass running", component.statusChipClass("synthesizing") === "chip-warn");
check("toastClass error", component.toastClass("error") === "toast-error");
check("toastClass success", component.toastClass("success") === "toast-success");
check("toastClass info", component.toastClass("info") === "toast-info");
check("toastIconClass returns a string", typeof component.toastIconClass("error") === "string");

// --------------------------------------------------------------------------- //
// Subtitle timeline geometry
// --------------------------------------------------------------------------- //
component.selectedProject = { duration_seconds: 10 };
component.segments = [
  { id: 1, start_time: 0, end_time: 2.5, original_text: "a", dubbed_audio_url: "/media/x.wav" },
  { id: 2, start_time: 2.5, end_time: 10, original_text: "b", translated_text: "B", dubbed_audio_url: null },
];
const cues = component.timelineCues;
check("timeline produces one block per cue", cues.length === 2, cues.length);
check("first cue starts at 0%", cues[0].left === 0, cues[0].left);
check("first cue is 25% wide", cues[0].width === 25, cues[0].width);
check("second cue is 75% wide", cues[1].width === 75, cues[1].width);
check("cue marks dubbed audio", cues[0].hasAudio === true);
check("cue prefers translated text", cues[1].title === "B", cues[1].title);
check("projectDuration from project", component.projectDuration === 10, component.projectDuration);

component.selectedProject = { duration_seconds: null };
check("projectDuration falls back to last cue", component.projectDuration === 10, component.projectDuration);
component.segments = [];
check("timeline is empty without cues", component.timelineCues.length === 0);

// --------------------------------------------------------------------------- //
// Toasts
// --------------------------------------------------------------------------- //
component.$nextTick = (fn) => (typeof fn === "function" ? fn() : undefined);
component.toast("hello", "success", "detail");
check("toast is pushed", component.toasts.length === 1, component.toasts.length);
check("toast keeps its kind", component.toasts[0].kind === "success");
check("toast carries an icon name", component.toasts[0].icon === "check-circle-2");
component.toast("boom", "error");
check("toast error icon", component.toasts[1].icon === "alert-triangle");
component.dismissToast(component.toasts[0].id);
check("dismissToast removes one", component.toasts.length === 1, component.toasts.length);

// --------------------------------------------------------------------------- //
// Class-name safety: no interpolated Tailwind classes in the templates
// --------------------------------------------------------------------------- //
const template = readFileSync(resolve(root, "app/templates/index.html"), "utf8");
const interpolated = template.match(/class="[^"]*\$\{/g) || [];
check("no interpolated class attributes", interpolated.length === 0, interpolated);
check("template loads the compiled stylesheet", template.includes("/static/css/tailwind.css"));
check("template loads the vendored Alpine", template.includes("/static/vendor/alpine.min.js"));
check("template loads the vendored Wavesurfer", template.includes("/static/vendor/wavesurfer.min.js"));
check("template loads the vendored Lucide", template.includes("/static/vendor/lucide.min.js"));
check("no runtime CDN references", !/https?:\/\/cdn\./.test(template));
check("three studio tabs", ["voice", "dubbing", "library"].every((tab) => template.includes(`tab = '${tab}'`)));

// --------------------------------------------------------------------------- //
// Dubbing tab layout: left player/controls, right 5-column editor
// --------------------------------------------------------------------------- //
const dubbingSection = template.slice(
  template.indexOf("2. DUBBING STUDIO"),
  template.indexOf("3. VOICE LIBRARY"),
);
check("dubbing tab is present", dubbingSection.length > 0);
for (const heading of ["Time", "Original", "Translation", "Voice", "Preview"]) {
  check(`editor column "${heading}"`, dubbingSection.includes(`>${heading}</th>`), heading);
}
check("video element is wired to the player ref", dubbingSection.includes('x-ref="player"'));
check("video drives the timeline sync", dubbingSection.includes("@timeupdate=\"onVideoTimeUpdate($event)\""));
check("rows seek the player", dubbingSection.includes("@click=\"seekToSegment(segment, index)\""));
check("render button present", dubbingSection.includes("Render dubbed video"));
check("final-video download present", dubbingSection.includes("Download final video"));
check("auto-translate button present", dubbingSection.includes("autoTranslate()"));
check("per-line preview button present", dubbingSection.includes("previewLine(segment)"));
check("per-line preview button is labelled", dubbingSection.includes('aria-label="Preview line"'));
check("per-line replay button is labelled", dubbingSection.includes('aria-label="Replay this line"'));
check("editor is keyboard aware", dubbingSection.includes("onEditorKeydown($event)"));
check("segment list is scrollable", dubbingSection.includes('x-ref="segmentList"'));
check(
  "player and editor sit side by side on xl",
  dubbingSection.includes("xl:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]"),
);
check("four pipeline stage tiles", (dubbingSection.match(/stageClass\(\d\)/g) || []).length === 4);

// Every data-lucide icon must sit on an element with no Alpine directive,
// otherwise Lucide replacing the node would strip reactivity.
const lucideOnDirective = template.match(/<[a-z-]+[^>]*x-[a-z-]+[^>]*data-lucide/gi) || [];
check("no Alpine directives on data-lucide elements", lucideOnDirective.length === 0, lucideOnDirective);

// --------------------------------------------------------------------------- //
// Drag state: dragenter/dragleave also fire while crossing the dropzone's own
// children, so the highlight is driven by a depth counter rather than a bool.
// --------------------------------------------------------------------------- //
const drag = sandbox.studio();
drag.$nextTick = () => {};
check("dropzone is idle initially", drag.dragging === false, drag.dragging);

drag.startDrag("dragging");
check("enter activates the dropzone", drag.dragging === true);
drag.startDrag("dragging"); // pointer moved onto a child element
drag.endDrag("dragging");
check("stays active after a nested leave", drag.dragging === true, drag.dragging);
drag.endDrag("dragging");
check("deactivates once every leave has fired", drag.dragging === false, drag.dragging);

// A dropped file must clear the state even if a dragleave was missed.
drag.startDrag("dragging");
drag.startDrag("dragging");
drag.handleDrop({ dataTransfer: { files: [] } });
check(
  "drop resets the drag state",
  drag.dragging === false && drag.dragDepth.dragging === 0,
  { dragging: drag.dragging, depth: drag.dragDepth.dragging },
);

// The audio and video dropzones must not share a flag.
drag.startDrag("draggingVideo");
drag.startDrag("dragging");
drag.endDrag("dragging");
check(
  "video dropzone counts independently",
  drag.draggingVideo === true && drag.dragging === false,
  { video: drag.draggingVideo, audio: drag.dragging },
);
drag.endDrag("draggingVideo");
check("video flag clears", drag.draggingVideo === false);

// --------------------------------------------------------------------------- //
// Load order: the Alpine CDN build starts itself from a queueMicrotask that
// drains as soon as its own <script> finishes — before the next deferred
// script. So studio.js must come first or `x-data="studio()"` would evaluate
// against an undefined factory and the whole page would fail to mount.
// --------------------------------------------------------------------------- //
const studioIndex = template.indexOf("/static/js/studio.js");
const alpineIndex = template.indexOf("/static/vendor/alpine.min.js");
check("studio.js is present in the template", studioIndex !== -1);
check("alpine.min.js is present in the template", alpineIndex !== -1);
check(
  "studio.js loads before alpine.min.js",
  studioIndex !== -1 && alpineIndex !== -1 && studioIndex < alpineIndex,
  { studioIndex, alpineIndex },
);
check("component init is wired", template.includes('x-init="init()"'));
check("body mounts the studio component", template.includes('x-data="studio()"'));

// --------------------------------------------------------------------------- //
// init() runs once even though Alpine auto-invokes it and x-init also wires it
// --------------------------------------------------------------------------- //
const lifecycle = sandbox.studio();
lifecycle.$nextTick = () => {};
let fetchCalls = 0;
sandbox.fetch = async () => {
  fetchCalls += 1;
  return { ok: true, status: 200, text: async () => "{}" };
};

await lifecycle.init();
const afterFirstInit = fetchCalls;
await lifecycle.init();
check("init() bootstraps all four endpoints", afterFirstInit === 4, afterFirstInit);
check("init() is idempotent", fetchCalls === afterFirstInit, {
  afterFirstInit,
  afterSecondInit: fetchCalls,
});

if (failures) {
  console.error(`\n${failures} frontend check(s) failed.`);
  process.exit(1);
}
console.log("\nAll frontend checks passed.");
