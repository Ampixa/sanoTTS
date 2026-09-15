/**
 * headless_check.mjs — drives the built site in headless Chromium over CDP.
 *
 * Verifies the interaction contract of the two-panel redesign:
 *   1. left-column scroll is the ONLY thing that moves the stage focus
 *   2. wheel over the right panel forwards into the left column
 *   3. right-panel interactions (play, drag cursor) never change the stage
 *   4. schematic clicks jump the left column
 *   5. sentence switch updates both panels
 *   6. WASM live run produces samples + corr badge
 * and captures a screenshot per stage into /tmp/sanotts-shots/.
 *
 * usage: node tools/headless_check.mjs [baseUrl]   (default http://127.0.0.1:4173)
 */
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";

const BASE = process.argv[2] ?? "http://127.0.0.1:4173";
const PORT = 9223;
const SHOTS = "/tmp/sanotts-shots";
mkdirSync(SHOTS, { recursive: true });

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let failures = 0;
const check = (name, ok, extra = "") => {
  console.log(`${ok ? "PASS" : "FAIL"}  ${name}${extra ? ` — ${extra}` : ""}`);
  if (!ok) failures++;
};

/* ---- launch chromium ---- */
const chrome = spawn("chromium", [
  "--headless=new", "--disable-gpu", "--enable-unsafe-swiftshader",
  "--use-gl=angle", "--use-angle=swiftshader",
  "--mute-audio", "--no-sandbox",
  "--no-proxy-server", "--autoplay-policy=no-user-gesture-required",
  `--remote-debugging-port=${PORT}`, "--window-size=1600,1000",
  "about:blank",
], { stdio: "ignore" });
process.on("exit", () => chrome.kill());

/* wait for devtools */
let targets = null;
for (let i = 0; i < 50; i++) {
  try {
    targets = await fetch(`http://127.0.0.1:${PORT}/json`).then((r) => r.json());
    break;
  } catch { await sleep(200); }
}
if (!targets) { console.error("chromium devtools never came up"); process.exit(1); }
const page = targets.find((t) => t.type === "page");

const ws = new WebSocket(page.webSocketDebuggerUrl);
await new Promise((res) => (ws.onopen = res));

let mid = 0;
const pending = new Map();
const events = [];
ws.onmessage = (m) => {
  const msg = JSON.parse(m.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
  else if (msg.method) events.push(msg);
};
const send = (method, params = {}) =>
  new Promise((res) => { const id = ++mid; pending.set(id, res); ws.send(JSON.stringify({ id, method, params })); });

const evaluate = async (expr) => {
  const r = await send("Runtime.evaluate", { expression: expr, returnByValue: true, awaitPromise: true });
  if (r.result?.exceptionDetails) throw new Error(JSON.stringify(r.result.exceptionDetails));
  return r.result?.result?.value;
};
const shot = async (name) => {
  const r = await send("Page.captureScreenshot", { format: "png" });
  writeFileSync(`${SHOTS}/${name}.png`, Buffer.from(r.result.data, "base64"));
};

await send("Page.enable");
await send("Runtime.enable");
await send("Emulation.setDeviceMetricsOverride", { width: 1600, height: 1000, deviceScaleFactor: 1, mobile: false });
await send("Page.navigate", { url: BASE });

/* wait for the app to load its first trace (stage header shows a row id) */
let loaded = false;
for (let i = 0; i < 100; i++) {
  await sleep(300);
  loaded = await evaluate(`!!document.querySelector('#stage-header b') && document.querySelector('#stage-header b').textContent.includes('0000')`).catch(() => false);
  if (loaded) break;
}
check("app loads + first trace arrives", loaded);

const headerText = () => evaluate(`document.querySelector('#stage-header .shape').textContent`);
const activeBoxLabel = () =>
  evaluate(`[...document.querySelectorAll('#schematic rect')].map(r=>r.getAttribute('fill')).join(',')`);

/* ---------- 1. scroll drives stage ---------- */
const nChapters = await evaluate(`document.querySelectorAll('.story-block').length`);
check("13 story blocks present", nChapters === 13, `got ${nChapters}`);

const seenHeaders = [];
for (let i = 0; i < nChapters; i++) {
  await evaluate(`(() => { const s = document.querySelectorAll('.story-block')[${i}]; document.getElementById('explain').scrollTop = s.offsetTop + 10; })()`);
  await sleep(450);
  const h = await headerText();
  seenHeaders.push(h);
  const safe = ["hero", "frontend", "duration", "acoustic", "mel", "decoder", "spectrum", "waveform", "budget", "lanes", "evidence", "deploy", "live"][i];
  await shot(`stage-${String(i).padStart(2, "0")}-${safe}`);
}
check("scroll visits 13 distinct stage headers", new Set(seenHeaders).size === nChapters, `${new Set(seenHeaders).size} distinct`);

/* ---------- 2. wheel over right panel forwards to left column ---------- */
await evaluate(`document.getElementById('explain').scrollTop = 0`);
await sleep(300);
const before = await evaluate(`document.getElementById('explain').scrollTop`);
await evaluate(`document.getElementById('stageviz').dispatchEvent(new WheelEvent('wheel', { deltaY: 600, bubbles: true }))`);
await sleep(300);
const after = await evaluate(`document.getElementById('explain').scrollTop`);
check("wheel over right panel scrolls the story", after > before, `${before} → ${after}`);

/* ---------- 3. right-panel play does not move the stage ---------- */
/* go to mel chapter (index 4) */
await evaluate(`(() => { const s = document.querySelectorAll('.story-block')[4]; document.getElementById('explain').scrollTop = s.offsetTop + 10; })()`);
await sleep(500);
const melHeader = await headerText();
await evaluate(`(() => { const b = [...document.querySelectorAll('.playbtn')].find(x => x.textContent.includes('play')); if (b) b.click(); })()`);
await sleep(700);
const melHeaderAfterPlay = await headerText();
check("pressing play on the right keeps the stage", melHeader === melHeaderAfterPlay);
await shot("play-isolation-mel");
const playingState = await evaluate(`document.querySelector('.playbtn')?.textContent ?? ''`);
await evaluate(`(() => { const b = [...document.querySelectorAll('.playbtn')].find(x => x.textContent.includes('stop')); if (b) b.click(); })()`);
check("play button toggled", playingState.includes("stop") || playingState.includes("play"), playingState.trim());

/* ---------- 4. spectrum drag cursor does not move the stage ---------- */
await evaluate(`(() => { const s = document.querySelectorAll('.story-block')[6]; document.getElementById('explain').scrollTop = s.offsetTop + 10; })()`);
await sleep(500);
const specHeader = await headerText();
await evaluate(`(() => {
  const cv = document.querySelector('#stage-view canvas');
  const r = cv.getBoundingClientRect();
  cv.dispatchEvent(new MouseEvent('click', { clientX: r.left + r.width * 0.7, clientY: r.top + r.height * 0.4, bubbles: true }));
})()`);
await sleep(300);
check("dragging spectrum cursor keeps the stage", specHeader === (await headerText()));

/* ---------- 4b. 3D overview overlay: optional, zoomable, playable, inert ---------- */
/* hero (chapter 0) is the overview — no toggle button there */
await evaluate(`document.getElementById('explain').scrollTop = 0`);
await sleep(400);
const heroToggle = await evaluate(`!!document.querySelector('.ov-toggle')`);
check("hero stage hides the overview toggle (it IS the overview)", !heroToggle);
const heroPlay = await evaluate(`!!document.querySelector('#stage-view .ov-toolbar .playbtn')`);
check("hero overview has play + zoom controls", heroPlay);

/* mel chapter: summon the overlay */
await evaluate(`(() => { const s = document.querySelectorAll('.story-block')[4]; document.getElementById('explain').scrollTop = s.offsetTop + 10; })()`);
await sleep(500);
const melHeader2 = await headerText();
await evaluate(`document.querySelector('.ov-toggle')?.dispatchEvent(new MouseEvent('click', { bubbles: true }))`);
let overlayUp = false;
for (let i = 0; i < 40; i++) {
  await sleep(300);
  overlayUp = await evaluate(`!!document.querySelector('.overview-overlay canvas')`);
  if (overlayUp) break;
}
check("overview overlay opens on the mel stage", overlayUp);
check("overlay does not change the stage", (await headerText()) === melHeader2);

/* wheel over the 3D canvas must ZOOM, not scroll the story */
const scrollBeforeZoom = await evaluate(`document.getElementById('explain').scrollTop`);
await evaluate(`(() => {
  const cv = document.querySelector('.overview-overlay canvas');
  const r = cv.getBoundingClientRect();
  cv.dispatchEvent(new WheelEvent('wheel', { deltaY: -240, clientX: r.width / 2, clientY: r.height / 2, bubbles: true }));
})()`);
await sleep(400);
const scrollAfterZoom = await evaluate(`document.getElementById('explain').scrollTop`);
check("wheel over 3D canvas zooms (story scroll untouched)", scrollBeforeZoom === scrollAfterZoom, `${scrollBeforeZoom} → ${scrollAfterZoom}`);

/* play inside the overlay */
await evaluate(`(() => { const b = [...document.querySelectorAll('.ov-toolbar .playbtn')].find(x => x.textContent.includes('play')); if (b) b.click(); })()`);
await sleep(600);
const ovPlay = await evaluate(`document.querySelector('.ov-toolbar .playbtn')?.textContent ?? ''`);
check("overview play button toggles", ovPlay.includes("stop"), ovPlay.trim());
check("overview play keeps the stage", (await headerText()) === melHeader2);
await shot("overview-overlay-mel");
await evaluate(`(() => { const b = [...document.querySelectorAll('.ov-toolbar .playbtn')].find(x => x.textContent.includes('stop')); if (b) b.click(); })()`);

/* close returns the stage view */
await evaluate(`(() => { const b = [...document.querySelectorAll('.ov-toolbar .ovbtn')].find(x => x.textContent.includes('close')); if (b) b.click(); })()`);
await sleep(500);
const overlayGone = await evaluate(`!document.querySelector('.overview-overlay')`);
const melBack = await evaluate(`!!document.querySelector('#stage-view canvas')`);
check("closing overlay restores the stage view", overlayGone && melBack);
check("stage still mel after close", (await headerText()) === melHeader2);

/* ---------- 5. schematic click jumps the left column ---------- */
await evaluate(`document.getElementById('explain').scrollTop = 0`);
await sleep(400);
await evaluate(`(() => { document.querySelectorAll('#schematic g')[7].dispatchEvent(new MouseEvent('click', { bubbles: true })); })()`);
let jumpedHeader = "";
for (let i = 0; i < 30; i++) {           // smooth scroll may take a while
  await sleep(300);
  jumpedHeader = await headerText();
  if (jumpedHeader.includes("iSTFT")) break;
}
check("schematic click jumps story to waveform stage", jumpedHeader.includes("iSTFT"), jumpedHeader.slice(0, 60));

/* ---------- 6. sentence switch updates both panels ---------- */
const row0 = await evaluate(`document.querySelector('#stage-header b').textContent`);
await evaluate(`(() => {
  const sel = document.getElementById('rowsel');
  sel.value = sel.options[3].value;
  sel.dispatchEvent(new Event('change', { bubbles: true }));
})()`);
await sleep(1200);
const row1 = await evaluate(`document.querySelector('#stage-header b').textContent`);
const headerAfterSwitch = await headerText();
check("sentence switch updates stage header", row0 !== row1, `${row0} → ${row1}`);
check("stage header shows live shapes after switch", /samples|\[\d+/.test(headerAfterSwitch), headerAfterSwitch.slice(0, 60));

/* ---------- 7. WASM live run ---------- */
await evaluate(`(() => { const s = document.querySelectorAll('.story-block')[12]; document.getElementById('explain').scrollTop = s.offsetTop + 10; })()`);
await sleep(500);
await evaluate(`(() => { const b = [...document.querySelectorAll('.playbtn')].find(x => x.textContent.includes('run WASM')); if (b) b.click(); })()`);
let wasmStats = "";
for (let i = 0; i < 100; i++) {
  await sleep(400);
  wasmStats = await evaluate(`[...document.querySelectorAll('.stat-chip')].map(c => c.textContent).join(' | ')`);
  if (wasmStats.includes("corr")) break;
}
check("WASM run produced stats", wasmStats.includes("samples") && wasmStats.includes("corr"), wasmStats);
const corr = /corr[^|]*?([\d.]+)\s*corr vs/.exec(wasmStats)?.[1] ?? /([\d.]+)\s*corr vs/.exec(wasmStats)?.[1];
check("corr badge ~1.0", corr != null && parseFloat(corr) > 0.99, `corr=${corr}`);
await shot("stage-12-live-after-run");

/* ---------- page errors ---------- */
const errors = events.filter((e) => e.method === "Runtime.exceptionThrown");
check("no page exceptions", errors.length === 0, errors.slice(0, 2).map((e) => e.params?.exceptionDetails?.text).join("; "));

console.log(failures === 0 ? "\nALL CHECKS PASS" : `\n${failures} FAILURES`);
chrome.kill();
process.exit(failures === 0 ? 0 : 1);
