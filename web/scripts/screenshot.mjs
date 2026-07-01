// SPDX-License-Identifier: Apache-2.0
/**
 * Reusable UI screenshot generator for dFactory-Lab.
 *
 * Launches Google Chrome (stable / "main" channel) and falls back to the
 * Playwright-bundled Chromium if Chrome isn't installed. Captures the listed
 * pages of the running app to PNGs for the README.
 *
 * Prereqs:
 *   - the frontend is built:        cd web && npm run build
 *   - the server is serving it:     python server/run.py   (serves SPA at /)
 *
 * Run:
 *   node web/scripts/screenshot.mjs
 *   BASE_URL=http://127.0.0.1:8000 OUT_DIR=web/screenshots node web/scripts/screenshot.mjs
 */
import { chromium } from "playwright";
import { mkdir } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const BASE = process.env.BASE_URL || "http://127.0.0.1:8000";
const OUT = process.env.OUT_DIR || path.resolve(__dirname, "..", "screenshots");

// (route, filename, optional readiness wait + optional pre-shoot interaction)
const SHOTS = [
  {
    route: "/", name: "dashboard", label: "Dashboard",
    ready: async (p) => p.waitForSelector("text=ok", { timeout: 12000 }),
  },
  { route: "/models", name: "models", label: "Models",
    ready: async (p) => p.waitForSelector("text=LLaDA2.0", { timeout: 10000 }) },
  { route: "/training", name: "training", label: "Training",
    ready: async (p) => p.waitForSelector("text=Hyperparameters", { timeout: 10000 }) },
  {
    route: "/chat", name: "chat", label: "Chat & diffusion playground",
    interact: async (page) => {
      await page.fill('input[placeholder="Message…"]', "Explain masked diffusion in one line.");
      await page.click("button.btn-primary");
      await page.waitForSelector("text=Unmasking", { timeout: 10000 });
      await page.waitForTimeout(500);
    },
  },
  {
    route: "/settings", name: "settings", label: "Settings & security",
    ready: async (p) => p.waitForFunction(
      () => [...document.querySelectorAll("span")]
        .some((s) => /^\d+\.\d+\.\d+$/.test((s.textContent || "").trim())),
      { timeout: 10000 }),
  },
];

async function launchBrowser() {
  // 1) Prefer the system Google Chrome (stable channel).
  try {
    const b = await chromium.launch({ channel: "chrome" });
    console.log("browser: Google Chrome (channel=chrome)");
    return b;
  } catch (e) {
    console.warn(`chrome channel unavailable (${String(e.message).split("\n")[0]}); falling back to bundled Chromium`);
  }
  // 2) Fallback: Playwright-bundled Chromium.
  try {
    const b = await chromium.launch();
    console.log("browser: bundled Chromium");
    return b;
  } catch (e) {
    throw new Error(`Could not launch any browser. Last error: ${e.message}`);
  }
}

async function unclipMain(page) {
  // Let the scrollable <main> expand so fullPage captures all content.
  await page.addStyleTag({ content: "main, main:focus { overflow: visible !important; height: auto !important; } html, body { height: auto !important; }" });
}

(async () => {
  await mkdir(OUT, { recursive: true });
  const browser = await launchBrowser();
  const page = await browser.newPage({
    viewport: { width: 1440, height: 900 },
    deviceScaleFactor: 2,
  });

  const saved = [];
  try {
    for (const shot of SHOTS) {
      const url = BASE + shot.route;
      process.stdout.write(`  ${shot.label.padEnd(28)} `);
      try {
        await page.goto(url, { waitUntil: "networkidle", timeout: 15000 });
      } catch {
        await page.goto(url, { waitUntil: "domcontentloaded", timeout: 15000 });
      }
      await page.waitForTimeout(300);            // let React settle
      if (shot.ready) await shot.ready(page);    // wait for real data
      if (shot.interact) await shot.interact(page);
      await unclipMain(page);
      const file = path.join(OUT, `${shot.name}.png`);
      await page.screenshot({ path: file, fullPage: true });
      saved.push(file);
      console.log(`✔  ${path.relative(process.cwd(), file) || file}`);
    }
  } finally {
    await browser.close();
  }
  console.log(`\nSaved ${saved.length} screenshots to ${OUT}`);
})().catch((e) => {
  console.error("screenshot failed:", e);
  process.exit(1);
});
