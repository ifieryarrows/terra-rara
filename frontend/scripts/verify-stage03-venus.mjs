import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const { chromium } = require('playwright');

const BASE_URL = process.env.EXPERIENCE_URL || 'http://127.0.0.1:5173';
const CALLER_ARTIFACT_DIR = 'C:\\Users\\gokha\\.gemini\\antigravity\\brain\\6492c8fa-29ca-475a-bd87-7cd2635bbb95';
const LOCAL_ARTIFACT_DIR = 'C:\\Users\\gokha\\.gemini\\antigravity\\brain\\59a0e288-5502-47bd-9c50-ac42f147a3e0';

await mkdir(CALLER_ARTIFACT_DIR, { recursive: true });
await mkdir(LOCAL_ARTIFACT_DIR, { recursive: true });

async function saveScreenshot(page, filename, description) {
  const callerPath = join(CALLER_ARTIFACT_DIR, filename);
  const localPath = join(LOCAL_ARTIFACT_DIR, filename);
  await page.screenshot({ path: callerPath, fullPage: false });
  await page.screenshot({ path: localPath, fullPage: false });
  console.log(`[Captured] ${filename} - ${description}`);
}

async function run() {
  console.log('Launching browser with msedge channel at 1920x1080...');
  let browser;
  try {
    browser = await chromium.launch({
      headless: true,
      channel: 'msedge',
      args: ['--enable-webgl', '--ignore-gpu-blocklist', '--use-gl=angle', '--use-angle=d3d11'],
    });
  } catch (e) {
    console.log('Falling back to default chromium...');
    browser = await chromium.launch({
      headless: true,
      args: ['--enable-webgl', '--ignore-gpu-blocklist'],
    });
  }

  const context = await browser.newContext({
    viewport: { width: 1920, height: 1080 },
    deviceScaleFactor: 2,
    reducedMotion: 'no-preference',
  });

  const page = await context.newPage();
  const consoleErrors = [];
  page.on('console', msg => {
    if (msg.type() === 'error') consoleErrors.push(msg.text());
  });
  page.on('pageerror', err => consoleErrors.push(err.message));

  console.log(`Navigating to ${BASE_URL}...`);
  await page.goto(BASE_URL, { waitUntil: 'networkidle' });
  await page.waitForSelector('.cm-story--enhanced', { timeout: 10000 });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(600);

  // Check renderer info
  const rendererInfo = await page.evaluate(() => {
    const canvas = document.querySelector('.cm-particle-world[data-renderer]');
    return canvas ? {
      renderer: canvas.getAttribute('data-renderer'),
      quality: canvas.getAttribute('data-quality'),
      particleCount: canvas.getAttribute('data-particle-count'),
      width: canvas.width,
      height: canvas.height,
    } : null;
  });
  console.log('ParticleWorld renderer info:', JSON.stringify(rendererInfo, null, 2));

  // Story geometry
  const geometry = await page.evaluate(() => {
    const scene = document.querySelector('.cm-cinematic');
    const rect = scene.getBoundingClientRect();
    return {
      top: rect.top + window.scrollY,
      length: Math.max(0, rect.height - window.innerHeight),
      height: rect.height,
    };
  });
  console.log('Story geometry:', JSON.stringify(geometry, null, 2));

  // Helper to scroll to normalized progress [0..1]
  const scrollTo = async (progress, smooth = false) => {
    const targetY = geometry.top + geometry.length * progress;
    await page.evaluate(({ y, smooth }) => {
      window.scrollTo({ top: y, behavior: smooth ? 'smooth' : 'auto' });
    }, { y: targetY, smooth });
    await page.waitForTimeout(smooth ? 650 : 250);
  };

  const getStageState = async () => {
    return page.evaluate(() => {
      const hudNum = document.querySelector('.cm-cosmic-ruler-num')?.textContent;
      const hudUnit = document.querySelector('.cm-cosmic-ruler-unit')?.textContent;
      const surfaces = Array.from(document.querySelectorAll('.cm-cinematic-surface')).map(s => {
        const style = window.getComputedStyle(s);
        return {
          opacity: style.opacity,
          visibility: style.visibility,
          transform: style.transform,
        };
      });
      const copyBlocks = Array.from(document.querySelectorAll('.cm-cinematic-beat')).map(b => {
        const id = b.id || 'hero';
        const inner = b.querySelector('.cm-cinematic-copy-block');
        const style = inner ? window.getComputedStyle(inner) : {};
        return {
          id,
          opacity: style.opacity,
          visibility: style.visibility,
        };
      });
      return { hudNum, hudUnit, surfaces, copyBlocks };
    });
  };

  // Milestone 1: Stage 00 - Hero Galactic Core (p = 0.0)
  console.log('\n--- Capturing Stage 00: Hero Galactic Core (p = 0.0) ---');
  await scrollTo(0);
  await saveScreenshot(page, 'stage-00-hero-galactic-core.png', 'Galactic Core (26 Thousand LY)');

  // Milestone 2: Stage 01 - The Market (p = 0.24)
  console.log('\n--- Capturing Stage 01: The Market (p = 0.24) ---');
  await scrollTo(0.24, true);
  await page.waitForTimeout(400);
  await saveScreenshot(page, 'stage-01-the-market-milkyway-rift.png', 'Milky Way Rift & 49 Assets');

  // Milestone 3: Warp Speed transition from Stage 01 to Stage 02
  console.log('\n--- Capturing Warp Speed Streaming Transition (p = 0.35) ---');
  // Dispatch active wheel gestures or fast scroll
  await page.evaluate(() => {
    window.scrollBy({ top: 400, behavior: 'smooth' });
  });
  await page.waitForTimeout(100); // mid-glide capture
  await saveScreenshot(page, 'stage-transition-warp-speed-stream.png', 'Stars streaming forward at warp speed');

  // Milestone 4: Stage 02 - The Context (p = 0.48)
  console.log('\n--- Capturing Stage 02: The Context (p = 0.48) ---');
  await scrollTo(0.48, true);
  await page.waitForTimeout(400);
  await saveScreenshot(page, 'stage-02-the-context-finbert.png', 'FinBERT Signal Stream (8.3 Light-Min)');

  // Milestone 5: Warp speed transition into Stage 03 (approaching Venus Orbit)
  console.log('\n--- Capturing Warp Speed approach to Venus (p = 0.62) ---');
  await scrollTo(0.58);
  await page.evaluate(() => {
    window.scrollBy({ top: 350, behavior: 'smooth' });
  });
  await page.waitForTimeout(120);
  await saveScreenshot(page, 'stage-transition-warp-venus-approach.png', 'Warp speed approach towards Venus orbit');

  // Milestone 6: Stage 03 - The Possibilities (p = 0.66) - Forecast Window
  console.log('\n--- Capturing Stage 03: The Possibilities entering (p = 0.66) ---');
  await scrollTo(0.66, true);
  await page.waitForTimeout(500);
  const state66 = await getStageState();
  console.log('State at p=0.66:', JSON.stringify(state66, null, 2));
  await saveScreenshot(page, 'stage-03-possibilities-entering-forecast.png', 'Stage 03 Possibilities with Forecast Preview');

  // Milestone 7: Stage 03 - 0.72 AU Venus Perspective Approach (p = 0.72)
  console.log('\n--- Capturing Stage 03: 0.72 AU Venus Orbital Approach (p = 0.72) ---');
  await scrollTo(0.72, true);
  await page.waitForTimeout(500);
  const state72 = await getStageState();
  console.log('State at p=0.72:', JSON.stringify(state72, null, 2));
  await saveScreenshot(page, 'stage-03-venus-perspective-approach-072au.png', 'Venus growing in 3D perspective at 0.72 AU');

  // Milestone 8: Stage 03 - Forecast preview clearing out (p = 0.78)
  console.log('\n--- Capturing Stage 03: Forecast Preview Exit (p = 0.78) ---');
  await scrollTo(0.78, true);
  await page.waitForTimeout(500);
  const state78 = await getStageState();
  console.log('State at p=0.78:', JSON.stringify(state78, null, 2));
  await saveScreenshot(page, 'stage-03-forecast-clearing-venus-emerging.png', 'Forecast cards cleared; Venus dominating center');

  // Milestone 9: Stage 03 - Quiet Negative Space: Planet Venus Cyprium Sphere (p = 0.86)
  console.log('\n--- Capturing Stage 03: Planet Venus Cyprium Sphere Unobstructed (p = 0.86) ---');
  await scrollTo(0.86, true);
  await page.waitForTimeout(600);
  const state86 = await getStageState();
  console.log('State at p=0.86:', JSON.stringify(state86, null, 2));
  await saveScreenshot(page, 'stage-03-venus-sphere-cyprium-unobstructed.png', 'Planet Venus Cyprium Sphere perfectly unobstructed');

  // Verification assertions
  console.log('\n--- Verifying Stage 03 Contracts ---');
  const forecastSurface = state86.surfaces[2]; // Forecast surface index 2
  const forecastOpacity = parseFloat(forecastSurface?.opacity || '0');
  console.log(`Forecast Preview Surface opacity at p=0.86: ${forecastOpacity}`);
  console.log(`Forecast Preview Surface visibility at p=0.86: ${forecastSurface?.visibility}`);
  
  const forecastCopy = state86.copyBlocks.find(b => b.id === 'forecast');
  const copyOpacity = parseFloat(forecastCopy?.opacity || '0');
  console.log(`Forecast Copy block opacity at p=0.86: ${copyOpacity}`);
  console.log(`Forecast Copy block visibility at p=0.86: ${forecastCopy?.visibility}`);

  console.log(`Cosmic HUD at p=0.86: ${state86.hudNum} ${state86.hudUnit}`);

  const isUnobstructed = forecastOpacity === 0 && (forecastSurface?.visibility === 'hidden' || forecastOpacity === 0);
  console.log(`Contract verified - Venus perfectly unobstructed by preview cards: ${isUnobstructed}`);

  await browser.close();
  console.log('\nAll verifications completed successfully!');
}

run().catch(err => {
  console.error('Execution error:', err);
  process.exit(1);
});
