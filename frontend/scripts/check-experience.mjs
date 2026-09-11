// Optional browser QA runner; uses an externally installed Playwright package.
// PLAYWRIGHT_MODULE may point to that package; BROWSER_CHANNEL defaults to msedge.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.EXPERIENCE_URL || 'http://127.0.0.1:5173';
const output = process.env.EXPERIENCE_OUTPUT || join(tmpdir(), 'coppermind-experience-qa');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: process.env.BROWSER_CHANNEL || 'msedge' });
const results = [];
const workspaceResults = [];
const adaptiveResults = [];
try {
  for (const [width, height, reducedMotion] of [[1536, 900, 'no-preference'], [1280, 600, 'no-preference'], [1024, 650, 'no-preference'], [768, 800, 'no-preference'], [320, 650, 'no-preference'], [390, 844, 'no-preference'], [1280, 720, 'reduce']]) {
    const page = await browser.newPage({ viewport: { width, height }, reducedMotion });
    const apiRequests = [];
    const errors = [];
    page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.locator('.cm-story').waitFor();
    await page.evaluate(() => document.fonts.ready);
    await page.waitForTimeout(400);
    assert.equal(await page.locator('.cm-evidence-preview').count(), 0, 'Landing moves directly from possibilities to the research CTA');
    assert.equal(await page.locator('.cm-news-intelligence-preview').count(), 1, 'News preview keeps the production-shaped sequence');
    assert.equal(await page.locator('.cm-news-sequence-layer').count(), 4, 'News sequence has headline, entity, dial and two-read layers');
    const landingText = await page.locator('body').textContent();
    assert.ok(landingText?.includes('LLM rationale'), 'News preview exposes article-level rationale');
    assert.ok(landingText?.includes('SIGNAL READ'), 'News preview exposes the rotating signal read');
    for (const href of ['#market', '#news', '#forecast']) {
      assert.equal(await page.locator(href).count(), 1);
    }
    assert.equal(await page.locator('h1').count(), 1);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    assert.ok(overflow <= 1, `Landing overflow ${width}: ${overflow}`);
    const enhanced = reducedMotion !== 'reduce';
    assert.equal(await page.locator('.cm-story--enhanced').count(), Number(enhanced));
    const samples = [];
    if (enhanced) {
      const story = await page.locator('.cm-story').boundingBox();
      assert.equal(await page.locator('.cm-particle-world').count(), 2, 'WebGL and fallback canvases share one particle field');
      const particle = await page.locator('.cm-particle-world[data-renderer]').evaluate(canvas => ({ renderer: canvas.dataset.renderer, quality: canvas.dataset.quality, count: Number(canvas.dataset.particleCount) }));
      assert.ok(['webgl2', 'canvas2d'].includes(particle.renderer), `Unsupported particle renderer: ${particle.renderer}`);
      const expectedQuality = particle.renderer === 'webgl2' && width >= 1024 ? 'high' : 'balanced';
      assert.equal(particle.quality, expectedQuality);
      assert.equal(particle.count, expectedQuality === 'high' ? 640 : 420);
      if ((width === 1536 || width === 390) && reducedMotion === 'no-preference') await page.screenshot({ path: join(output, `landing-${width}-${height}-hero.png`) });
      assert.equal(await page.locator('.cm-background-word').count(), 5, 'The global typography follows the continuous research story');
      assert.equal(await page.locator('.cm-cinematic-beat').count(), 4, 'Hero and three research beats share one scroll scene');
      assert.equal(await page.locator('.cm-cinematic-surface').count(), 3, 'Dashboard fragments stay inside the shared world');
      assert.equal(await page.locator('.cm-atmosphere-stars').count(), 1, 'Global atmosphere keeps one shared star field');
      if (height > 650) {
        assert.ok(story.height >= height * 3.9 && story.height <= height * 4.2, `Pinned story tail is too long or short: ${story.height}`);
      }
      // The forecast now exits before the particle ingot tail, so sample its
      // settled state before the negative-space handoff begins.
      for (const [sampleIndex, progress] of [0, .25, .5, .68].entries()) {
        const targetY = story.y + (story.height - height) * progress;
        await page.evaluate(y => { document.documentElement.scrollTop = y; }, targetY);
        await page.waitForFunction(y => Math.abs(scrollY - y) < 2, targetY);
        if (sampleIndex > 0) {
          await page.waitForFunction(expectedIndex => {
            const surface = document.querySelectorAll('.cm-cinematic-surface')[expectedIndex];
            return Boolean(surface && Number(getComputedStyle(surface).opacity) > .9);
          }, sampleIndex - 1);
        }
        const sample = await page.evaluate(() => {
          const stage = document.querySelector('.cm-cinematic-sticky').getBoundingClientRect();
          const canvas = document.querySelector('.cm-particle-world[data-renderer]').getBoundingClientRect();
          return { scrollY, top: stage.top, height: stage.height, canvas: { width: canvas.width, height: canvas.height },
            symbol: Number(getComputedStyle(document.querySelector('.cm-cinematic-symbol')).opacity),
            surfaces: [...document.querySelectorAll('.cm-cinematic-surface')].map(e => Number(getComputedStyle(e).opacity)),
            surfaceX: [...document.querySelectorAll('.cm-cinematic-surface')].map(e => new DOMMatrix(getComputedStyle(e).transform).m41),
            words: [...document.querySelectorAll('.cm-background-word')].map(e => Number(getComputedStyle(e).opacity)) };
        });
        assert.ok(Math.abs(sample.top) < 2, `Sticky stage ${width}x${height}: ${sample.top}`);
        assert.ok(sample.height <= height + 1, `Stretched grid stage ${sample.height}`);
        assert.ok(sample.canvas.width > 0 && sample.canvas.height > 0, 'Particle canvas fills the pinned world');
        if (sampleIndex === 0) assert.ok(sample.symbol > .95, `Hero symbol missing: ${sample.symbol}`);
        else {
          assert.ok(sample.surfaces[sampleIndex - 1] > .9, `Wrong continuous composition at progress ${progress}: ${sample.surfaces}`);
          assert.ok(Math.abs(sample.surfaceX[sampleIndex - 1]) > .5, `Pinned surface lost horizontal scroll translation at progress ${progress}: ${sample.surfaceX}`);
        }
        samples.push(sample);
      }
    } else {
      assert.equal(await page.locator('.cm-story--static article').count(), 3);
    }
    const forbidden = await page.locator('body').innerText();
    assert.equal(/Cu\s*\/\s*29|naive scroll continous signal|native scroll\s*\/\s*continuous signal/i.test(forbidden), false);
    await page.screenshot({ path: join(output, `landing-${width}-${height}-${reducedMotion}.png`) });
    assert.deepEqual(apiRequests, [], 'Landing must not fetch financial APIs');
    assert.deepEqual(errors, []);
    results.push({ viewport: { width, height }, reducedMotion, overflow, enhanced, samples, apiRequests, errors });
    await page.close();
  }
  for (const mode of ['save-data', 'low-cpu']) {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, reducedMotion: 'no-preference' });
    await page.addInitScript(fallbackMode => {
      if (fallbackMode === 'save-data') Object.defineProperty(navigator, 'connection', { configurable: true, value: { saveData: true, addEventListener() {}, removeEventListener() {} } });
      if (fallbackMode === 'low-cpu') Object.defineProperty(navigator, 'hardwareConcurrency', { configurable: true, value: 2 });
    }, mode);
    await page.goto(base);
    await page.locator('.cm-story').waitFor();
    await page.waitForTimeout(300);
    assert.equal(await page.locator('.cm-story--static').count(), 1, `${mode} must use the complete static fallback`);
    assert.equal(await page.locator('.cm-particle-world').count(), 0, `${mode} must not start the particle renderer`);
    adaptiveResults.push({ mode, static: true, canvases: 0 });
    await page.close();
  }
  {
    const page = await browser.newPage({ viewport: { width: 1280, height: 720 }, reducedMotion: 'no-preference' });
    await page.addInitScript(() => {
      const original = HTMLCanvasElement.prototype.getContext;
      HTMLCanvasElement.prototype.getContext = function(type, ...args) {
        if (type === 'webgl2') return null;
        return original.call(this, type, ...args);
      };
    });
    await page.goto(base);
    const fallbackCanvas = page.locator('.cm-particle-world[data-renderer="canvas2d"]');
    await fallbackCanvas.waitFor();
    const fallback = await fallbackCanvas.evaluate(element => ({ renderer: element.dataset.renderer, quality: element.dataset.quality, count: Number(element.dataset.particleCount) }));
    assert.deepEqual(fallback, { renderer: 'canvas2d', quality: 'balanced', count: 420 });
    adaptiveResults.push({ mode: 'webgl-unavailable', ...fallback });
    await page.close();
  }
  // Deliberately synthetic API responses: these checks establish UI behavior,
  // not provider availability, model quality or production performance.
  const fixtures = {
    '/api/models/tft/summary': { symbol: 'HG=F', trained_at: '2026-09-01T12:00:00Z', metrics: { weekly_directional_accuracy: .54, weekly_sample_count: 100, weekly_sharpe_ratio: .3, weekly_sortino_ratio: .4, weekly_magnitude_ratio: .95, weekly_tail_capture_rate: .5, weekly_raw_magnitude_ratio: 1.2, weekly_median_bound_applied_rate: .1, weekly_pi80_coverage: .8, directional_accuracy: .49, mae: .02 }, quality_gate: { passed: true, reasons: [], warnings: [] } },
    '/api/models/tft/backtest/latest': { available: true, report_date: '2026-09-01T12:00:00Z', summary_metrics: { mean_da: .54, mean_sharpe: .3, mean_mae: .02, mean_vr: 1.1 }, window_metrics: [{ da: .54, sharpe: .3, mae: .02, rmse: .03, variance_ratio: 1.1 }], theta_comparison: { tft_da: .54, theta_da: .51, tft_mae: .02, theta_mae: .03, tft_sharpe: .3, theta_sharpe: .2 } },
    '/api/health': { status: 'healthy', db_type: 'postgresql', redis_ok: true, pipeline_locked: false, models_found: 1, last_snapshot_age_seconds: 3600, news_count: 25, price_bars_count: 100, timestamp: '2026-09-01T12:00:00Z' },
  };
  for (const width of [1536, 390]) {
    const page = await browser.newPage({ viewport: { width, height: 900 } });
    const errors = [];
    const cinematicAssets = [];
    page.on('pageerror', error => errors.push(error.message));
    page.on('request', request => { if (request.url().includes('/assets/CinematicLanding-')) cinematicAssets.push(request.url()); });
    await page.route('**/api/**', route => {
      const payload = fixtures[new URL(route.request().url()).pathname];
      return route.fulfill({ status: payload ? 200 : 503, contentType: 'application/json', body: JSON.stringify(payload || { detail: 'Controlled QA: service unavailable' }) });
    });
    for (const path of ['/models', '/validation', '/system']) {
      await page.goto(new URL(path, base).href);
      try { await page.locator('.cm-page-header').waitFor(); }
      catch (error) {
        console.error(JSON.stringify({ path, errors, body: await page.locator('body').innerText() }));
        throw error;
      }
      const heading = path === '/models' ? 'Weekly strategy' : path === '/validation' ? 'Against the Theta baseline' : 'Core services';
      await page.getByRole('heading', { name: heading, exact: true }).waitFor();
      const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
      assert.ok(overflow <= 1, `${path} overflow ${width}: ${overflow}`);
      assert.equal(await page.getByRole('navigation', { name: 'Workspace', exact: true }).getByRole('link').count(), 4);
      if (path === '/models') {
        assert.equal(await page.getByText('Daily Directional Accuracy', { exact: true }).isVisible(), false);
        await page.locator('.cm-diagnostics summary').focus();
        await page.keyboard.press('Enter');
        assert.equal(await page.getByText('Daily Directional Accuracy', { exact: true }).isVisible(), true);
        await page.locator('.cm-diagnostics summary').click();
      }
      if (path === '/validation') {
        assert.equal(await page.getByRole('table').count(), 2);
        assert.ok(await page.getByText('54.00%', { exact: true }).count() >= 3);
      }
      await page.evaluate(() => scrollTo(0, 0));
      await page.screenshot({ path: join(output, `fixture-${path.slice(1)}-${width}.png`), fullPage: true });
      workspaceResults.push({ path, width, overflow, fixture: true });
    }
    assert.deepEqual(cinematicAssets, [], 'Direct workspace routes must not load landing cinematic assets');
    assert.deepEqual(errors, []);
    await page.close();
  }
  await writeFile(join(output, 'results.json'), JSON.stringify({ base, checkedAt: new Date().toISOString(), results, adaptiveResults, workspaceResults }, null, 2));
  console.log(JSON.stringify({ landingPassed: results.length, adaptivePassed: adaptiveResults.length, workspacePassed: workspaceResults.length, output, results, adaptiveResults, workspaceResults }, null, 2));
} finally { await browser.close(); }
