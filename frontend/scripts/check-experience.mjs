// Optional browser QA runner; uses an externally installed Playwright package.
// PLAYWRIGHT_MODULE may point to that package; BROWSER_CHANNEL defaults to msedge.
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';

const require = createRequire(import.meta.url);
const { chromium } = require(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.EXPERIENCE_URL || 'http://127.0.0.1:5174';
const output = process.env.EXPERIENCE_OUTPUT || join(tmpdir(), 'coppermind-experience-qa');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ headless: true, channel: process.env.BROWSER_CHANNEL || 'msedge' });
const results = [];
const workspaceResults = [];
try {
  for (const [width, height, reducedMotion] of [[1536, 900, 'no-preference'], [1280, 600, 'no-preference'], [1024, 650, 'no-preference'], [768, 800, 'no-preference'], [390, 844, 'no-preference'], [1280, 720, 'reduce']]) {
    const page = await browser.newPage({ viewport: { width, height }, reducedMotion });
    const apiRequests = [];
    const errors = [];
    page.on('request', request => { if (new URL(request.url()).pathname.startsWith('/api/')) apiRequests.push(request.url()); });
    page.on('pageerror', error => errors.push(error.message));
    await page.goto(base);
    await page.locator('.cm-story').waitFor();
    await page.waitForTimeout(400);
    assert.equal(await page.locator('h1').count(), 1);
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    assert.ok(overflow <= 1, `Landing overflow ${width}: ${overflow}`);
    const enhanced = width >= 1024 && reducedMotion !== 'reduce';
    assert.equal(await page.locator('.cm-story--enhanced').count(), Number(enhanced));
    const samples = [];
    if (enhanced) {
      const story = await page.locator('.cm-story').boundingBox();
      for (const progress of [0, .5, .94]) {
        await page.evaluate(y => scrollTo(0, y), story.y + (story.height - height) * progress);
        await page.waitForTimeout(100);
        const sample = await page.evaluate(() => {
          const stage = document.querySelector('.cm-story-stage').getBoundingClientRect();
          const inner = document.querySelector('.cm-story-stage-inner').getBoundingClientRect();
          return { top: stage.top, height: stage.height, innerHeight: inner.height,
            layers: [...document.querySelectorAll('.cm-story-layer')].map(e => Number(getComputedStyle(e).opacity)) };
        });
        assert.ok(Math.abs(sample.top) < 2, `Sticky stage ${width}x${height}: ${sample.top}`);
        assert.ok(sample.height <= height + 1, `Stretched grid stage ${sample.height}`);
        assert.ok(sample.innerHeight <= height, `Preview cropped ${width}x${height}: ${sample.innerHeight}`);
        assert.ok(sample.layers[samples.length] > .95, `Wrong chapter: ${sample.layers}`);
        samples.push(sample);
      }
    } else {
      assert.equal(await page.locator('.cm-story--static article').count(), 3);
    }
    await page.screenshot({ path: join(output, `landing-${width}-${height}-${reducedMotion}.png`) });
    assert.deepEqual(apiRequests, [], 'Landing must not fetch financial APIs');
    assert.deepEqual(errors, []);
    results.push({ viewport: { width, height }, reducedMotion, overflow, enhanced, samples, apiRequests, errors });
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
    page.on('pageerror', error => errors.push(error.message));
    await page.route('**/api/**', route => {
      const payload = fixtures[new URL(route.request().url()).pathname];
      return route.fulfill({ status: payload ? 200 : 503, contentType: 'application/json', body: JSON.stringify(payload || { detail: 'Controlled QA: service unavailable' }) });
    });
    for (const path of ['/models', '/validation', '/system']) {
      await page.goto(base + path);
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
    assert.deepEqual(errors, []);
    await page.close();
  }
  await writeFile(join(output, 'results.json'), JSON.stringify({ base, checkedAt: new Date().toISOString(), results, workspaceResults }, null, 2));
  console.log(JSON.stringify({ landingPassed: results.length, workspacePassed: workspaceResults.length, output, results, workspaceResults }, null, 2));
} finally { await browser.close(); }
