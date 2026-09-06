import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { interceptDashboard, tft } from './dashboard-fixtures.mjs';
const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE || 'playwright');
const output = process.env.EXPERIENCE_OUTPUT || join(tmpdir(), 'coppermind-dashboard-qa');
const base = process.env.EXPERIENCE_URL || 'http://127.0.0.1:5185';
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const results = [];
try {
  for (const [width, height, reducedMotion] of [[1536, 900, 'no-preference'], [1280, 600, 'no-preference'], [768, 900, 'no-preference'], [390, 844, 'reduce'], [320, 650, 'reduce']]) {
    const page = await browser.newPage({ viewport: { width, height }, reducedMotion });
    const errors = [];
    const requests = [];
    page.on('pageerror', e => errors.push(e.message));
    page.on('request', r => { if (new URL(r.url()).pathname.startsWith('/api/')) requests.push(r.url()); });
    await interceptDashboard(page);
    await page.goto(base + '/dashboard');
    await page.getByRole('button', { name: '30 closes', exact: true }).waitFor();
    await page.locator('#price-forecast .recharts-surface').waitFor();
    await page.getByRole('button', { name: 'News filters', exact: true }).waitFor();
    await page.waitForTimeout(600);
    const chartRequests = requests.filter(url => /\/api\/(history|analysis)/.test(url)).length;
    await page.getByRole('button', { name: '90 closes', exact: true }).click();
    await page.getByRole('button', { name: '180 closes', exact: true }).click();
    const summary = page.locator('#price-forecast summary');
    await summary.focus();
    await page.keyboard.press('Enter');
    await page.locator('#price-forecast table').waitFor();
    assert.equal(await page.locator('#price-forecast tbody tr').count(), 185);
    const table = page.locator('#price-forecast .cm-table-scroll');
    if (width <= 390) {
      await table.focus();
      await page.keyboard.press('ArrowRight');
      await page.waitForTimeout(200);
      assert.ok(await table.evaluate(el => el.scrollLeft) > 0, 'Mobile table keyboard scroll');
    }
    await summary.click();
    await page.getByRole('button', { name: 'Forecast median', exact: true }).click();
    assert.equal(await page.getByRole('button', { name: 'Forecast median', exact: true }).getAttribute('aria-pressed'), 'false');
    await page.getByRole('button', { name: 'Forecast median', exact: true }).click();
    await page.getByRole('button', { name: 'Q10–Q90 range', exact: true }).click();
    await page.getByRole('button', { name: 'Q10–Q90 range', exact: true }).click();
    await page.getByRole('button', { name: '30 closes', exact: true }).click();
    assert.equal(requests.filter(url => /\/api\/(history|analysis)/.test(url)).length, chartRequests, 'Chart controls must not fetch new data');
    await page.locator('#price-forecast .recharts-surface').focus();
    await page.keyboard.press('ArrowRight');
    await page.waitForTimeout(200);
    assert.ok(await page.locator('.cm-chart-tooltip').isVisible(), 'Chart keyboard tooltip');

    await page.getByRole('button', { name: 'News filters', exact: true }).click();
    await page.getByRole('button', { name: 'NewsAPI', exact: true }).click();
    await page.getByRole('button', { name: 'All channels', exact: true }).click();
    await page.getByRole('searchbox', { name: 'Search headlines' }).fill('no-match');
    await page.getByRole('heading', { name: 'No matching headlines' }).waitFor();
    await page.getByRole('button', { name: 'Show all headlines', exact: true }).click();
    const newsCard = page.getByRole('button', { name: /QA fixture: copper market update 1/ });
    await newsCard.click();
    const dialog = page.getByRole('dialog', { name: 'News detail' });
    await dialog.waitFor();
    assert.equal(await page.evaluate(() => document.body.style.overflow), 'hidden');
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'hidden' });
    assert.equal(await newsCard.evaluate(el => document.activeElement === el), true, 'Drawer focus restoration');
    assert.notEqual(await page.evaluate(() => document.body.style.overflow), 'hidden');
    await page.getByRole('button', { name: 'News filters', exact: true }).click();

    await page.getByRole('combobox', { name: 'Filter top-level category' }).selectOption('Technology');
    await page.getByRole('combobox', { name: 'Cell sizing' }).selectOption('Performance');
    await page.getByRole('button', { name: 'Reset map filters' }).click();
    assert.equal(await page.getByRole('combobox', { name: 'Filter top-level category' }).inputValue(), 'ALL');
    await page.getByRole('button', { name: 'Zoom in', exact: true }).click();
    assert.equal(await page.getByRole('button', { name: 'Reset map zoom' }).innerText(), '150%');
    await page.getByRole('button', { name: 'Reset map zoom' }).click();
    assert.equal(await page.getByRole('button', { name: 'Zoom out', exact: true }).isDisabled(), true);
    await page.getByRole('button', { name: 'Fullscreen', exact: true }).click();
    const fullscreen = page.getByRole('dialog', { name: 'Fullscreen market map' });
    await fullscreen.waitFor();
    assert.equal(await page.evaluate(() => document.body.style.overflow), 'hidden');
    const lastControl = fullscreen.getByRole('link', { name: 'Logos provided by Logo.dev' });
    await lastControl.focus();
    await page.keyboard.press('Tab');
    assert.equal(await fullscreen.evaluate(el => el.contains(document.activeElement)), true, 'Fullscreen focus containment');
    assert.ok(await fullscreen.evaluate(el => el.getBoundingClientRect().height <= innerHeight + 1));
    // The first Escape dismisses an active category panel before the map.
    // Keep the pointer off the map: typography can move a tile beneath it.
    await page.mouse.move(1, 1);
    await page.waitForTimeout(250);
    await page.keyboard.press('Escape');
    if (await fullscreen.isVisible()) await page.keyboard.press('Escape');
    await fullscreen.waitFor({ state: 'hidden' });
    assert.equal(await page.getByRole('button', { name: 'Fullscreen', exact: true }).evaluate(el => document.activeElement === el), true);
    assert.notEqual(await page.evaluate(() => document.body.style.overflow), 'hidden');
    const overflow = await page.evaluate(() => document.documentElement.scrollWidth - innerWidth);
    assert.ok(overflow <= 1, `Page overflow ${width}: ${overflow}`);
    assert.deepEqual(errors, []);
    await page.evaluate(() => scrollTo(0, 0));
    await page.screenshot({ path: join(output, `dashboard-${width}-${height}.png`), fullPage: true });
    results.push({ width, height, reducedMotion, overflow, passed: true });
    await page.close();
  }
  for (const state of ['stale', 'degraded', 'empty', 'failure', 'refresh-failure']) {
    const page = await browser.newPage({ viewport: { width: 1280, height: 800 }, reducedMotion: 'reduce' });
    let fail = false;
    await interceptDashboard(page, url => {
      if (state === 'failure' || fail) return { status: 503, payload: { detail: 'Controlled failure' } };
      if (state === 'empty' && url.pathname === '/api/history') return { payload: { data: [] } };
      if (url.pathname.startsWith('/api/analysis/tft/')) {
        if (state === 'stale') return { payload: { ...tft, prediction: { ...tft.prediction, reference_price_date: '2026-01-01' } } };
        if (state === 'degraded') return { payload: { ...tft, quality_state: 'degraded' } };
      }
    });
    await page.goto(base + '/dashboard');
    await page.getByRole('button', { name: 'Refresh overview' }).waitFor();
    if (state === 'stale' || state === 'degraded') {
      assert.equal(await page.getByRole('button', { name: 'Forecast median', exact: true }).isDisabled(), true);
      assert.ok(await page.locator('#price-forecast').innerText().then(t => t.includes(state === 'stale' ? 'Waiting for a forecast based on the latest close' : 'Forecast is degraded')));
    } else if (state === 'empty') await page.getByRole('heading', { name: 'No chart data available', exact: true }).waitFor();
    else if (state === 'failure') await page.getByRole('heading', { name: 'Price history could not be loaded' }).waitFor();
    else {
      fail = true;
      await page.getByRole('button', { name: 'Refresh overview' }).click();
      await page.getByText('Price history could not be refreshed; any visible values are from the previous response', { exact: true }).waitFor();
      assert.equal(await page.getByRole('button', { name: '30 closes' }).isVisible(), true);
    }
    results.push({ state, passed: true });
    await page.close();
  }
  await writeFile(join(output, 'results.json'), JSON.stringify({ base, checkedAt: new Date().toISOString(), fixtures: true, results }, null, 2));
  console.log(JSON.stringify({ output, results }));
} finally { await browser.close(); }
