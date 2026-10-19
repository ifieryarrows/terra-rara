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
    await page.getByRole('button', { name: 'Last 30 daily closes', exact: true }).waitFor();
    await page.locator('#price-forecast .recharts-surface').waitFor();
    // Below-fold panels mount on approach: visit the news section before asserting it.
    await page.locator('#news-intelligence').scrollIntoViewIfNeeded();
    await page.getByRole('button', { name: 'News filters', exact: true }).waitFor();
    await page.locator('#price-forecast').scrollIntoViewIfNeeded();
    await page.waitForTimeout(600);
    const chartRequests = requests.filter(url => /\/api\/(history|analysis)/.test(url)).length;
    await page.getByRole('button', { name: 'Last 90 daily closes', exact: true }).click();
    await page.getByRole('button', { name: 'Last 180 daily closes', exact: true }).click();
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
    await page.getByRole('button', { name: 'Last 30 daily closes', exact: true }).click();
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
    const dialog = page.getByRole('dialog', { name: /NEWS INTELLIGENCE/i });
    await dialog.waitFor();
    assert.equal(await page.evaluate(() => document.body.style.overflow), 'hidden');
    await page.keyboard.press('Escape');
    await dialog.waitFor({ state: 'hidden' });
    assert.equal(await newsCard.evaluate(el => document.activeElement === el), true, 'Drawer focus restoration');
    assert.notEqual(await page.evaluate(() => document.body.style.overflow), 'hidden');
    await page.getByRole('button', { name: 'News filters', exact: true }).click();

    // The filter controls are button groups now (aria-pressed chips), not selects.
    await page.getByRole('button', { name: 'Technology', exact: true }).click();
    await page.getByRole('button', { name: 'Change', exact: true }).click();
    await page.getByRole('button', { name: 'All categories', exact: true }).click();
    await page.getByRole('button', { name: 'Weight', exact: true }).click();
    assert.equal(await page.getByRole('button', { name: 'All categories', exact: true }).getAttribute('aria-pressed'), 'true');
    assert.equal(await page.getByRole('button', { name: 'Weight', exact: true }).getAttribute('aria-pressed'), 'true');
    // The map panel mounts on approach; scroll its section into view first.
    await page.locator('#market-map').scrollIntoViewIfNeeded();
    // Wheel zoom is the current zoom mechanism; the map mirrors it in data-zoomed.
    const map = page.locator('.cm-heatmap-map');
    await map.scrollIntoViewIfNeeded();
    const mapBox = await map.boundingBox();
    assert.ok(mapBox, 'Market map must be visible');
    const zoomX = mapBox.x + mapBox.width / 2;
    const zoomY = mapBox.y + mapBox.height / 2;
    const zoomPointInsideMap = await page.evaluate(([x, y]) => document.querySelector('.cm-heatmap-map')?.contains(document.elementFromPoint(x, y)), [zoomX, zoomY]);
    assert.ok(zoomPointInsideMap, 'Zoom point must land inside the map');
    await page.mouse.move(zoomX, zoomY);
    await page.mouse.wheel(0, -240);
    await page.waitForFunction(() => document.querySelector('.cm-heatmap-map')?.dataset.zoomed === 'true');
    await page.mouse.wheel(0, 240);
    await page.waitForFunction(() => document.querySelector('.cm-heatmap-map')?.dataset.zoomed === 'false');
    await page.getByRole('button', { name: 'Open fullscreen', exact: true }).click();
    const fullscreen = page.getByRole('dialog', { name: 'Fullscreen market map' });
    await fullscreen.waitFor();
    assert.equal(await page.evaluate(() => document.body.style.overflow), 'hidden');
    const lastControl = fullscreen.getByRole('link', { name: 'Logos by Logo.dev' });
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
    assert.equal(await page.getByRole('button', { name: 'Open fullscreen', exact: true }).evaluate(el => document.activeElement === el), true);
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
      const outlook = await page.locator('#weekly-outlook').innerText();
      if (state === 'stale') {
        // The chart drops a forecast whose baseline close is not aligned with the history;
        // the weekly card surfaces the stale baseline date rather than bridging it.
        assert.ok(outlook.includes('CLOSE DATE') && outlook.includes('2026-01-01'), `Stale baseline date must be visible: ${outlook.slice(0, 200)}`);
      } else {
        assert.ok(outlook.includes('FORECAST DEGRADED'), `Degraded notice must be visible: ${outlook.slice(0, 200)}`);
        assert.ok(outlook.includes('TFT weekly forecast is unavailable until the weekly model artifacts are refreshed.'));
      }
    } else if (state === 'empty') await page.getByRole('heading', { name: 'No chart data available', exact: true }).waitFor();
    else if (state === 'failure') await page.getByRole('heading', { name: 'Price history could not be loaded' }).waitFor();
    else {
      fail = true;
      await page.getByRole('button', { name: 'Refresh overview' }).click();
      await page.getByText('Some data could not be refreshed', { exact: true }).waitFor();
      assert.ok(await page.locator('.cm-dashboard-alerts').innerText().then(t => t.includes('previous response')));
      assert.equal(await page.getByRole('button', { name: 'Last 30 daily closes' }).isVisible(), true);
    }
    results.push({ state, passed: true });
    await page.close();
  }
  await writeFile(join(output, 'results.json'), JSON.stringify({ base, checkedAt: new Date().toISOString(), fixtures: true, results }, null, 2));
  console.log(JSON.stringify({ output, results }));
} finally { await browser.close(); }
