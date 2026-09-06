// Identical before/after local lab workload; not field INP, FPS or provider validation.
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { interceptDashboard, fixtureFor } from './dashboard-fixtures.mjs';
const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE || 'playwright');
const output = process.env.EXPERIENCE_OUTPUT || join(tmpdir(), 'coppermind-dashboard-profile');
await mkdir(output, { recursive: true });
const browser = await chromium.launch({ channel: 'msedge', headless: true });
const results = [];
try {
  for (let run = 0; run < 3; run++) {
    const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, reducedMotion: 'reduce' });
    // Preserve the original baseline workload: empty news feed, populated chart/map.
    await interceptDashboard(page, url => url.pathname === '/api/news' ? { payload: { ...fixtureFor(url), items: [], total: 0 } } : url.pathname === '/api/news/stats' ? { payload: { ...fixtureFor(url), total_articles: 0, label_distribution: { NEUTRAL: 0 } } } : undefined);
    const client = await page.context().newCDPSession(page);
    await client.send('Network.enable');
    await client.send('Network.setCacheDisabled', { cacheDisabled: true });
    await client.send('Emulation.setCPUThrottlingRate', { rate: 4 });
    await client.send('Performance.enable');
    await page.addInitScript(() => {
      window.qaPerf = { longTasks: [], lcp: 0, cls: 0, events: [] };
      new PerformanceObserver(list => list.getEntries().forEach(e => window.qaPerf.longTasks.push({ start: e.startTime, duration: e.duration }))).observe({ type: 'longtask', buffered: true });
      new PerformanceObserver(list => list.getEntries().forEach(e => { window.qaPerf.lcp = e.startTime; })).observe({ type: 'largest-contentful-paint', buffered: true });
      new PerformanceObserver(list => list.getEntries().forEach(e => { if (!e.hadRecentInput) window.qaPerf.cls += e.value; })).observe({ type: 'layout-shift', buffered: true });
      new PerformanceObserver(list => list.getEntries().forEach(e => { if (e.interactionId) window.qaPerf.events.push({ name: e.name, duration: e.duration }); })).observe({ type: 'event', buffered: true, durationThreshold: 16 });
    });
    await page.goto((process.env.EXPERIENCE_URL || 'http://127.0.0.1:5185') + '/dashboard');
    await page.locator('#price-forecast .recharts-surface').waitFor();
    await page.getByRole('heading', { name: 'Market Heatmap', exact: true }).waitFor();
    await page.waitForTimeout(1200);
    const start = await page.evaluate(() => performance.now());
    for (let i = 0; i < 4; i++) {
      await page.getByText('View chart data', { exact: true }).click();
      await page.locator('#price-forecast').scrollIntoViewIfNeeded();
      const box = await page.locator('#price-forecast .recharts-surface').boundingBox();
      await page.mouse.move(box.x + box.width * .3, box.y + box.height * .5);
      await page.mouse.move(box.x + box.width * .8, box.y + box.height * .5, { steps: 20 });
      await page.locator('#market-map').scrollIntoViewIfNeeded();
      await page.waitForTimeout(600);
    }
    const measurements = await page.evaluate(() => ({ ...window.qaPerf, elapsed: performance.now() }));
    const metrics = (await client.send('Performance.getMetrics')).metrics;
    results.push({ run, start, ...measurements, heapUsed: metrics.find(m => m.name === 'JSHeapUsedSize')?.value, interactionLongTasks: measurements.longTasks.filter(t => t.start >= start) });
    if (!run) await page.screenshot({ path: join(output, 'dashboard.png'), fullPage: true });
    await page.close();
  }
  await writeFile(join(output, 'profile.json'), JSON.stringify({ environment: 'Headless Edge, 1440x900, 4x CPU, cold cache, local fixtures, reduced motion; 3 runs', results }, null, 2));
  console.log(JSON.stringify({ output, results: results.map(r => ({ run: r.run, lcp: r.lcp, cls: r.cls, heapUsed: r.heapUsed, interactionMs: r.elapsed - r.start, interactionLongTasks: r.interactionLongTasks, events: r.events })) }));
} finally { await browser.close(); }
