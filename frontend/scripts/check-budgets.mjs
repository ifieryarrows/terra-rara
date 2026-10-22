import { readFile } from 'node:fs/promises';
import { gzipSync } from 'node:zlib';

const manifest = JSON.parse(await readFile('dist/.vite/manifest.json', 'utf8'));
const entryKey = Object.keys(manifest).find(key => manifest[key].isEntry);
if (!entryKey) throw new Error('No production entry in build manifest');
function closure(key, found = new Set()) {
  if (found.has(key)) return found;
  found.add(key);
  for (const imported of manifest[key]?.imports || []) closure(imported, found);
  return found;
}
async function sizes(keys) {
  const js = new Set(), css = new Set();
  for (const key of keys) { js.add(manifest[key].file); for (const file of manifest[key].css || []) css.add(file); }
  const sum = async files => (await Promise.all([...files].map(async file => gzipSync(await readFile('dist/' + file)).length))).reduce((a, b) => a + b, 0);
  return { jsGzipBytes: await sum(js), cssGzipBytes: await sum(css) };
}
const initial = await sizes(closure(entryKey));
const overviewKey = Object.keys(manifest).find(key => manifest[key].name === 'OverviewPage');
if (!overviewKey) throw new Error('No Overview route in build manifest');
const dashboard = await sizes(new Set([...closure(entryKey), ...closure(overviewKey), ...closure('src/layouts/AppShell.tsx')]));
// Project budgets in gzip bytes, not web standards. Each limit keeps real margin above
// the current measurement (landing JS ~127 KB, shared CSS ~13.9 KB, dashboard ~280 KB) so a
// copy or style tweak does not turn CI red on a few hundred bytes, while a genuine
// dependency or asset regression still trips the gate. Keep the page payload itself flat:
// raise a limit only when the slack is gone, never to justify new weight.
const LIMITS = { initialJS: 190000, css: 15000, dashboardJS: 300000 };
const headroom = {
  initialJS: LIMITS.initialJS - initial.jsGzipBytes,
  css: LIMITS.css - initial.cssGzipBytes,
  dashboardJS: LIMITS.dashboardJS - dashboard.jsGzipBytes,
};
console.log(JSON.stringify({ initial, dashboardBeforeNewsAndHeatmap: dashboard, limits: LIMITS, headroomBytes: headroom }, null, 2));
const exceeded = Object.entries(headroom).filter(([, bytes]) => bytes < 0);
if (exceeded.length) {
  for (const [name, bytes] of exceeded) console.error(`Budget exceeded: ${name} is ${-bytes} gzip bytes above its ${LIMITS[name]} byte limit`);
  process.exitCode = 1;
}
const html = await readFile('dist/index.html', 'utf8');
if (!html.includes('Read the market.') || !html.includes('href="/dashboard"')) throw new Error('Critical landing HTML/CTA missing');
const workspace = await readFile('dist/workspace.html', 'utf8');
const workspaceBody = workspace.match(/<body\b[^>]*>([\s\S]*?)<\/body>/i)?.[1];
if (!workspaceBody) throw new Error('Workspace shell body missing');
if (workspaceBody.includes('Read the market.')) throw new Error('Workspace shell includes the marketing story');
