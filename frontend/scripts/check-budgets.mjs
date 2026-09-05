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
console.log(JSON.stringify({ initial, dashboardBeforeNewsAndHeatmap: dashboard, limits: { initialJS: 190000, css: 14000, dashboardJS: 280000 } }, null, 2));
if (initial.jsGzipBytes > 190000 || initial.cssGzipBytes > 14000 || dashboard.jsGzipBytes > 280000) process.exitCode = 1;
const html = await readFile('dist/index.html', 'utf8');
if (!html.includes('Read the market.') || !html.includes('href="/dashboard"')) throw new Error('Critical landing HTML/CTA missing');
const workspace = await readFile('dist/workspace.html', 'utf8');
if (workspace.includes('Read the market.')) throw new Error('Workspace shell includes the marketing story');
