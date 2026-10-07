import { access, readFile, mkdir, writeFile } from 'node:fs/promises';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { chromium } from 'playwright';

const scriptDirectory = dirname(fileURLToPath(import.meta.url));
const brandDirectory = resolve(scriptDirectory, '../public/brand');
const fontPath = resolve(brandDirectory, '../fonts/geist-sans-v1.7.2.woff2');
const geometryPath = resolve(scriptDirectory, '../src/components/ui/brand-mark-geometry.json');

function vectorAssets({ starPath, compactStarPath }) {
  const smallCopper = `
    <linearGradient id="copper" x1="0" y1="0" x2="1" y2="1">
      <stop offset="0" stop-color="#dca17d"/><stop offset=".52" stop-color="#a86647"/><stop offset="1" stop-color="#704431"/>
    </linearGradient>`;
  const socialMark = `
    <svg x="78" y="127" width="132" height="132" viewBox="0 0 48 48" aria-label="Terra Rara mark">
      <defs>
        <linearGradient id="mark-copper" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f2c39f"/><stop offset=".34" stop-color="#d9956c"/><stop offset=".68" stop-color="#a96545"/><stop offset="1" stop-color="#704431"/></linearGradient>
        <linearGradient id="mark-ring" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#f2c39f" stop-opacity=".9"/><stop offset=".58" stop-color="#c9825b" stop-opacity=".64"/><stop offset="1" stop-color="#764832" stop-opacity=".82"/></linearGradient>
      </defs>
      <circle cx="24" cy="24" r="15.75" fill="none" stroke="url(#mark-ring)" stroke-width="1.35"/>
      <path d="${starPath}" transform="translate(.35 1.05)" fill="#4c2b25" opacity=".48"/>
      <path d="${starPath}" fill="url(#mark-copper)"/>
      <path d="M24 11.25 26.55 21.45 36.75 24" fill="none" stroke="#fff4e7" stroke-width=".6" stroke-linecap="round" opacity=".58"/>
      <circle cx="24" cy="24" r="1.15" fill="#fff8ef" opacity=".88"/>
    </svg>`;
  const favicon = `<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48"><defs>${smallCopper}</defs><circle cx="24" cy="24" r="23" fill="url(#copper)"/><path d="${compactStarPath}" transform="translate(0 .6)" fill="#4c2b25" opacity=".38"/><path d="${compactStarPath}" fill="#fff1df"/></svg>`;
  const appIcon = `<svg xmlns="http://www.w3.org/2000/svg" width="180" height="180" viewBox="0 0 180 180"><defs><linearGradient id="copper" x1="20" y1="24" x2="152" y2="154" gradientUnits="userSpaceOnUse"><stop offset="0" stop-color="#e1ad8c"/><stop offset=".48" stop-color="#a86647"/><stop offset="1" stop-color="#704431"/></linearGradient></defs><rect width="180" height="180" fill="#080e17"/><circle cx="90" cy="90" r="68" fill="url(#copper)"/><g transform="translate(90 90) scale(2.4) translate(-24 -24)"><path d="${compactStarPath}" transform="translate(0 .42)" fill="#4c2b25" opacity=".38"/><path d="${compactStarPath}" fill="#fff1df"/></g></svg>`;
  const social = `<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="630" viewBox="0 0 1200 630">
  <defs>
    <linearGradient id="background" x1="0" y1="0" x2="1" y2="1"><stop stop-color="#0b121b"/><stop offset="1" stop-color="#080e17"/></linearGradient>
    <radialGradient id="atmosphere" cx="0" cy="0" r="1" gradientTransform="translate(123 68) rotate(49) scale(700 690)" gradientUnits="userSpaceOnUse"><stop stop-color="#b76c48" stop-opacity=".22"/><stop offset="1" stop-color="#b76c48" stop-opacity="0"/></radialGradient>
  </defs>
  <rect width="1200" height="630" fill="url(#background)"/><rect width="1200" height="630" fill="url(#atmosphere)"/>
  <g fill="none" stroke="#e6a47a" stroke-opacity=".09"><circle cx="1018" cy="320" r="150"/><circle cx="1018" cy="320" r="181"/><circle cx="1018" cy="320" r="213"/><circle cx="1018" cy="320" r="246"/></g>
  <path d="M0 566H1200" stroke="#e6a47a" stroke-opacity=".18"/>
  ${socialMark}
  <text x="224" y="184" fill="#e6a47a" font-family="Geist Sans, Arial, sans-serif" font-size="13" font-weight="600" letter-spacing="3.4">COPPER INTELLIGENCE / TERRA RARA</text>
  <text x="224" y="226" fill="#f2f4f7" font-family="Geist Sans, Arial, sans-serif" font-size="28" font-weight="600" letter-spacing="4.2">COPPERMIND</text>
  <text x="225" y="251" fill="#a3afbf" font-family="Geist Sans, Arial, sans-serif" font-size="12" font-weight="500" letter-spacing="4.2">TERRA RARA</text>
  <text x="94" y="360" fill="#f2f4f7" font-family="Geist Sans, Arial, sans-serif" font-size="61" font-weight="500" letter-spacing="-2.5">Read the market.</text>
  <text x="94" y="432" fill="#e6a47a" font-family="Geist Sans, Arial, sans-serif" font-size="61" font-weight="500" letter-spacing="-2.5">See the structure.</text>
  <text x="98" y="489" fill="#b2bdcb" font-family="Geist Sans, Arial, sans-serif" font-size="18" font-weight="400">Market context, news, forecasts and evidence in one workspace.</text>
  <text x="96" y="603" fill="#a3afbf" font-family="Geist Sans, Arial, sans-serif" font-size="12" font-weight="500" letter-spacing="2.1">BUILT AROUND COPPER. DESIGNED FOR PERSPECTIVE.</text>
  <text x="1104" y="603" text-anchor="end" fill="#e6a47a" font-family="Geist Sans, Arial, sans-serif" font-size="12" font-weight="600" letter-spacing="1.9">TERRA RARA</text>
</svg>`;
  return [
    { path: resolve(scriptDirectory, '../public/favicon.svg'), content: favicon },
    { path: resolve(brandDirectory, 'terra-rara-app-icon.svg'), content: appIcon },
    { path: resolve(brandDirectory, 'terra-rara-social.svg'), content: social },
  ];
}
const assets = [
  { source: 'terra-rara-app-icon.svg', output: 'terra-rara-app-icon.png', width: 180, height: 180 },
  { source: 'terra-rara-social.svg', output: 'terra-rara-social.png', width: 1200, height: 630 },
];

await mkdir(brandDirectory, { recursive: true });
const geistFont = (await readFile(fontPath)).toString('base64');
const geometry = JSON.parse(await readFile(geometryPath, 'utf8'));
for (const asset of vectorAssets(geometry)) await writeFile(asset.path, asset.content);
const launchOptions = { headless: true };
if (process.env.BRAND_RENDER_BROWSER) {
  launchOptions.executablePath = process.env.BRAND_RENDER_BROWSER;
} else if (process.platform === 'win32') {
  const edgePaths = [process.env['ProgramFiles(x86)'], process.env.ProgramFiles]
    .filter(Boolean)
    .map(programFiles => resolve(programFiles, 'Microsoft/Edge/Application/msedge.exe'));
  for (const edgePath of edgePaths) {
    try {
      await access(edgePath);
      launchOptions.executablePath = edgePath;
      break;
    } catch {
      // Fall back to the Playwright-managed Chromium if Edge is not installed.
    }
  }
}
const browser = await chromium.launch(launchOptions);
try {
  for (const asset of assets) {
    const svg = await readFile(resolve(brandDirectory, asset.source), 'utf8');
    const page = await browser.newPage({ viewport: { width: asset.width, height: asset.height }, deviceScaleFactor: 1 });
    const fontFace = `@font-face{font-family:'Geist Sans';src:url(data:font/woff2;base64,${geistFont}) format('woff2');font-style:normal;font-weight:100 900;font-display:block}`;
    await page.setContent(`<!doctype html><html><head><style>${fontFace}html,body{width:${asset.width}px;height:${asset.height}px;margin:0;overflow:hidden;background:transparent}svg{display:block;width:${asset.width}px;height:${asset.height}px}</style></head><body>${svg}</body></html>`);
    await page.evaluate(() => document.fonts.ready);
    await page.screenshot({ path: resolve(brandDirectory, asset.output), type: 'png', clip: { x: 0, y: 0, width: asset.width, height: asset.height } });
    await page.close();
    process.stdout.write(`${asset.output} ${asset.width}x${asset.height}\n`);
  }
} finally {
  await browser.close();
}
