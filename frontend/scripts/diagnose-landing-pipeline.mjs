// Controlled style ablation for locating fill-rate and compositor bottlenecks.
// It does not modify application source or claim production/field performance.
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';

const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.EXPERIENCE_URL || 'http://127.0.0.1:4173';
const duration = Number(process.env.LANDING_DIAGNOSTIC_DURATION || 3000);
const diagnosticOutput = process.env.LANDING_DIAGNOSTIC_OUTPUT ? resolve(process.env.LANDING_DIAGNOSTIC_OUTPUT) : null;
const variants = [
  { name: 'normal', css: '' },
  { name: 'legacy-runtime-turbulence', css: `.cm-atmosphere-grain{inset:-50%!important;opacity:.19!important;transform:rotate(4deg)!important;mix-blend-mode:soft-light!important;background-size:auto!important;background-image:url("data:image/svg+xml,%3Csvg viewBox='0 0 180 180' xmlns='http://www.w3.org/2000/svg'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.82' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='.18'/%3E%3C/svg%3E")!important}` },
  { name: 'without-particles', css: '.cm-particle-world{display:none!important}' },
  { name: 'without-grain', css: '.cm-atmosphere-grain{display:none!important}' },
  { name: 'grain-viewport-only', css: '.cm-atmosphere-grain{inset:0!important;transform:none!important}' },
  { name: 'grain-normal-blend', css: '.cm-atmosphere-grain{inset:0!important;transform:none!important;mix-blend-mode:normal!important;opacity:.055!important}' },
  { name: 'grain-css-pattern', css: '.cm-atmosphere-grain{inset:0!important;transform:none!important;mix-blend-mode:soft-light!important;opacity:.16!important;background-image:radial-gradient(circle at 18% 23%,#fff 0 .45px,transparent .8px),radial-gradient(circle at 72% 61%,#000 0 .5px,transparent .9px),radial-gradient(circle at 41% 82%,#fff 0 .35px,transparent .75px)!important;background-size:7px 11px,13px 17px,19px 23px!important}' },
  { name: 'grain-css-normal', css: '.cm-atmosphere-grain{inset:0!important;transform:none!important;mix-blend-mode:normal!important;opacity:.045!important;background-image:radial-gradient(circle at 18% 23%,#fff 0 .45px,transparent .8px),radial-gradient(circle at 72% 61%,#000 0 .5px,transparent .9px),radial-gradient(circle at 41% 82%,#fff 0 .35px,transparent .75px)!important;background-size:7px 11px,13px 17px,19px 23px!important}' },
  { name: 'without-backdrop', css: '.cm-cinematic-surface .cm-preview,.cm-cinematic-symbol .cm-signal-note,.cm-cinematic-copy .cm-story-tags li{backdrop-filter:none!important}' },
  { name: 'without-hidden-surfaces', css: '.cm-cinematic-surface[style*="opacity: 0"]{visibility:hidden!important}' },
  { name: 'low-overdraw', css: '.cm-atmosphere-grain{display:none!important}.cm-cinematic-surface .cm-preview,.cm-cinematic-symbol .cm-signal-note,.cm-cinematic-copy .cm-story-tags li{backdrop-filter:none!important}.cm-cinematic-surface[style*="opacity: 0"]{visibility:hidden!important}' },
];
const selectedNames = new Set((process.env.LANDING_DIAGNOSTIC_VARIANTS || '').split(',').filter(Boolean));
const selectedVariants = selectedNames.size ? variants.filter(variant => selectedNames.has(variant.name)) : variants;

function percentile(values, amount) {
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * amount))] || 0;
}

const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || 'msedge', headless: true });
const results = [];
try {
  for (const variant of selectedVariants) {
    const context = await browser.newContext({ viewport: { width: 1536, height: 900 }, deviceScaleFactor: 1, reducedMotion: 'no-preference' });
    const page = await context.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.evaluate(() => document.fonts.ready);
    if (variant.css) await page.addStyleTag({ content: variant.css });
    await page.waitForTimeout(400);
    const measurement = await page.evaluate(async scrollDuration => {
      const scene = document.querySelector('.cm-cinematic');
      const frames = [];
      const startY = scene?.offsetTop || 0;
      const distance = scene ? scene.offsetHeight - innerHeight : document.documentElement.scrollHeight - innerHeight;
      await new Promise(resolve => {
        let start;
        const step = timestamp => {
          if (start === undefined) start = timestamp;
          frames.push(timestamp);
          const progress = Math.min(1, (timestamp - start) / scrollDuration);
          scrollTo(0, startY + distance * progress);
          if (progress < 1) requestAnimationFrame(step);
          else requestAnimationFrame(() => requestAnimationFrame(resolve));
        };
        requestAnimationFrame(step);
      });
      return frames.slice(1).map((timestamp, index) => timestamp - frames[index]);
    }, duration);
    const average = measurement.reduce((sum, value) => sum + value, 0) / measurement.length;
    results.push({ name: variant.name, frames: measurement.length, fps: Number((1000 / average).toFixed(1)), averageMs: Number(average.toFixed(1)), p95Ms: Number(percentile(measurement, .95).toFixed(1)) });
    await context.close();
  }
  const report = { environment: 'Headless Microsoft Edge, 1536x900, no CDP tracing', duration, results };
  if (diagnosticOutput) {
    await mkdir(dirname(diagnosticOutput), { recursive: true });
    await writeFile(diagnosticOutput, JSON.stringify(report, null, 2));
  }
  console.log(JSON.stringify({ output: diagnosticOutput, ...report }, null, 2));
} finally {
  await browser.close();
}
