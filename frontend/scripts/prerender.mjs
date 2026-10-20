import { readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'vite';

process.env.NODE_ENV = 'production';

// The landing upgrades to the cinematic story after the main bundle boots. If
// that chunk only starts downloading when React first suspends, the static
// hero is replaced seconds later and the page appears to jump. Start the chunk
// and its atmosphere dependency with the document instead. The workspace shell
// deliberately skips these tags: it never renders the cinematic on first paint.
async function cinematicPreload() {
  try {
    const manifest = JSON.parse(await readFile('dist/.vite/manifest.json', 'utf8'));
    const entry = manifest['src/features/landing/CinematicLanding.tsx'];
    if (!entry) return '';
    const files = new Set([entry.file]);
    const styles = new Set(entry.css ?? []);
    for (const imported of entry.imports ?? []) {
      const dependency = imported === 'index.html' ? null : manifest[imported];
      if (!dependency) continue;
      files.add(dependency.file);
      for (const file of dependency.css ?? []) styles.add(file);
    }
    return [
      ...[...files].map(file => `<link rel="modulepreload" crossorigin href="/${file}">`),
      ...[...styles].map(file => `<link rel="preload" as="style" href="/${file}">`),
    ].join('');
  } catch {
    return '';
  }
}

// Build-time HTML only. Keep Vite, React Router and the existing Vercel deployment.
// This server has a different optimizer configuration from `npm run dev`.
// Sharing its cache can invalidate the running dev server's dependency URLs.
const server = await createServer({ cacheDir: 'node_modules/.vite-prerender', server: { middlewareMode: true }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
try {
  const template = await readFile('dist/index.html', 'utf8');
  const preload = await cinematicPreload();
  const { renderLanding } = await server.ssrLoadModule('/src/prerender.tsx');
  const landing = renderLanding();
  if (!landing.includes('Read the market.')) throw new Error('Landing prerender has no critical content');
  const landingHtml = template
    .replace('<div id="root"></div>', '<div id="root">' + landing + '</div>')
    .replace('</head>', `${preload}</head>`);
  await writeFile('dist/index.html', landingHtml);
  const workspace = template.replace('<title>Terra Rara | Copper Market Intelligence</title>', '<title>Research Workspace | CopperMind</title>');
  await writeFile('dist/workspace.html', workspace);
  console.log(`Prerendered landing HTML (${preload ? 'cinematic preload on' : 'no cinematic preload'}); separate workspace shell preserves deep links.`);
} finally {
  await server.close();
}
