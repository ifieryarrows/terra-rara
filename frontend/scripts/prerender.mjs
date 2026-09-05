import { readFile, writeFile } from 'node:fs/promises';
import { createServer } from 'vite';

process.env.NODE_ENV = 'production';

// Build-time HTML only. Keep Vite, React Router and the existing Vercel deployment.
const server = await createServer({ server: { middlewareMode: true }, appType: 'custom', optimizeDeps: { noDiscovery: true, include: [] } });
try {
  const template = await readFile('dist/index.html', 'utf8');
  const { renderLanding } = await server.ssrLoadModule('/src/prerender.tsx');
  const landing = renderLanding();
  if (!landing.includes('Read the market.')) throw new Error('Landing prerender has no critical content');
  await writeFile('dist/index.html', template.replace('<div id="root"></div>', '<div id="root">' + landing + '</div>'));
  const workspace = template.replace('<title>Terra Rara | Copper Market Intelligence</title>', '<title>Research Workspace | CopperMind</title>');
  await writeFile('dist/workspace.html', workspace);
  console.log('Prerendered landing HTML; separate workspace shell preserves deep links.');
} finally {
  await server.close();
}
