// Repeatable Chrome DevTools Protocol profile for the native-scroll landing scene.
// This is a lab measurement, not field telemetry. Run the same build, browser,
// viewport and throttling settings for before/after comparisons.
import { createRequire } from 'node:module';
import { mkdir, writeFile } from 'node:fs/promises';
import { dirname, resolve } from 'node:path';
import { interceptDashboard } from './dashboard-fixtures.mjs';

const { chromium } = createRequire(import.meta.url)(process.env.PLAYWRIGHT_MODULE || 'playwright');
const base = process.env.EXPERIENCE_URL || 'http://127.0.0.1:4173';
const output = resolve(process.env.LANDING_PROFILE_OUTPUT || 'landing-profile.json');
const profileLabel = process.env.LANDING_PROFILE_LABEL || 'unlabelled';
const scrollDuration = Number(process.env.LANDING_SCROLL_DURATION || 4200);
const runCount = Number(process.env.LANDING_PROFILE_RUNS || 3);

const scenarios = [
  { name: 'desktop', viewport: { width: 1536, height: 900 }, cpuRate: 1, deviceScaleFactor: 1, isMobile: false, hasTouch: false },
  { name: 'desktop-4x-cpu', viewport: { width: 1280, height: 720 }, cpuRate: 4, deviceScaleFactor: 1, isMobile: false, hasTouch: false },
  { name: 'mobile-4x-cpu', viewport: { width: 390, height: 844 }, cpuRate: 4, deviceScaleFactor: 2, isMobile: true, hasTouch: true },
];

function percentile(values, amount) {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * amount))];
}

function round(value, digits = 3) {
  return Number(Number(value || 0).toFixed(digits));
}

async function readTrace(client, stream) {
  let text = '';
  while (true) {
    const chunk = await client.send('IO.read', { handle: stream });
    text += chunk.data;
    if (chunk.eof) break;
  }
  await client.send('IO.close', { handle: stream });
  return JSON.parse(text).traceEvents;
}

function summarizeTrace(events) {
  const thread = events.find(event => event.ph === 'M' && event.name === 'thread_name' && event.args?.name === 'CrRendererMain');
  const mainEvents = thread ? events.filter(event => event.ph === 'X' && event.pid === thread.pid && event.tid === thread.tid) : [];
  const allComplete = events.filter(event => event.ph === 'X');
  const duration = event => (event.dur || 0) / 1000;
  const summarize = (source, names) => {
    const matches = source.filter(event => names.includes(event.name));
    return { count: matches.length, totalMs: round(matches.reduce((sum, event) => sum + duration(event), 0)), maxMs: round(Math.max(0, ...matches.map(duration))) };
  };
  const tasks = mainEvents.filter(event => event.name === 'RunTask' || event.name === 'Task');
  const longTasks = tasks.filter(event => duration(event) >= 50).map(event => round(duration(event)));
  return {
    rendererMainThreadFound: Boolean(thread),
    mainThreadTasks: { count: tasks.length, totalMs: round(tasks.reduce((sum, event) => sum + duration(event), 0)), maxMs: round(Math.max(0, ...tasks.map(duration))) },
    longTasks: { count: longTasks.length, durationsMs: longTasks },
    style: summarize(mainEvents, ['UpdateLayoutTree', 'RecalculateStyles']),
    layout: summarize(mainEvents, ['Layout']),
    prePaint: summarize(mainEvents, ['PrePaint']),
    paint: summarize(mainEvents, ['Paint', 'PaintImage']),
    composite: summarize(allComplete, ['CompositeLayers', 'Commit']),
    raster: summarize(allComplete, ['RasterTask']),
    animationFrames: summarize(mainEvents, ['AnimationFrame', 'FireAnimationFrame']),
    script: summarize(mainEvents, ['FunctionCall', 'EvaluateScript', 'EventDispatch']),
  };
}

function metricsByName(metrics) {
  return Object.fromEntries(metrics.map(metric => [metric.name, metric.value]));
}

async function safeSystemInfo(browserSession) {
  try {
    const [system, processes] = await Promise.all([
      browserSession.send('SystemInfo.getInfo'),
      browserSession.send('SystemInfo.getProcessInfo'),
    ]);
    return {
      gpu: system.gpu?.devices?.map(device => ({ vendor: device.vendorString, device: device.deviceString, driver: device.driverVersion })) || [],
      gpuFeatureStatus: system.gpu?.featureStatus || {},
      gpuProcesses: processes.processInfo.filter(process => process.type === 'GPU').map(process => ({ id: process.id, cpuTime: process.cpuTime, privateMemory: process.privateMemory || null })),
    };
  } catch (error) {
    return { unavailable: String(error) };
  }
}

async function runScenario(browser, browserSession, scenario, run) {
  const context = await browser.newContext({
    viewport: scenario.viewport,
    deviceScaleFactor: scenario.deviceScaleFactor,
    isMobile: scenario.isMobile,
    hasTouch: scenario.hasTouch,
    reducedMotion: 'no-preference',
  });
  await context.addInitScript(() => {
    const profile = window.__cmProfile = {
      canvas: { clear: 0, fill: 0, stroke: 0, drawImage: 0, arc: 0, contexts2d: 0, contextsWebgl: 0 },
      webgl: { drawArrays: 0, drawElements: 0, bufferData: 0, bufferSubData: 0, compileShader: 0, linkProgram: 0, createTexture: 0, texImage2D: 0, createBuffer: 0, deleteBuffer: 0, createProgram: 0, deleteProgram: 0 },
      longTasks: [], layoutShifts: 0, lcp: 0, reactCommits: 0,
    };
    const originalGetContext = HTMLCanvasElement.prototype.getContext;
    HTMLCanvasElement.prototype.getContext = function(type, ...args) {
      if (type === '2d') profile.canvas.contexts2d += 1;
      if (type === 'webgl' || type === 'webgl2') profile.canvas.contextsWebgl += 1;
      return originalGetContext.call(this, type, ...args);
    };
    for (const [method, counter] of [['clearRect', 'clear'], ['fill', 'fill'], ['stroke', 'stroke'], ['drawImage', 'drawImage'], ['arc', 'arc']]) {
      const original = CanvasRenderingContext2D.prototype[method];
      CanvasRenderingContext2D.prototype[method] = function(...args) {
        profile.canvas[counter] += 1;
        return original.apply(this, args);
      };
    }
    const webgl2 = window.WebGL2RenderingContext?.prototype;
    if (webgl2) {
      for (const method of Object.keys(profile.webgl)) {
        const original = webgl2[method];
        if (typeof original !== 'function') continue;
        webgl2[method] = function(...args) {
          profile.webgl[method] += 1;
          return original.apply(this, args);
        };
      }
    }
    try {
      new PerformanceObserver(list => list.getEntries().forEach(entry => profile.longTasks.push({ start: entry.startTime, duration: entry.duration }))).observe({ type: 'longtask', buffered: true });
      new PerformanceObserver(list => list.getEntries().forEach(entry => { if (!entry.hadRecentInput) profile.layoutShifts += entry.value; })).observe({ type: 'layout-shift', buffered: true });
      new PerformanceObserver(list => list.getEntries().forEach(entry => { profile.lcp = entry.startTime; })).observe({ type: 'largest-contentful-paint', buffered: true });
    } catch { /* Older engines may omit one of the observer entry types. */ }
    let rendererId = 0;
    Object.defineProperty(window, '__REACT_DEVTOOLS_GLOBAL_HOOK__', { configurable: true, value: {
      supportsFiber: true, renderers: new Map(),
      inject(renderer) { rendererId += 1; this.renderers.set(rendererId, renderer); return rendererId; },
      onCommitFiberRoot() { profile.reactCommits += 1; }, onCommitFiberUnmount() {},
    }});
  });
  const page = await context.newPage();
  await interceptDashboard(page);
  const client = await context.newCDPSession(page);
  await client.send('Network.enable');
  await client.send('Network.setCacheDisabled', { cacheDisabled: true });
  await client.send('Emulation.setCPUThrottlingRate', { rate: scenario.cpuRate });
  await client.send('Performance.enable');
  await client.send('LayerTree.enable');
  let layers = [];
  client.on('LayerTree.layerTreeDidChange', event => { layers = event.layers || []; });

  await page.goto(base, { waitUntil: 'networkidle' });
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(500);
  const initial = await page.evaluate(() => {
    const navigation = performance.getEntriesByType('navigation')[0];
    const resources = performance.getEntriesByType('resource').map(entry => ({
      name: new URL(entry.name).pathname,
      type: entry.initiatorType,
      startMs: entry.startTime,
      durationMs: entry.duration,
      transferBytes: entry.transferSize,
      encodedBytes: entry.encodedBodySize,
      decodedBytes: entry.decodedBodySize,
    }));
    return {
      navigation: navigation ? { responseEndMs: navigation.responseEnd, domContentLoadedMs: navigation.domContentLoadedEventEnd, loadMs: navigation.loadEventEnd, transferBytes: navigation.transferSize } : null,
      resources,
      domNodes: document.getElementsByTagName('*').length,
      canvases: [...document.querySelectorAll('canvas')].map(canvas => ({ className: canvas.className, width: canvas.width, height: canvas.height, renderer: canvas.dataset.renderer || null, particleCount: Number(canvas.dataset.particleCount || 0), quality: canvas.dataset.quality || null, shaderCompileMs: Number(canvas.dataset.shaderCompileMs || 0) || null })),
      enhanced: Boolean(document.querySelector('.cm-cinematic-sticky')),
      visibleForbiddenLabels: document.body.innerText.match(/Cu\s*\/\s*29|naive scroll continous signal|native scroll\s*\/\s*continuous signal/ig) || [],
      profile: structuredClone(window.__cmProfile),
    };
  });
  const beforeMetrics = metricsByName((await client.send('Performance.getMetrics')).metrics);
  const traceComplete = new Promise(resolve => client.once('Tracing.tracingComplete', resolve));
  await client.send('Tracing.start', {
    transferMode: 'ReturnAsStream',
    categories: 'devtools.timeline,disabled-by-default-devtools.timeline,blink.user_timing,v8,renderer.scheduler,cc,gpu',
  });
  const scroll = await page.evaluate(async duration => {
    const profile = window.__cmProfile;
    const scene = document.querySelector('.cm-cinematic');
    const frames = [];
    const startY = scene ? scene.offsetTop : 0;
    const distance = scene ? Math.max(0, scene.offsetHeight - innerHeight) : Math.max(0, document.documentElement.scrollHeight - innerHeight);
    const canvasBefore = { ...profile.canvas };
    const webglBefore = { ...profile.webgl };
    const commitsBefore = profile.reactCommits;
    const longTasksBefore = profile.longTasks.length;
    await new Promise(resolve => {
      let start;
      function step(timestamp) {
        if (start === undefined) start = timestamp;
        frames.push(timestamp);
        const progress = Math.min(1, (timestamp - start) / duration);
        scrollTo(0, startY + distance * progress);
        if (progress < 1) requestAnimationFrame(step);
        else requestAnimationFrame(() => requestAnimationFrame(resolve));
      }
      requestAnimationFrame(step);
    });
    const intervals = frames.slice(1).map((timestamp, index) => timestamp - frames[index]);
    return {
      elapsedMs: frames.at(-1) - frames[0],
      frameIntervalsMs: intervals,
      canvasDelta: Object.fromEntries(Object.entries(profile.canvas).map(([key, value]) => [key, value - canvasBefore[key]])),
      webglDelta: Object.fromEntries(Object.entries(profile.webgl).map(([key, value]) => [key, value - webglBefore[key]])),
      reactCommitDelta: profile.reactCommits - commitsBefore,
      observedLongTasks: profile.longTasks.slice(longTasksBefore),
      endScrollY: scrollY,
    };
  }, scrollDuration);
  await client.send('Tracing.end');
  const traceEvent = await traceComplete;
  const traceEvents = await readTrace(client, traceEvent.stream);
  const afterMetrics = metricsByName((await client.send('Performance.getMetrics')).metrics);
  const frameIntervals = scroll.frameIntervalsMs;
  const routeCleanup = await page.evaluate(async () => {
    scrollTo(0, 0);
    document.querySelector('a[href="/dashboard"]')?.click();
    await new Promise(resolve => setTimeout(resolve, 700));
    const before = window.__cmProfile.canvas.clear;
    const beforeWebgl = window.__cmProfile.webgl.drawArrays + window.__cmProfile.webgl.drawElements;
    window.dispatchEvent(new PointerEvent('pointermove', { clientX: 220, clientY: 180 }));
    await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
    return {
      path: location.pathname,
      landingCanvasCount: document.querySelectorAll('.cm-particle-world').length,
      canvasFramesAfterPointer: window.__cmProfile.canvas.clear - before,
      webglDrawsAfterPointer: window.__cmProfile.webgl.drawArrays + window.__cmProfile.webgl.drawElements - beforeWebgl,
      domNodes: document.getElementsByTagName('*').length,
    };
  });
  const network = initial.resources.reduce((summary, resource) => {
    summary.transferBytes += resource.transferBytes || 0;
    summary.decodedBytes += resource.decodedBytes || 0;
    if (resource.name.endsWith('.js')) summary.jsTransferBytes += resource.transferBytes || 0;
    if (resource.name.endsWith('.css')) summary.cssTransferBytes += resource.transferBytes || 0;
    summary.maxResourceDurationMs = Math.max(summary.maxResourceDurationMs, resource.durationMs || 0);
    return summary;
  }, { transferBytes: 0, decodedBytes: 0, jsTransferBytes: 0, cssTransferBytes: 0, maxResourceDurationMs: 0 });
  const result = {
    scenario: { ...scenario, run },
    initial,
    network: { ...network, resources: initial.resources },
    frames: {
      count: frameIntervals.length,
      averageMs: round(frameIntervals.reduce((sum, value) => sum + value, 0) / Math.max(1, frameIntervals.length)),
      p95Ms: round(percentile(frameIntervals, .95)),
      p99Ms: round(percentile(frameIntervals, .99)),
      estimatedFps: round(1000 / Math.max(.001, frameIntervals.reduce((sum, value) => sum + value, 0) / Math.max(1, frameIntervals.length)), 1),
      overBudget16_7: frameIntervals.filter(value => value > 16.7).length,
      overBudget50: frameIntervals.filter(value => value > 50).length,
    },
    scroll: { ...scroll, frameIntervalsMs: undefined },
    trace: summarizeTrace(traceEvents),
    performanceMetricsDelta: {
      taskMs: round((afterMetrics.TaskDuration - beforeMetrics.TaskDuration) * 1000),
      scriptMs: round((afterMetrics.ScriptDuration - beforeMetrics.ScriptDuration) * 1000),
      layoutMs: round((afterMetrics.LayoutDuration - beforeMetrics.LayoutDuration) * 1000),
      recalcStyleMs: round((afterMetrics.RecalcStyleDuration - beforeMetrics.RecalcStyleDuration) * 1000),
      layoutCount: round(afterMetrics.LayoutCount - beforeMetrics.LayoutCount),
      recalcStyleCount: round(afterMetrics.RecalcStyleCount - beforeMetrics.RecalcStyleCount),
      jsHeapDeltaBytes: round(afterMetrics.JSHeapUsedSize - beforeMetrics.JSHeapUsedSize),
      nodesDelta: round(afterMetrics.Nodes - beforeMetrics.Nodes),
    },
    layers: { count: layers.length, drawsContent: layers.filter(layer => layer.drawsContent).length, paintCount: layers.reduce((sum, layer) => sum + (layer.paintCount || 0), 0), memoryEstimateBytes: layers.reduce((sum, layer) => sum + (layer.memory || 0), 0) || null },
    routeCleanup,
    gpu: await safeSystemInfo(browserSession),
  };
  await context.close();
  return result;
}

await mkdir(dirname(output), { recursive: true });
const browser = await chromium.launch({ channel: process.env.BROWSER_CHANNEL || 'msedge', headless: true });
const browserSession = await browser.newBrowserCDPSession();
const results = [];
try {
  for (const scenario of scenarios) {
    for (let run = 0; run < runCount; run += 1) results.push(await runScenario(browser, browserSession, scenario, run));
  }
  const report = {
    label: profileLabel,
    measuredAt: new Date().toISOString(),
    environment: 'Headless Microsoft Edge; cold local production preview; native scroll; CDP Performance and Tracing; exact GPU throttling unavailable',
    scrollDurationMs: scrollDuration,
    runCount,
    results,
  };
  await writeFile(output, JSON.stringify(report, null, 2));
  console.log(JSON.stringify({ output, results: results.map(result => ({ name: result.scenario.name, enhanced: result.initial.enhanced, domNodes: result.initial.domNodes, canvases: result.initial.canvases, fps: result.frames.estimatedFps, p95FrameMs: result.frames.p95Ms, longTasks: result.trace.longTasks.count, mainThreadTaskMs: result.trace.mainThreadTasks.totalMs, paintMs: result.trace.paint.totalMs, canvas: result.scroll.canvasDelta, webgl: result.scroll.webglDelta, reactCommits: result.scroll.reactCommitDelta, routeCleanup: result.routeCleanup })) }, null, 2));
} finally {
  await browser.close();
}
