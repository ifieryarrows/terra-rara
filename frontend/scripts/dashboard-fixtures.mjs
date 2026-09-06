// Synthetic QA data only. Never imported by the application.
const date = index => new Date(Date.UTC(2026, 3, 1 + index)).toISOString().slice(0, 10);
export const history = { symbol: 'HG=F', data: Array.from({ length: 180 }, (_, i) => ({ date: date(i), price: 4.6 + i * .001 + Math.sin(i / 8) * .08, sentiment_index: null, sentiment_news_count: null })) };
const last = history.data.at(-1);
export const tft = {
  direction: 'BULLISH', risk_level: 'MEDIUM', primary_forecast_return: .025,
  primary_forecast_q10: -.015, primary_forecast_q90: .06, t1_impulse: 'BULLISH', t1_return: .005,
  generated_at: last.date + 'T20:00:00Z',
  model_metadata: { metrics: { weekly_directional_accuracy: .54, weekly_sample_count: 100, sharpe_ratio: .2 } },
  prediction: { reference_price: last.price, reference_price_date: last.date, weekly_price: last.price * 1.025, weekly_return: .025,
    daily_forecasts: Array.from({ length: 5 }, (_, i) => ({ day: i + 1, forecast_date: date(180 + i), price_median: last.price + .025 * (i + 1), price_q10: last.price - .02 * (i + 1), price_q90: last.price + .05 * (i + 1) })) },
};
export const articles = Array.from({ length: 8 }, (_, i) => ({ id: i + 1, raw_id: i + 1, title: `QA fixture: copper market update ${i + 1}`, description: 'Synthetic article for layout, search and drawer verification.', url: 'https://example.com/qa', channel: i % 2 ? 'newsapi' : 'google_news', publisher: i % 2 ? 'QA Publisher B' : 'QA Publisher A', published_at: last.date + 'T12:00:00Z', language: 'en', sentiment: { label: 'NEUTRAL', relevance: .8, confidence: .7, final_score: 0, reasoning: 'Synthetic explanation used to test the reading panel.', scoring_mode: 'llm' } }));
export const heatmap = { id: 'root', name: 'QA map', _meta: { payload_count: 24, is_stale: false, refresh_in_progress: false, source_delay_minutes: 15, last_updated_at: last.date, next_refresh_at: null }, children: ['Materials', 'Technology', 'Energy'].map((name, group) => ({ id: `sector-${group}`, name, children: [{ id: `industry-${group}`, name: `${name} industry`, children: Array.from({ length: 8 }, (_, i) => ({ id: `qa-${group}-${i}`, name: `QA${group}${i}`, shortName: `QA company ${group}-${i}`, weight: 100 - i * 8, price: 40 + i, changePercent: i - 3, sector: name, industry: `${name} industry`, sparkline: [40, 41, 40.5, 42], asOf: last.date })) }] })) };

export function fixtureFor(url) {
  const path = decodeURIComponent(url.pathname);
  if (path === '/api/history') return history;
  if (path.startsWith('/api/analysis/tft/')) return tft;
  if (path === '/api/analysis') return { symbol: 'HG=F', generated_at: last.date, top_influencers: [{ feature: 'qa_demand', label: 'Copper demand expectations across industrial markets', category: 'Macro', importance: .4 }], current_price: last.price };
  if (path === '/api/live-price') return { price: last.price, timestamp: last.date };
  if (path === '/api/commentary') return { commentary: 'Synthetic QA commentary. These values do not represent a live market view.', generation_mode: 'llm', generated_at: last.date };
  if (path === '/api/sentiment/summary') return { index: 0, label: 'Neutral', article_count: 8 };
  if (path === '/api/market-heatmap') return heatmap;
  if (path === '/api/market-heatmap/context') return { category: 'QA', news: [], peers: [] };
  if (/\/api\/news\/\d+$/.test(path)) return articles.find(a => a.id === Number(path.split('/').at(-1)));
  const selected = articles.filter(a => (!url.searchParams.get('search') || a.title.includes(url.searchParams.get('search'))) && (!url.searchParams.get('publisher') || a.publisher === url.searchParams.get('publisher')) && (!url.searchParams.get('channel') || url.searchParams.get('channel') === 'all' || a.channel === url.searchParams.get('channel')));
  if (path === '/api/news') return { items: selected, total: selected.length, offset: 0, limit: 20, has_more: false, as_of: last.date, generated_at: last.date };
  if (path === '/api/news/stats') return { total_articles: selected.length, label_distribution: { NEUTRAL: selected.length }, channel_distribution: { newsapi: 4, google_news: 4 }, top_publishers: [{ publisher: 'QA Publisher A', count: 4 }, { publisher: 'QA Publisher B', count: 4 }] };
  return undefined;
}

export async function interceptDashboard(page, override = () => undefined) {
  await page.route('**/api/**', async route => {
    const url = new URL(route.request().url());
    const custom = override(url);
    const payload = custom?.payload ?? fixtureFor(url);
    await route.fulfill({ status: custom?.status ?? (payload ? 200 : 404), contentType: 'application/json', body: JSON.stringify(payload ?? { detail: 'No QA fixture' }) });
  });
}
