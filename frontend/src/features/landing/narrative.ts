export const scenes = [
    { id: 'copper', word: 'COPPER', label: 'The metal', title: 'One metal. A world of signals.', text: 'Copper market intelligence. Connect market context, news and quantitative forecasts in one research workspace.', to: '/dashboard', action: 'Enter CopperMind' },
    { id: 'research', word: 'SIGNAL', label: 'Follow the copper signal', title: 'Find the thread.', text: 'A price is a starting point. Follow the connections.', to: '#market', action: 'Follow the signal' },
    { id: 'market', word: 'MARKET', label: '01 / Market context', title: 'Look beyond a single price.', text: 'Explore copper alongside producers, related instruments and the wider market. Start with context before drawing a conclusion.', to: '/dashboard#market-map', action: 'Explore market context' },
    { id: 'news', word: 'CONTEXT', label: '02 / News intelligence', title: 'A source. Then a perspective.', text: 'Read the news, examine sentiment and inspect the commentary. An explanation is a question to investigate, not proof of causation.', to: '/dashboard#news-intelligence', action: 'Read the intelligence' },
    { id: 'forecast', word: 'RANGE', label: '03 / The possibilities', title: 'Keep uncertainty in the picture.', text: 'Study the five-session outlook alongside history and available uncertainty intervals. A forecast is a range of possibilities.', to: '/dashboard#price-forecast', action: 'Examine the forecasts' },
    { id: 'evidence', word: 'EVIDENCE', label: '04 / Under examination', title: 'What stands up to scrutiny?', text: 'Inspect the published evaluation, its date and its limits. Model metadata and system health provide context, not proof of performance.', to: '/validation', action: 'Examine the evidence' },
    { id: 'workspace', word: 'YOUR MOVE', label: 'Your research starts here', title: 'Bring your next question.', text: 'The market keeps moving. Your workspace keeps the context close.', to: '/dashboard', action: 'Enter CopperMind' },
] as const;
export const clamp = (n: number) => Math.max(0, Math.min(1, n));
export const sceneIndex = (p: number) => Math.min(6, Math.max(0, Math.round(p * 6)));
export function sceneOpacity(p: number, index: number) { return clamp(1 - Math.max(0, Math.abs(p * 6 - index) - .3) / .2); }
// Identical point counts allow the same trace to change shape without replacing its owner.
export function tracePoints(scene: number) {
    return Array.from({ length: 65 }, (_, i) => {
        const t = i / 64, a = t * Math.PI * 2;
        if (scene === 0)
            return [320 + Math.cos(a) * (160 + 24 * Math.sin(3 * a)), 250 + Math.sin(a) * 130];
        if (scene === 1)
            return [40 + t * 560, 250 + Math.sin(t * Math.PI * 2) * 70];
        if (scene === 2)
            return [60 + t * 520, 250 + Math.sin(t * Math.PI * 4) * 22];
        if (scene === 3)
            return [60 + t * 520, 280 + Math.sin(t * Math.PI * 6) * 32 * Math.sin(t * Math.PI)];
        if (scene === 4)
            return [60 + t * 520, 330 - t * 170 + Math.sin(t * 29) * 12];
        if (scene === 5)
            return [60 + t * 520, 330];
        return [60 + t * 520, 250];
    });
}
const paths = scenes.map((_, i) => tracePoints(i));
export function tracePath(p: number) {
    const x = clamp(p) * 6, a = Math.floor(x), b = Math.min(6, a + 1), t = x - a;
    const eased = t * t * (3 - 2 * t);
    return paths[a].map(([px, py], i) => `${i ? 'L' : 'M'}${(px + (paths[b][i][0] - px) * eased).toFixed(2)},${(py + (paths[b][i][1] - py) * eased).toFixed(2)}`).join(' ');
}
