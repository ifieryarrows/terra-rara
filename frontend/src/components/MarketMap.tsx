import './MarketMap.css';

interface MarketSymbol {
    symbol: string;
    name: string;
    category: 'core' | 'etf' | 'titan' | 'regional' | 'junior';
    change?: number; // Percent change
    price?: number;
}

// Symbol definitions with categories
const MARKET_SYMBOLS: MarketSymbol[] = [
    // Core Indicators
    { symbol: 'HG=F', name: 'Copper Futures', category: 'core' },
    { symbol: 'DX-Y.NYB', name: 'US Dollar Index', category: 'core' },
    { symbol: 'CL=F', name: 'Crude Oil', category: 'core' },

    // ETFs
    { symbol: 'COPX', name: 'Global Copper Miners', category: 'etf' },
    { symbol: 'COPJ', name: 'Junior Copper Miners', category: 'etf' },

    // Titans (Majors)
    { symbol: 'BHP', name: 'BHP Group', category: 'titan' },
    { symbol: 'FCX', name: 'Freeport-McMoRan', category: 'titan' },
    { symbol: 'SCCO', name: 'Southern Copper', category: 'titan' },
    { symbol: 'RIO', name: 'Rio Tinto', category: 'titan' },

    // Regional/Strategic
    { symbol: 'TECK', name: 'Teck Resources', category: 'regional' },
    { symbol: 'IVN.TO', name: 'Ivanhoe Mines', category: 'regional' },
    { symbol: '2899.HK', name: 'Zijin Mining', category: 'regional' },

    // Juniors/M&A Targets
    { symbol: 'LUN.TO', name: 'Lundin Mining', category: 'junior' },
    { symbol: 'FIL.TO', name: 'Filo Corp', category: 'junior' },
];

const CATEGORY_LABELS: Record<string, { emoji: string; label: string }> = {
    core: { emoji: '🔵', label: 'Core Indicators' },
    etf: { emoji: '🟢', label: 'ETFs' },
    titan: { emoji: '🟡', label: 'Titans' },
    regional: { emoji: '🟠', label: 'Regional' },
    junior: { emoji: '🔴', label: 'Juniors' },
};

function getChangeClass(change?: number): string {
    if (change === undefined || change === null) return '';
    if (change > 2) return 'strong-up';
    if (change > 0) return 'up';
    if (change < -2) return 'strong-down';
    if (change < 0) return 'down';
    return 'neutral';
}

function formatChange(change?: number): string {
    if (change === undefined || change === null) return '--';
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)}%`;
}

export function MarketMap() {
    const symbols = MARKET_SYMBOLS;

    // Categories for grouping
    const categories = ['core', 'etf', 'titan', 'regional', 'junior'];

    return (
        <div className="market-map">
            <h2 className="market-map-title">🗺️ Market Intelligence Map</h2>
            <p className="market-map-subtitle">Copper ecosystem at a glance • Updated with pipeline</p>

            <div className="market-map-grid">
                {categories.map(category => {
                    const categorySymbols = symbols.filter(s => s.category === category);
                    const categoryInfo = CATEGORY_LABELS[category];

                    return (
                        <div key={category} className="market-category">
                            <div className="category-header">
                                <span className="category-emoji">{categoryInfo.emoji}</span>
                                <span className="category-label">{categoryInfo.label}</span>
                            </div>

                            <div className="category-cards">
                                {categorySymbols.map(sym => (
                                    <a
                                        key={sym.symbol}
                                        href={`https://finance.yahoo.com/quote/${sym.symbol}`}
                                        target="_blank"
                                        rel="noopener noreferrer"
                                        className={`market-card ${getChangeClass(sym.change)}`}
                                    >
                                        <div className="card-symbol">{sym.symbol}</div>
                                        <div className="card-name">{sym.name}</div>
                                        <div className={`card-change ${getChangeClass(sym.change)}`}>
                                            {formatChange(sym.change)}
                                        </div>
                                    </a>
                                ))}
                            </div>
                        </div>
                    );
                })}
            </div>

            <div className="market-map-legend">
                <span className="legend-item strong-up">Strong ↑</span>
                <span className="legend-item up">Up ↑</span>
                <span className="legend-item neutral">Flat</span>
                <span className="legend-item down">Down ↓</span>
                <span className="legend-item strong-down">Strong ↓</span>
            </div>
        </div>
    );
}

export default MarketMap;
