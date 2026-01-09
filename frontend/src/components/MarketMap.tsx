import { useEffect, useState, useRef } from 'react';
import { fetchMarketPrices, type MarketPricesResponse } from '../api';
import './MarketMap.css';

interface MarketSymbol {
    symbol: string;
    name: string;
    category: 'core' | 'etf' | 'titan' | 'regional' | 'junior';
    change?: number;
    price?: number;
    flash?: 'up' | 'down' | null;  // For flash animation
}

// Symbol definitions with categories
const MARKET_SYMBOLS: MarketSymbol[] = [
    // Core Indicators
    { symbol: 'HG=F', name: 'Copper Futures', category: 'core' },
    { symbol: 'DX-Y.NYB', name: 'US Dollar Index', category: 'core' },
    { symbol: 'CL=F', name: 'Crude Oil', category: 'core' },

    // ETFs
    { symbol: 'FXI', name: 'China Large-Cap', category: 'etf' },
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

    // Juniors
    { symbol: 'LUN.TO', name: 'Lundin Mining', category: 'junior' },
];

const CATEGORY_LABELS: Record<string, { emoji: string; label: string }> = {
    core: { emoji: '🔵', label: 'Core Indicators' },
    etf: { emoji: '🟢', label: 'ETFs' },
    titan: { emoji: '🟡', label: 'Titans' },
    regional: { emoji: '🟠', label: 'Regional' },
    junior: { emoji: '🔴', label: 'Juniors' },
};

// Refresh interval in milliseconds (30 seconds)
const REFRESH_INTERVAL = 30000;

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
    const [symbols, setSymbols] = useState<MarketSymbol[]>(MARKET_SYMBOLS);
    const [loading, setLoading] = useState(true);
    const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
    const prevPricesRef = useRef<Record<string, number>>({});

    const loadPrices = async (isInitial = false) => {
        try {
            if (isInitial) setLoading(true);

            const data: MarketPricesResponse = await fetchMarketPrices();

            // Merge prices into symbols and detect changes
            const updated = MARKET_SYMBOLS.map(sym => {
                const newPrice = data.symbols[sym.symbol]?.price ?? null;
                const prevPrice = prevPricesRef.current[sym.symbol];

                // Detect flash direction
                let flash: 'up' | 'down' | null = null;
                if (prevPrice !== undefined && newPrice !== null && prevPrice !== newPrice) {
                    flash = newPrice > prevPrice ? 'up' : 'down';
                }

                // Update prev prices ref
                if (newPrice !== null) {
                    prevPricesRef.current[sym.symbol] = newPrice;
                }

                return {
                    ...sym,
                    change: data.symbols[sym.symbol]?.change ?? undefined,
                    price: newPrice ?? undefined,
                    flash,
                };
            });

            setSymbols(updated);
            setLastUpdate(new Date());

            // Clear flash after animation
            setTimeout(() => {
                setSymbols(prev => prev.map(s => ({ ...s, flash: null })));
            }, 1000);

        } catch (error) {
            console.error('Failed to load market prices:', error);
        } finally {
            if (isInitial) setLoading(false);
        }
    };

    useEffect(() => {
        // Initial load
        loadPrices(true);

        // Set up polling interval
        const interval = setInterval(() => loadPrices(false), REFRESH_INTERVAL);

        return () => clearInterval(interval);
    }, []);

    const categories = ['core', 'etf', 'titan', 'regional', 'junior'];

    return (
        <div className="market-map">
            <h2 className="market-map-title">🗺️ Market Intelligence Map</h2>
            <p className="market-map-subtitle">
                Copper ecosystem • Live data (15-min delayed)
                {lastUpdate && (
                    <span className="last-update">
                        {' '}• Updated: {lastUpdate.toLocaleTimeString()}
                    </span>
                )}
                {loading && <span className="loading-indicator"> • Loading...</span>}
            </p>

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
                                        className={`market-card ${getChangeClass(sym.change)} ${sym.flash ? `flash-${sym.flash}` : ''}`}
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
