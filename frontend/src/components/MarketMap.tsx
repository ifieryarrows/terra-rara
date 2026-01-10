import { useEffect, useState, useRef } from 'react';

import { fetchMarketPrices, type MarketPricesResponse } from '../api';
import './MarketMap.css';

interface MarketSymbol {
    symbol: string;
    name: string;
    category: 'core' | 'etf' | 'titan' | 'regional' | 'junior';
    change?: number;
    price?: number;
    flash?: 'up' | 'down' | null;
}

const MARKET_SYMBOLS: MarketSymbol[] = [
    { symbol: 'HG=F', name: 'COPPER FUTURES', category: 'core' },
    { symbol: 'DX-Y.NYB', name: 'USD INDEX', category: 'core' },
    { symbol: 'CL=F', name: 'CRUDE OIL', category: 'core' },
    { symbol: 'FXI', name: 'CHINA LARGE-CAP', category: 'etf' },
    { symbol: 'COPX', name: 'GLOBAL MINERS', category: 'etf' },
    { symbol: 'COPJ', name: 'JUNIOR MINERS', category: 'etf' },
    { symbol: 'BHP', name: 'BHP GROUP', category: 'titan' },
    { symbol: 'FCX', name: 'FREEPORT-MCMO', category: 'titan' },
    { symbol: 'SCCO', name: 'SOUTHERN COPPER', category: 'titan' },
    { symbol: 'RIO', name: 'RIO TINTO', category: 'titan' },
    { symbol: 'TECK', name: 'TECK RESOURCES', category: 'regional' },
    { symbol: 'IVN.TO', name: 'IVANHOE MINES', category: 'regional' },
    { symbol: '2899.HK', name: 'ZIJIN MINING', category: 'regional' },
    { symbol: 'LUN.TO', name: 'LUNDIN MINING', category: 'junior' },
];

const CATEGORY_LABELS: Record<string, string> = {
    core: 'CORE INDICATORS',
    etf: 'SECTOR ETFs',
    titan: 'INDUSTRY TITANS',
    regional: 'REGIONAL PLAYERS',
    junior: 'JUNIOR MINERS',
};

const REFRESH_INTERVAL = 30000;

function formatChange(change?: number): string {
    if (change === undefined || change === null) return '--';
    const sign = change >= 0 ? '+' : '';
    return `${sign}${change.toFixed(2)}%`;
}

export function MarketMap() {
    const [symbols, setSymbols] = useState<MarketSymbol[]>(MARKET_SYMBOLS);
    const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
    const prevPricesRef = useRef<Record<string, number>>({});

    const loadPrices = async () => {
        try {
            const data: MarketPricesResponse = await fetchMarketPrices();
            const updated = MARKET_SYMBOLS.map(sym => {
                const newPrice = data.symbols[sym.symbol]?.price ?? null;
                const prevPrice = prevPricesRef.current[sym.symbol];

                let flash: 'up' | 'down' | null = null;
                if (prevPrice !== undefined && newPrice !== null && prevPrice !== newPrice) {
                    flash = newPrice > prevPrice ? 'up' : 'down';
                }

                if (newPrice !== null) prevPricesRef.current[sym.symbol] = newPrice;

                return {
                    ...sym,
                    change: data.symbols[sym.symbol]?.change ?? undefined,
                    price: newPrice ?? undefined,
                    flash,
                };
            });

            setSymbols(updated);
            setLastUpdate(new Date());

            setTimeout(() => {
                setSymbols(prev => prev.map(s => ({ ...s, flash: null })));
            }, 1000);
        } catch (error) {
            console.error('Failed to load market prices:', error);
        }
    };

    useEffect(() => {
        loadPrices();
        const interval = setInterval(() => loadPrices(), REFRESH_INTERVAL);
        return () => clearInterval(interval);
    }, []);

    const categories = ['core', 'etf', 'titan', 'regional', 'junior'];

    return (
        <div className="market-map-container">
            <div className="map-header">
                <div className="map-status">
                    STATUS: {lastUpdate ? 'LIVE' : 'CONNECTING...'}
                    {lastUpdate && <span className="text-titanium"> // LAST UPDATE: {lastUpdate.toLocaleTimeString()}</span>}
                </div>
            </div>

            <div className="market-grid">
                {categories.map(category => {
                    const categorySymbols = symbols.filter(s => s.category === category);
                    return (
                        <div key={category} className="map-column">
                            <div className="column-header">{CATEGORY_LABELS[category]}</div>
                            <div className="column-cards">
                                {categorySymbols.map(sym => {
                                    const isUp = (sym.change || 0) >= 0;
                                    const flashClass = sym.flash ? `flash-${sym.flash}` : '';

                                    return (
                                        <a
                                            key={sym.symbol}
                                            href={`https://finance.yahoo.com/quote/${sym.symbol}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className={`ticker-card ${flashClass}`}
                                            style={{ borderColor: isUp ? 'var(--bull-teal)' : 'var(--bear-rust)' }}
                                        >
                                            <div className="ticker-top">
                                                <span className="sym-code">{sym.symbol}</span>
                                                <span className={`sym-change ${isUp ? 'text-bull' : 'text-bear'}`}>
                                                    {formatChange(sym.change)}
                                                </span>
                                            </div>
                                            <div className="ticker-name">{sym.name}</div>
                                            <div className="ticker-price">${sym.price?.toFixed(2) || '--'}</div>
                                        </a>
                                    );
                                })}
                            </div>
                        </div>
                    );
                })}
            </div>
        </div>
    );
}
