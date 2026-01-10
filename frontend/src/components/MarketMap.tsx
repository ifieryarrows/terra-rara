import { useEffect, useState, useRef } from 'react';
import { fetchMarketPrices, type MarketPricesResponse } from '../api';
import clsx from 'clsx';

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
        <div className="p-2">
            <div className="flex justify-between items-end border-b border-white/5 pb-2 mb-6">
                <div className="flex items-center gap-2">
                    <div className={clsx("w-2 h-2 rounded-full", lastUpdate ? "bg-emerald-500 animate-pulse" : "bg-gray-500")} />
                    <span className="text-xs font-mono text-gray-400 tracking-wider">
                        {lastUpdate ? 'MARKET FEED ACTIVE' : 'CONNECTING...'}
                    </span>
                </div>
                {lastUpdate && (
                    <span className="text-[10px] font-mono text-gray-600">
                        UPDATED: {lastUpdate.toLocaleTimeString()}
                    </span>
                )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
                {categories.map(category => {
                    const categorySymbols = symbols.filter(s => s.category === category);
                    return (
                        <div key={category} className="flex flex-col gap-3">
                            <div className="text-[10px] font-bold text-gray-500 uppercase tracking-widest text-center py-2 border-b border-white/5 border-dashed">
                                {CATEGORY_LABELS[category]}
                            </div>
                            <div className="flex flex-col gap-2">
                                {categorySymbols.map(sym => {
                                    const isUp = (sym.change || 0) >= 0;

                                    return (
                                        <a
                                            key={sym.symbol}
                                            href={`https://finance.yahoo.com/quote/${sym.symbol}`}
                                            target="_blank"
                                            rel="noopener noreferrer"
                                            className={clsx(
                                                "group relative block p-3 rounded-xl border transition-all duration-300",
                                                "bg-white/5 hover:bg-white/10",
                                                isUp ? "border-emerald-500/20 hover:border-emerald-500/40" : "border-rose-500/20 hover:border-rose-500/40",
                                                sym.flash === 'up' && "animate-[flash-emerald_1s_ease-out]",
                                                sym.flash === 'down' && "animate-[flash-rose_1s_ease-out]"
                                            )}
                                        >
                                            <div className="flex justify-between items-start mb-1">
                                                <span className="text-xs font-bold text-gray-300 font-mono group-hover:text-white transition-colors">
                                                    {sym.symbol}
                                                </span>
                                                <span className={clsx(
                                                    "text-xs font-mono font-medium",
                                                    isUp ? "text-emerald-400" : "text-rose-400"
                                                )}>
                                                    {formatChange(sym.change)}
                                                </span>
                                            </div>
                                            <div className="text-[10px] text-gray-500 uppercase tracking-wide truncate mb-2">
                                                {sym.name}
                                            </div>
                                            <div className="text-right font-mono text-sm text-gray-200">
                                                ${sym.price?.toFixed(2) || '--'}
                                            </div>
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
