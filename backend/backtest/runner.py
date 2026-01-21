"""
Champion/Challenger Backtest Script

Rolling 6-year train window with weekly retrain and daily 1D predictions.
Compares model performance between champion and challenger symbol sets.

Audit-ready outputs:
- backtest_report.json: Summary metrics and decision
- predictions.csv: Daily predictions with timestamps

Usage:
    python -m backend.backtest.runner --champion config/symbol_sets/champion.json --challenger runs/latest/selected_symbols.json
"""

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""
    # Time parameters
    oos_start: str = "2024-01-01"
    oos_end: str = "2025-01-17"
    train_window_years: int = 6
    retrain_frequency: str = "weekly"  # Monday
    prediction_horizon: int = 1  # days
    
    # Model parameters
    random_seed: int = 42
    xgb_params: dict = None
    
    # Promote thresholds
    promote_threshold_pct: float = 5.0  # Champion MAE must improve by 5%
    reject_threshold_pct: float = -5.0  # Challenger MAE 5% worse = reject
    
    def __post_init__(self):
        if self.xgb_params is None:
            self.xgb_params = {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": self.random_seed,
                "n_jobs": -1
            }


@dataclass
class SymbolSet:
    """A set of symbols with metadata."""
    name: str
    symbols: list[str]
    version: str
    source_path: str
    content_hash: str


@dataclass
class BacktestMetrics:
    """Metrics from a backtest run."""
    # Price metrics
    mae: float
    rmse: float
    n_predictions: int
    mean_actual: float
    mean_predicted: float
    # Direction metrics
    directional_accuracy: float
    precision_up: float = 0.0
    recall_up: float = 0.0
    mcc: float = 0.0  # Matthews Correlation Coefficient
    confusion_matrix: dict = None
    # Baselines
    baseline_always_up: float = 0.0
    baseline_always_down: float = 0.0
    baseline_repeat: float = 0.0


@dataclass
class BacktestResult:
    """Full backtest result with comparison."""
    run_id: str
    generated_at: str
    config: BacktestConfig
    champion: dict  # SymbolSet + BacktestMetrics
    challenger: dict  # SymbolSet + BacktestMetrics
    delta_mae_pct: float
    delta_rmse_pct: float
    delta_dir_acc_pct: float
    decision: str  # PROMOTE | REJECT | MANUAL_REVIEW
    decision_reason: str


def compute_content_hash(data: dict) -> str:
    """Compute deterministic hash of symbol set."""
    # Sort symbols for determinism
    symbols = sorted(data.get("symbols", []))
    canonical = json.dumps({"symbols": symbols}, sort_keys=True)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def load_symbol_set(path: str | Path) -> SymbolSet:
    """Load symbol set from JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    
    # Handle selected_symbols.json format (has "selected" key with objects)
    if "selected" in data:
        symbols = [s["ticker"] for s in data["selected"]]
        name = data.get("screener_run_id", "challenger")
        version = data.get("selection_rules_version", "unknown")
    else:
        symbols = data.get("symbols", [])
        name = data.get("name", "unknown")
        version = data.get("version", "unknown")
    
    return SymbolSet(
        name=name,
        symbols=symbols,
        version=version,
        source_path=str(path),
        content_hash=compute_content_hash({"symbols": symbols})
    )


def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Get trading days in range (approximate - weekdays only)."""
    dates = pd.date_range(start=start, end=end, freq='B')  # Business days
    return dates


def get_retrain_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Get Monday retrain dates from trading days."""
    # Get first trading day of each week
    weekly = trading_days.to_series().groupby(pd.Grouper(freq='W-MON')).first()
    return pd.DatetimeIndex(weekly.dropna())


class BacktestRunner:
    """
    Run champion/challenger backtest.
    
    Implements:
    - Rolling 6-year train window
    - Weekly retrain on Mondays
    - Daily 1D predictions
    - No lookahead (strict asof convention)
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.run_id = f"backtest-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"
    
    def fetch_prices(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """
        Fetch historical prices for symbols.
        
        Returns DataFrame with columns: date, symbol, close
        """
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required: pip install yfinance")
        
        # Extend start to include train window
        train_start = (pd.Timestamp(start) - pd.DateOffset(years=self.config.train_window_years + 1)).strftime('%Y-%m-%d')
        
        all_data = []
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=train_start, end=end, interval="1d")
                if not hist.empty:
                    df = hist[['Close']].reset_index()
                    df.columns = ['date', 'close']
                    df['symbol'] = symbol
                    all_data.append(df)
            except Exception as e:
                logger.warning(f"Failed to fetch {symbol}: {e}")
        
        if not all_data:
            raise ValueError("No price data fetched")
        
        result = pd.concat(all_data, ignore_index=True)
        # Handle tz-aware dates from yfinance
        result['date'] = pd.to_datetime(result['date'], utc=True).dt.tz_localize(None)
        return result
    
    def prepare_features(self, prices: pd.DataFrame, target_symbol: str = "HG=F") -> pd.DataFrame:
        """
        Prepare feature matrix for modeling.
        
        Creates lag features, returns, and rolling metrics.
        """
        # Pivot to wide format
        pivot = prices.pivot(index='date', columns='symbol', values='close')
        pivot = pivot.sort_index()
        
        # Normalize index to date only (remove time component for matching)
        pivot.index = pd.to_datetime(pivot.index).normalize()
        
        # Forward fill missing values for symbols with sparse data
        pivot = pivot.ffill()
        
        # Compute returns
        returns = pivot.pct_change()
        
        # Create feature DataFrame
        features = pd.DataFrame(index=pivot.index)
        
        # Target: next day close
        if target_symbol not in pivot.columns:
            raise ValueError(f"Target symbol {target_symbol} not in price data")
        
        features['y_target'] = pivot[target_symbol].shift(-1)  # Next day price
        features['y_current'] = pivot[target_symbol]
        
        # Features for each symbol
        for symbol in pivot.columns:
            if symbol == target_symbol:
                continue
            
            # Skip if symbol has too many missing values
            if pivot[symbol].isna().sum() > len(pivot) * 0.5:
                logger.warning(f"Skipping {symbol}: too many missing values")
                continue
            
            # Price ratio to target
            features[f'{symbol}_ratio'] = pivot[symbol] / pivot[target_symbol]
            
            # Returns
            features[f'{symbol}_ret_1d'] = returns[symbol]
            features[f'{symbol}_ret_5d'] = pivot[symbol].pct_change(5)
            
            # Rolling volatility
            features[f'{symbol}_vol_20d'] = returns[symbol].rolling(20).std()
        
        # Target's own features
        features['target_ret_1d'] = returns[target_symbol]
        features['target_ret_5d'] = pivot[target_symbol].pct_change(5)
        features['target_vol_20d'] = returns[target_symbol].rolling(20).std()
        features['target_mom_10d'] = pivot[target_symbol].pct_change(10)
        
        # Only drop rows where TARGET values are missing (not all features)
        features = features.dropna(subset=['y_target', 'y_current'])
        
        # Fill remaining NaN in features with 0 (for model training)
        features = features.fillna(0)
        
        return features
    
    def train_and_predict(
        self,
        features: pd.DataFrame,
        train_end: pd.Timestamp,
        predict_dates: list[pd.Timestamp]
    ) -> list[dict]:
        """
        Train model on data up to train_end and predict for predict_dates.
        
        Returns list of prediction records.
        """
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost required: pip install xgboost")
        
        # Train window: last N years
        train_start = train_end - pd.DateOffset(years=self.config.train_window_years)
        
        # Get training data
        train_mask = (features.index >= train_start) & (features.index <= train_end)
        train_data = features.loc[train_mask].copy()
        
        if len(train_data) < 100:
            logger.warning(f"Insufficient training data: {len(train_data)} rows (train_start={train_start.date()}, train_end={train_end.date()}, features range: {features.index.min().date()} to {features.index.max().date()})")
            return []
        
        # Prepare X, y
        feature_cols = [c for c in train_data.columns if c not in ['y_target', 'y_current']]
        X_train = train_data[feature_cols]
        y_train = train_data['y_target']
        
        # Train model
        model = XGBRegressor(**self.config.xgb_params)
        model.fit(X_train, y_train)
        
        # Predict for each date
        predictions = []
        for pred_date in predict_dates:
            if pred_date not in features.index:
                continue
            
            row = features.loc[[pred_date]]
            X_pred = row[feature_cols]
            y_pred = model.predict(X_pred)[0]
            y_current = row['y_current'].iloc[0]
            y_actual = row['y_target'].iloc[0]
            
            predictions.append({
                'date': pred_date,
                'y_pred': y_pred,
                'y_current': y_current,
                'y_actual': y_actual,
                'pred_return': (y_pred / y_current) - 1 if y_current else None,
                'actual_return': (y_actual / y_current) - 1 if y_current else None,
                'train_end': train_end,
                'train_samples': len(train_data)
            })
        
        return predictions
    
    def run_backtest(self, symbols: list[str]) -> tuple[BacktestMetrics, pd.DataFrame]:
        """
        Run full backtest for a symbol set.
        
        Returns metrics and prediction DataFrame.
        """
        logger.info(f"Running backtest with {len(symbols)} symbols")
        
        # Fetch prices
        target = "HG=F"
        all_symbols = list(set(symbols + [target]))
        prices = self.fetch_prices(all_symbols, self.config.oos_start, self.config.oos_end)
        
        logger.info(f"Fetched prices: {len(prices)} rows, date range: {prices['date'].min()} to {prices['date'].max()}")
        
        # Prepare features
        features = self.prepare_features(prices, target)
        
        logger.info(f"Features prepared: {len(features)} rows, date range: {features.index.min()} to {features.index.max()}")
        
        # Get trading days and retrain dates for OOS period
        trading_days = get_trading_days(self.config.oos_start, self.config.oos_end)
        retrain_dates = get_retrain_dates(trading_days)
        
        logger.info(f"OOS period: {self.config.oos_start} to {self.config.oos_end}")
        logger.info(f"Retrain dates: {len(retrain_dates)}")
        
        # Run rolling predictions
        all_predictions = []
        
        for i, retrain_date in enumerate(retrain_dates[:-1]):
            next_retrain = retrain_dates[i + 1] if i + 1 < len(retrain_dates) else pd.Timestamp(self.config.oos_end)
            
            # Train end is the day BEFORE retrain (no lookahead)
            train_end = retrain_date - pd.Timedelta(days=1)
            
            # Predict for days between retrains
            predict_dates = [d for d in trading_days if retrain_date <= d < next_retrain]
            
            if predict_dates:
                preds = self.train_and_predict(features, train_end, predict_dates)
                all_predictions.extend(preds)
        
        if not all_predictions:
            raise ValueError("No predictions generated")
        
        # Convert to DataFrame
        pred_df = pd.DataFrame(all_predictions)
        pred_df = pred_df.dropna(subset=['y_actual', 'y_pred'])
        
        # Compute price metrics
        mae = np.abs(pred_df['y_actual'] - pred_df['y_pred']).mean()
        rmse = np.sqrt(((pred_df['y_actual'] - pred_df['y_pred']) ** 2).mean())
        
        # Compute DIRECTION metrics properly
        # Convert returns to binary direction: 1 = up, 0 = down/flat
        pred_df['actual_dir'] = (pred_df['actual_return'] > 0).astype(int)
        pred_df['pred_dir'] = (pred_df['pred_return'] > 0).astype(int)
        
        # Directional accuracy (hit rate)
        pred_df['dir_correct'] = pred_df['pred_dir'] == pred_df['actual_dir']
        dir_acc = pred_df['dir_correct'].mean()
        
        # Confusion matrix components
        tp = ((pred_df['pred_dir'] == 1) & (pred_df['actual_dir'] == 1)).sum()
        tn = ((pred_df['pred_dir'] == 0) & (pred_df['actual_dir'] == 0)).sum()
        fp = ((pred_df['pred_dir'] == 1) & (pred_df['actual_dir'] == 0)).sum()
        fn = ((pred_df['pred_dir'] == 0) & (pred_df['actual_dir'] == 1)).sum()
        
        # Precision, Recall for UP predictions
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Matthews Correlation Coefficient (MCC) - best single metric
        mcc_num = (tp * tn) - (fp * fn)
        mcc_den = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = mcc_num / mcc_den if mcc_den > 0 else 0
        
        # Baselines
        baseline_always_up = pred_df['actual_dir'].mean()  # If always predict UP
        baseline_always_down = 1 - baseline_always_up  # If always predict DOWN
        
        # Last direction repeat baseline
        pred_df['prev_dir'] = pred_df['actual_dir'].shift(1)
        pred_df['repeat_correct'] = pred_df['actual_dir'] == pred_df['prev_dir']
        baseline_repeat = pred_df['repeat_correct'].dropna().mean()
        
        metrics = BacktestMetrics(
            mae=round(mae, 6),
            rmse=round(rmse, 6),
            directional_accuracy=round(dir_acc, 4),
            n_predictions=len(pred_df),
            mean_actual=round(pred_df['y_actual'].mean(), 4),
            mean_predicted=round(pred_df['y_pred'].mean(), 4),
            # Extended direction metrics
            precision_up=round(precision, 4),
            recall_up=round(recall, 4),
            mcc=round(mcc, 4),
            confusion_matrix={"tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn)},
            baseline_always_up=round(baseline_always_up, 4),
            baseline_always_down=round(baseline_always_down, 4),
            baseline_repeat=round(baseline_repeat, 4)
        )
        
        return metrics, pred_df
    
    def compare(
        self,
        champion_set: SymbolSet,
        challenger_set: SymbolSet
    ) -> BacktestResult:
        """
        Run backtest for both sets and compare.
        """
        logger.info(f"=== CHAMPION: {champion_set.name} ({len(champion_set.symbols)} symbols) ===")
        champion_metrics, champion_preds = self.run_backtest(champion_set.symbols)
        champion_preds['symbol_set'] = 'champion'
        
        logger.info(f"=== CHALLENGER: {challenger_set.name} ({len(challenger_set.symbols)} symbols) ===")
        challenger_metrics, challenger_preds = self.run_backtest(challenger_set.symbols)
        challenger_preds['symbol_set'] = 'challenger'
        
        # Combine predictions
        all_preds = pd.concat([champion_preds, challenger_preds], ignore_index=True)
        
        # Compute deltas
        delta_mae = ((champion_metrics.mae - challenger_metrics.mae) / champion_metrics.mae) * 100
        delta_rmse = ((champion_metrics.rmse - challenger_metrics.rmse) / champion_metrics.rmse) * 100
        delta_dir = ((challenger_metrics.directional_accuracy - champion_metrics.directional_accuracy) / champion_metrics.directional_accuracy) * 100
        
        # Decision
        if delta_mae >= self.config.promote_threshold_pct:
            decision = "PROMOTE"
            reason = f"Challenger MAE {delta_mae:.1f}% better than champion"
        elif delta_mae <= self.config.reject_threshold_pct:
            decision = "REJECT"
            reason = f"Challenger MAE {-delta_mae:.1f}% worse than champion"
        else:
            decision = "MANUAL_REVIEW"
            reason = f"MAE delta {delta_mae:.1f}% within threshold band"
        
        result = BacktestResult(
            run_id=self.run_id,
            generated_at=datetime.utcnow().isoformat() + "Z",
            config=self.config,
            champion={
                "symbol_set": asdict(champion_set),
                "metrics": asdict(champion_metrics)
            },
            challenger={
                "symbol_set": asdict(challenger_set),
                "metrics": asdict(challenger_metrics)
            },
            delta_mae_pct=round(delta_mae, 2),
            delta_rmse_pct=round(delta_rmse, 2),
            delta_dir_acc_pct=round(delta_dir, 2),
            decision=decision,
            decision_reason=reason
        )
        
        return result, all_preds


def main():
    parser = argparse.ArgumentParser(description="Champion/Challenger Backtest")
    parser.add_argument("--champion", required=True, help="Path to champion symbol set JSON")
    parser.add_argument("--challenger", required=True, help="Path to challenger symbol set JSON")
    parser.add_argument("--output-dir", default="backend/artifacts/backtests", help="Output directory")
    parser.add_argument("--oos-start", default="2024-01-01", help="OOS start date")
    parser.add_argument("--oos-end", default="2025-01-17", help="OOS end date")
    args = parser.parse_args()
    
    # Load symbol sets
    logger.info(f"Loading champion from: {args.champion}")
    champion = load_symbol_set(args.champion)
    
    logger.info(f"Loading challenger from: {args.challenger}")
    challenger = load_symbol_set(args.challenger)
    
    # Configure and run
    config = BacktestConfig(
        oos_start=args.oos_start,
        oos_end=args.oos_end
    )
    
    runner = BacktestRunner(config)
    result, predictions = runner.compare(champion, challenger)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save report
    report_path = output_dir / f"{result.run_id}_report.json"
    with open(report_path, 'w') as f:
        json.dump(asdict(result), f, indent=2, default=str)
    logger.info(f"Report saved: {report_path}")
    
    # Save predictions
    preds_path = output_dir / f"{result.run_id}_predictions.csv"
    predictions.to_csv(preds_path, index=False)
    logger.info(f"Predictions saved: {preds_path}")
    
    # Print summary
    print("\n" + "=" * 60)
    print(f"BACKTEST RESULT: {result.decision}")
    print("=" * 60)
    print(f"Champion MAE:    {result.champion['metrics']['mae']:.6f}")
    print(f"Challenger MAE:  {result.challenger['metrics']['mae']:.6f}")
    print(f"Delta MAE:       {result.delta_mae_pct:+.2f}%")
    print(f"Decision:        {result.decision}")
    print(f"Reason:          {result.decision_reason}")
    print("=" * 60)


if __name__ == "__main__":
    main()
