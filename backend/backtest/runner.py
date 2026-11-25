"""
Champion/Challenger Backtest Script

Evaluates:
1. XGBoost rolling walk-forward T+1 return backtest comparing Champion vs
   Challenger symbol sets with stationary features, strict zero target-date
   lookahead leak, shared price caching, and paired block-bootstrap significance.
2. Promoted TFT-ASRO walk-forward 5-day cumulative log-return backtest with RSS
   quantile aggregation, Split CQR calibration, HAC Sharpe, and horizon-matched
   Theta baseline comparison (no moving-average stubs).

Usage:
    python -m backend.backtest.runner --champion config/symbol_sets/active.json --challenger config/symbol_sets/challenger.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
import tempfile
from typing import Any, Optional

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _ensure_yf_tz_cache() -> None:
    """Point yfinance's tz cache at a writable temp dir before any download call."""
    try:
        from screener.core.cache import ensure_yf_tz_cache

        ensure_yf_tz_cache()
        return
    except Exception:
        pass
    try:
        import yfinance as yf

        cache_dir = Path(tempfile.gettempdir()) / "py-yfinance-tzcache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(yf, "set_tz_cache_location"):
            yf.set_tz_cache_location(str(cache_dir))
    except Exception as exc:
        logger.debug("Could not configure yfinance tz cache location: %s", exc)


def _normalize_exchange_dates(values: Any) -> pd.DatetimeIndex | pd.Series:
    """Normalize timestamps to tz-naive exchange-local calendar midnights."""
    if isinstance(values, pd.DatetimeIndex):
        stripped = values.tz_localize(None) if values.tz is not None else values
        return stripped.normalize()

    if isinstance(values, pd.Series):
        if isinstance(values.dtype, pd.DatetimeTZDtype):
            return values.dt.tz_localize(None).dt.normalize()
        if pd.api.types.is_datetime64_any_dtype(values.dtype):
            return values.dt.normalize()

        normalized_dates = []
        for raw_val in values.tolist():
            if pd.isna(raw_val):
                normalized_dates.append(pd.NaT)
            elif isinstance(raw_val, (datetime, pd.Timestamp)):
                normalized_dates.append(raw_val.date())
            else:
                try:
                    parsed = pd.Timestamp(raw_val)
                    normalized_dates.append(parsed.date() if pd.notna(parsed) else pd.NaT)
                except Exception:
                    normalized_dates.append(pd.NaT)
        return pd.Series(pd.to_datetime(normalized_dates), index=values.index)

    idx = pd.DatetimeIndex(values)
    return idx.tz_localize(None).normalize() if idx.tz is not None else idx.normalize()


def _default_oos_dates() -> tuple[str, str]:
    """Return dynamic rolling 365-day OOS window ending today UTC."""
    today = datetime.now(timezone.utc).date()
    start = today - timedelta(days=365)
    return start.isoformat(), today.isoformat()


def _to_json_safe(obj: Any) -> Any:
    """Recursively coerce numpy/pandas scalars and non-finite floats into JSON-native values."""
    if obj is None or obj is pd.NaT:
        return None
    if isinstance(obj, dict):
        return {str(k): _to_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_json_safe(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        val = float(obj)
        return val if math.isfinite(val) else None
    if isinstance(obj, np.ndarray):
        return [_to_json_safe(v) for v in obj.tolist()]
    if isinstance(obj, (pd.Timestamp, datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, str):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        return _to_json_safe(asdict(obj))
    try:
        json.dumps(obj)
        return obj
    except TypeError:
        return str(obj)


@dataclass
class BacktestConfig:
    """Configuration for backtest run."""

    oos_start: str = ""
    oos_end: str = ""
    train_window_years: int = 6
    retrain_frequency: str = "weekly"  # Monday
    prediction_horizon: int = 1  # days

    random_seed: int = 42
    xgb_params: Optional[dict] = None

    # Return-space MAE percentage improvement thresholds
    promote_threshold_pct: float = 2.0
    reject_threshold_pct: float = -2.0
    bootstrap_samples: int = 1000
    bootstrap_block_size: int = 5

    def __post_init__(self):
        default_start, default_end = _default_oos_dates()
        if not self.oos_end:
            self.oos_end = default_end
        if not self.oos_start:
            end_dt = pd.Timestamp(self.oos_end).date()
            self.oos_start = (end_dt - timedelta(days=365)).isoformat()
        if self.xgb_params is None:
            self.xgb_params = {
                "n_estimators": 150,
                "max_depth": 4,
                "learning_rate": 0.05,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_lambda": 2.0,
                "random_state": self.random_seed,
                "n_jobs": -1,
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
    """Return-space and directional metrics from a backtest run."""

    mae: float
    rmse: float
    n_predictions: int
    mean_actual: float
    mean_predicted: float
    directional_accuracy: float
    precision_up: float = 0.0
    recall_up: float = 0.0
    mcc: float = 0.0
    confusion_matrix: Optional[dict] = None
    baseline_always_up: float = 0.0
    baseline_always_down: float = 0.0
    baseline_repeat: float = 0.0
    mae_vs_naive_zero: float = 1.0
    sharpe_ratio: float = 0.0
    price_mae: float = 0.0
    metric_space: str = "simple_return_1d"


@dataclass
class BacktestResult:
    """Full champion/challenger comparison result."""

    run_id: str
    generated_at: str
    config: BacktestConfig
    champion: dict
    challenger: dict
    delta_mae_pct: float
    delta_rmse_pct: float
    delta_dir_acc_pct: float
    improvement_mae_pct: float
    decision: str  # PROMOTE | REJECT | MANUAL_REVIEW | KEEP_CHAMPION
    decision_reason: str
    ci95_improvement_mae_pct: list[float] = field(default_factory=lambda: [0.0, 0.0])


def compute_content_hash(data: dict) -> str:
    """Compute deterministic hash of symbol set."""
    symbols = sorted(data.get("symbols", []))
    canonical = json.dumps({"symbols": symbols}, sort_keys=True)
    return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()[:16]}"


def load_symbol_set(path: str | Path) -> SymbolSet:
    """Load symbol set from JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)

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
        content_hash=compute_content_hash({"symbols": symbols}),
    )


def get_trading_days(start: str, end: str) -> pd.DatetimeIndex:
    """Get trading days in range (weekdays)."""
    return pd.date_range(start=start, end=end, freq="B")


def get_retrain_dates(trading_days: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Get Monday retrain dates from trading days."""
    weekly = trading_days.to_series().groupby(pd.Grouper(freq="W-MON")).first()
    return pd.DatetimeIndex(weekly.dropna())


def _compute_rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """Causal Wilder-style simple rolling RSI scaled into [-1, 1]."""
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0.0)).rolling(window, min_periods=window).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return ((rsi - 50.0) / 50.0).fillna(0.0)


def _paired_block_bootstrap_improvement_ci(
    champ_preds: pd.DataFrame,
    chall_preds: pd.DataFrame,
    *,
    n_boot: int = 1000,
    block_size: int = 5,
    seed: int = 42,
) -> tuple[float, float]:
    """Compute paired circular block-bootstrap 95% CI on MAE % improvement."""
    if champ_preds.empty or chall_preds.empty:
        return 0.0, 0.0
    merged = pd.merge(
        champ_preds[["date", "actual_return", "pred_return"]].rename(
            columns={"pred_return": "champ_pred"}
        ),
        chall_preds[["date", "pred_return"]].rename(columns={"pred_return": "chall_pred"}),
        on="date",
        how="inner",
    ).dropna()
    if len(merged) < 10:
        return 0.0, 0.0

    err_champ = np.abs(merged["actual_return"].to_numpy() - merged["champ_pred"].to_numpy())
    err_chall = np.abs(merged["actual_return"].to_numpy() - merged["chall_pred"].to_numpy())
    n = len(err_champ)
    eff_block = max(1, min(block_size, n))
    n_blocks = int(np.ceil(n / eff_block))

    rng = np.random.default_rng(seed)
    improvements = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = np.concatenate([(np.arange(eff_block) + s) % n for s in starts])[:n]
        base = float(err_champ[idx].mean())
        if base <= 1e-12:
            improvements[b] = 0.0
        else:
            improvements[b] = ((base - float(err_chall[idx].mean())) / base) * 100.0

    low, high = np.percentile(improvements, [2.5, 97.5])
    return float(low), float(high)


def _compute_backtest_metrics(
    pred_df: pd.DataFrame,
    *,
    annualisation: float = 252.0,
    use_hac: bool = False,
    metric_space: str = "simple_return_1d",
) -> BacktestMetrics:
    """Compute return-space and directional metrics on evaluated predictions."""
    actual_ret = pred_df["actual_return"].to_numpy(dtype=float)
    pred_ret = pred_df["pred_return"].to_numpy(dtype=float)

    mae = float(np.abs(actual_ret - pred_ret).mean())
    rmse = float(np.sqrt(np.mean((actual_ret - pred_ret) ** 2)))
    naive_zero_mae = float(np.abs(actual_ret).mean())
    mae_vs_naive = float(mae / naive_zero_mae) if naive_zero_mae > 1e-12 else 1.0

    strategy_returns = np.sign(pred_ret) * actual_ret
    if use_hac:
        from deep_learning.training.metrics import sharpe_ratio_hac

        sharpe = float(sharpe_ratio_hac(strategy_returns, annualisation=annualisation))
    else:
        std = float(strategy_returns.std())
        sharpe = float(np.sqrt(annualisation) * strategy_returns.mean() / std) if std > 1e-9 else 0.0

    price_mae = 0.0
    if "y_actual" in pred_df.columns and "y_pred" in pred_df.columns:
        valid_px = pred_df[["y_actual", "y_pred"]].dropna()
        if not valid_px.empty:
            price_mae = float(np.abs(valid_px["y_actual"] - valid_px["y_pred"]).mean())

    actual_dir = (actual_ret > 0).astype(int)
    pred_dir = (pred_ret > 0).astype(int)
    dir_acc = float((pred_dir == actual_dir).mean())

    tp = int(((pred_dir == 1) & (actual_dir == 1)).sum())
    tn = int(((pred_dir == 0) & (actual_dir == 0)).sum())
    fp = int(((pred_dir == 1) & (actual_dir == 0)).sum())
    fn = int(((pred_dir == 0) & (actual_dir == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    mcc_num = (tp * tn) - (fp * fn)
    mcc_den = float(np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float(mcc_num / mcc_den) if mcc_den > 0 else 0.0

    baseline_always_up = float(actual_dir.mean())
    baseline_always_down = 1.0 - baseline_always_up
    if len(actual_dir) > 1:
        baseline_repeat = float((actual_dir[1:] == actual_dir[:-1]).mean())
    else:
        baseline_repeat = 0.0

    return BacktestMetrics(
        mae=round(mae, 6),
        rmse=round(rmse, 6),
        directional_accuracy=round(dir_acc, 4),
        n_predictions=len(pred_df),
        mean_actual=round(float(actual_ret.mean()), 6),
        mean_predicted=round(float(pred_ret.mean()), 6),
        precision_up=round(precision, 4),
        recall_up=round(recall, 4),
        mcc=round(mcc, 4),
        confusion_matrix={"tp": tp, "tn": tn, "fp": fp, "fn": fn},
        baseline_always_up=round(baseline_always_up, 4),
        baseline_always_down=round(baseline_always_down, 4),
        baseline_repeat=round(baseline_repeat, 4),
        mae_vs_naive_zero=round(mae_vs_naive, 4),
        sharpe_ratio=round(sharpe, 4),
        price_mae=round(price_mae, 6),
        metric_space=metric_space,
    )


class BacktestRunner:
    """
    Run champion/challenger XGBoost backtest.

    Implements:
    - Shared per-symbol price caching across champion and challenger
    - Strictly stationary features and T+1 simple return targets
    - Rolling N-year train window with strict target_date <= train_end cutoff
    - Weekly Monday retrains and paired block-bootstrap error comparison
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.run_id = f"backtest-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self._price_cache: dict[tuple[str, str, str], pd.DataFrame] = {}

    def fetch_prices(self, symbols: list[str], start: str, end: str) -> pd.DataFrame:
        """Fetch historical daily close prices with shared instance cache."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance required: pip install yfinance")

        _ensure_yf_tz_cache()

        train_start = (
            pd.Timestamp(start) - pd.DateOffset(years=self.config.train_window_years + 1)
        ).strftime("%Y-%m-%d")

        missing: list[str] = []
        cached_frames: list[pd.DataFrame] = []
        for sym in dict.fromkeys(symbols):
            key = (sym, train_start, end)
            if key in self._price_cache:
                cached_frames.append(self._price_cache[key])
            else:
                missing.append(sym)

        # Fast batch download when multiple uncached symbols are needed
        if len(missing) > 1 and hasattr(yf, "download"):
            try:
                batch = yf.download(
                    missing,
                    start=train_start,
                    end=end,
                    interval="1d",
                    group_by="ticker",
                    auto_adjust=True,
                    threads=False,
                    progress=False,
                )
                if batch is not None and not batch.empty:
                    still_missing: list[str] = []
                    for sym in missing:
                        sub = None
                        if isinstance(batch.columns, pd.MultiIndex) and sym in batch.columns.get_level_values(0):
                            sub = batch[sym]
                        elif "Close" in batch.columns and len(missing) == 1:
                            sub = batch
                        if sub is not None and "Close" in sub.columns:
                            close_df = sub[["Close"]].dropna().reset_index()
                            if not close_df.empty:
                                close_df.columns = ["date", "close"]
                                close_df["date"] = _normalize_exchange_dates(close_df["date"])
                                close_df = close_df.dropna(subset=["date"]).drop_duplicates(
                                    subset=["date"], keep="last"
                                )
                                close_df["symbol"] = sym
                                self._price_cache[(sym, train_start, end)] = close_df
                                cached_frames.append(close_df)
                                continue
                        still_missing.append(sym)
                    missing = still_missing
            except Exception as exc:
                logger.debug("Batch yfinance download fell back to per-symbol fetch: %s", exc)

        for symbol in missing:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(start=train_start, end=end, interval="1d")
                if not hist.empty and "Close" in hist.columns:
                    df = hist[["Close"]].dropna().reset_index()
                    df.columns = ["date", "close"]
                    df["date"] = _normalize_exchange_dates(df["date"])
                    df = df.dropna(subset=["date"]).drop_duplicates(subset=["date"], keep="last")
                    df["symbol"] = symbol
                    self._price_cache[(symbol, train_start, end)] = df
                    cached_frames.append(df)
            except Exception as exc:
                logger.warning("Failed to fetch %s: %s", symbol, exc)

        if not cached_frames:
            raise ValueError("No price data fetched")

        result = pd.concat(cached_frames, ignore_index=True)
        result["date"] = _normalize_exchange_dates(result["date"])
        return result

    def prepare_features(self, prices: pd.DataFrame, target_symbol: str = "HG=F") -> pd.DataFrame:
        """
        Prepare strictly stationary feature matrix predicting T+1 simple return.

        Avoids raw price ratios (``symbol / target``) and raw price targets.
        Records ``target_date`` so training can enforce zero lookahead leak.
        """
        norm_prices = prices.copy()
        norm_prices["date"] = _normalize_exchange_dates(norm_prices["date"])
        norm_prices = norm_prices.dropna(subset=["date"])

        pivot = norm_prices.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
        pivot = pivot.groupby(level=0).last().sort_index()
        pivot.index = _normalize_exchange_dates(pivot.index)

        if target_symbol not in pivot.columns:
            raise ValueError(f"Target symbol {target_symbol} not in price data")

        target_valid_dates = pivot[target_symbol].dropna().index
        aux_cols = [c for c in pivot.columns if c != target_symbol]
        if aux_cols:
            pivot[aux_cols] = pivot[aux_cols].ffill(limit=5)
        pivot = pivot.loc[target_valid_dates]

        target_close = pivot[target_symbol]
        target_ret_1d = target_close.pct_change()
        next_close = target_close.shift(-1)

        cols: dict[str, pd.Series] = {
            "y_current": target_close,
            "y_price_target": next_close,
            "y_target": (next_close / target_close) - 1.0,
            "target_date": pd.Series(pivot.index, index=pivot.index).shift(-1),
        }

        for symbol in pivot.columns:
            if symbol == target_symbol:
                continue
            series = pivot[symbol]
            if series.isna().sum() > len(pivot) * 0.5:
                logger.warning("Skipping %s: too many missing values", symbol)
                continue

            ret_1d = series.pct_change()
            sma20 = series.rolling(20, min_periods=10).mean()
            sma60 = series.rolling(60, min_periods=20).mean()
            cols[f"{symbol}_ret_1d"] = ret_1d
            cols[f"{symbol}_ret_5d"] = series.pct_change(5)
            cols[f"{symbol}_ret_20d"] = series.pct_change(20)
            cols[f"{symbol}_vol_20d"] = ret_1d.rolling(20, min_periods=10).std()
            cols[f"{symbol}_sma20_ratio"] = (series / sma20) - 1.0
            cols[f"{symbol}_sma60_ratio"] = (series / sma60) - 1.0
            cols[f"{symbol}_corr_target_20d"] = ret_1d.rolling(20, min_periods=10).corr(target_ret_1d)

        t_sma20 = target_close.rolling(20, min_periods=10).mean()
        t_sma60 = target_close.rolling(60, min_periods=20).mean()
        cols["target_ret_1d"] = target_ret_1d
        cols["target_ret_5d"] = target_close.pct_change(5)
        cols["target_ret_20d"] = target_close.pct_change(20)
        cols["target_mom_10d"] = target_close.pct_change(10)
        cols["target_vol_20d"] = target_ret_1d.rolling(20, min_periods=10).std()
        cols["target_sma20_ratio"] = (target_close / t_sma20) - 1.0
        cols["target_sma60_ratio"] = (target_close / t_sma60) - 1.0
        cols["target_rsi_14"] = _compute_rsi(target_close, 14)

        features = pd.DataFrame(cols, index=pivot.index)
        features = features.dropna(subset=["y_target", "y_current", "target_date"]).copy()
        feature_cols = [
            c for c in features.columns if c not in {"y_target", "y_price_target", "y_current", "target_date"}
        ]
        features[feature_cols] = features[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        return features

    def train_and_predict(
        self,
        features: pd.DataFrame,
        train_end: pd.Timestamp,
        predict_dates: list[pd.Timestamp],
    ) -> list[dict]:
        """Train XGBRegressor strictly on settled rows (`target_date <= train_end`)."""
        try:
            from xgboost import XGBRegressor
        except ImportError:
            raise ImportError("xgboost required: pip install xgboost")

        train_start = train_end - pd.DateOffset(years=self.config.train_window_years)
        if "target_date" in features.columns:
            train_mask = (
                (features.index >= train_start)
                & (features.index <= train_end)
                & (pd.to_datetime(features["target_date"]) <= train_end)
            )
        else:
            train_mask = (features.index >= train_start) & (features.index < train_end)

        train_data = features.loc[train_mask].copy()
        if len(train_data) < 100:
            logger.warning(
                "Insufficient training data: %d rows (train_start=%s, train_end=%s)",
                len(train_data),
                train_start.date(),
                train_end.date(),
            )
            return []

        excluded = {"y_target", "y_price_target", "y_current", "target_date"}
        feature_cols = [c for c in train_data.columns if c not in excluded]
        X_train = train_data[feature_cols]
        y_train = train_data["y_target"].astype(float)

        model = XGBRegressor(**self.config.xgb_params)
        model.fit(X_train, y_train)

        predictions = []
        for pred_date in predict_dates:
            if pred_date not in features.index:
                continue

            row = features.loc[[pred_date]]
            X_pred = row[feature_cols]
            pred_ret = float(model.predict(X_pred)[0])
            y_current = float(row["y_current"].iloc[0])
            actual_ret = float(row["y_target"].iloc[0])
            y_actual = (
                float(row["y_price_target"].iloc[0])
                if "y_price_target" in row.columns
                else y_current * (1.0 + actual_ret)
            )
            y_pred = y_current * (1.0 + pred_ret)

            predictions.append(
                {
                    "date": pred_date,
                    "y_pred": y_pred,
                    "y_current": y_current,
                    "y_actual": y_actual,
                    "pred_return": pred_ret,
                    "actual_return": actual_ret,
                    "train_end": train_end,
                    "train_samples": len(train_data),
                }
            )

        return predictions

    def run_backtest(self, symbols: list[str]) -> tuple[BacktestMetrics, pd.DataFrame]:
        """Run rolling XGBoost walk-forward backtest for a symbol set."""
        logger.info("Running XGBoost backtest with %d symbols", len(symbols))
        target = "HG=F"
        all_symbols = list(dict.fromkeys(symbols + [target]))
        prices = self.fetch_prices(all_symbols, self.config.oos_start, self.config.oos_end)
        features = self.prepare_features(prices, target)

        trading_days = get_trading_days(self.config.oos_start, self.config.oos_end)
        retrain_dates = get_retrain_dates(trading_days)
        if len(retrain_dates) < 2:
            raise ValueError("Not enough retrain windows in OOS range")

        all_predictions = []
        for i, retrain_date in enumerate(retrain_dates[:-1]):
            next_retrain = retrain_dates[i + 1]
            train_end = retrain_date - pd.Timedelta(days=1)
            predict_dates = [d for d in trading_days if retrain_date <= d < next_retrain]
            if predict_dates:
                all_predictions.extend(self.train_and_predict(features, train_end, predict_dates))

        if not all_predictions:
            raise ValueError("No predictions generated")

        pred_df = pd.DataFrame(all_predictions).dropna(subset=["actual_return", "pred_return"]).copy()
        metrics = _compute_backtest_metrics(
            pred_df,
            annualisation=252.0,
            use_hac=False,
            metric_space="simple_return_1d",
        )
        return metrics, pred_df

    def compare(
        self,
        champion_set: SymbolSet,
        challenger_set: SymbolSet,
    ) -> tuple[BacktestResult, pd.DataFrame]:
        """Run champion vs challenger comparison, short-circuiting identical sets."""
        logger.info("=== CHAMPION: %s (%d symbols) ===", champion_set.name, len(champion_set.symbols))
        champion_metrics, champion_preds = self.run_backtest(champion_set.symbols)
        champion_preds["symbol_set"] = "champion"

        if champion_set.content_hash == challenger_set.content_hash:
            logger.info(
                "Champion and Challenger symbol sets share hash %s — skipping redundant challenger run",
                champion_set.content_hash,
            )
            challenger_metrics = BacktestMetrics(**asdict(champion_metrics))
            result = BacktestResult(
                run_id=self.run_id,
                generated_at=datetime.now(timezone.utc).isoformat(),
                config=self.config,
                champion={"symbol_set": asdict(champion_set), "metrics": asdict(champion_metrics)},
                challenger={"symbol_set": asdict(challenger_set), "metrics": asdict(challenger_metrics)},
                delta_mae_pct=0.0,
                delta_rmse_pct=0.0,
                delta_dir_acc_pct=0.0,
                improvement_mae_pct=0.0,
                decision="KEEP_CHAMPION",
                decision_reason="Identical symbol sets (matching content_hash); skipped duplicate run",
                ci95_improvement_mae_pct=[0.0, 0.0],
            )
            return result, champion_preds

        logger.info("=== CHALLENGER: %s (%d symbols) ===", challenger_set.name, len(challenger_set.symbols))
        challenger_metrics, challenger_preds = self.run_backtest(challenger_set.symbols)
        challenger_preds["symbol_set"] = "challenger"

        all_preds = pd.concat([champion_preds, challenger_preds], ignore_index=True)
        delta_mae = ((challenger_metrics.mae - champion_metrics.mae) / max(champion_metrics.mae, 1e-12)) * 100.0
        delta_rmse = ((challenger_metrics.rmse - champion_metrics.rmse) / max(champion_metrics.rmse, 1e-12)) * 100.0
        delta_dir = (
            (challenger_metrics.directional_accuracy - champion_metrics.directional_accuracy)
            / max(champion_metrics.directional_accuracy, 1e-9)
        ) * 100.0
        improvement_mae_pct = -delta_mae

        ci_low, ci_high = _paired_block_bootstrap_improvement_ci(
            champion_preds,
            challenger_preds,
            n_boot=self.config.bootstrap_samples,
            block_size=self.config.bootstrap_block_size,
            seed=self.config.random_seed,
        )

        if (
            improvement_mae_pct >= self.config.promote_threshold_pct
            and ci_low > 0.0
            and challenger_metrics.mcc >= champion_metrics.mcc - 0.02
        ):
            decision = "PROMOTE"
            reason = (
                f"Challenger MAE {improvement_mae_pct:.2f}% better "
                f"(95% CI [{ci_low:.2f}%, {ci_high:.2f}%], MCC {challenger_metrics.mcc:.3f})"
            )
        elif improvement_mae_pct <= self.config.reject_threshold_pct or ci_high < 0.0:
            decision = "REJECT"
            reason = (
                f"Challenger underperformed Champion "
                f"(improvement={improvement_mae_pct:.2f}%, 95% CI [{ci_low:.2f}%, {ci_high:.2f}%])"
            )
        else:
            decision = "MANUAL_REVIEW"
            reason = (
                f"MAE improvement {improvement_mae_pct:.2f}% not statistically decisive "
                f"(95% CI [{ci_low:.2f}%, {ci_high:.2f}%])"
            )

        result = BacktestResult(
            run_id=self.run_id,
            generated_at=datetime.now(timezone.utc).isoformat(),
            config=self.config,
            champion={"symbol_set": asdict(champion_set), "metrics": asdict(champion_metrics)},
            challenger={"symbol_set": asdict(challenger_set), "metrics": asdict(challenger_metrics)},
            delta_mae_pct=round(delta_mae, 2),
            delta_rmse_pct=round(delta_rmse, 2),
            delta_dir_acc_pct=round(delta_dir, 2),
            improvement_mae_pct=round(improvement_mae_pct, 2),
            decision=decision,
            decision_reason=reason,
            ci95_improvement_mae_pct=[round(ci_low, 2), round(ci_high, 2)],
        )
        return result, all_preds


class TFTBacktestRunner:
    """
    Authentic walk-forward backtest runner for promoted TFT-ASRO checkpoints.

    Loads the promoted TFT-ASRO checkpoint and CQR artifact, evaluates 5-day
    cumulative log-return predictions across the OOS window, computes rolling
    window diagnostics, and compares against a horizon-matched Theta baseline.
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.run_id = f"tft-backtest-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
        self.last_walkforward_report: dict[str, Any] = {}
        self.last_theta_comparison: dict[str, Any] = {}
        self.last_split_dates: dict[str, Any] = {}
        self.last_strict_oos_report: dict[str, Any] = {}
        self.last_strict_oos_theta_comparison: dict[str, Any] = {}
        self._last_known_price: float = float("nan")

    def _ensure_tft_artifacts(self, cfg) -> Path:
        """Ensure valid promoted TFT checkpoint & companions exist locally."""
        from deep_learning.models.hub import validate_tft_artifact_set

        ckpt_path = Path(cfg.training.best_model_path)
        tft_dir = ckpt_path.parent
        repo_id = cfg.training.hf_model_repo

        try:
            from deep_learning.models.hub import sync_tft_artifacts_if_stale

            sync_tft_artifacts_if_stale(local_dir=tft_dir, repo_id=repo_id)
        except ImportError:
            if not validate_tft_artifact_set(tft_dir):
                from deep_learning.models.hub import download_tft_artifacts

                download_tft_artifacts(local_dir=tft_dir, repo_id=repo_id)
        except Exception as exc:
            logger.warning("HF Hub sync before TFT backtest warned: %s", exc)

        if not validate_tft_artifact_set(tft_dir):
            raise RuntimeError(
                f"TFT artifact set unavailable or incomplete at {tft_dir}. "
                "Cannot execute authentic TFT walk-forward backtest."
            )
        return ckpt_path

    def _load_cqr_artifact(self, tft_dir: Path) -> Optional[dict]:
        cqr_path = tft_dir / "conformal_calibration.json"
        if not cqr_path.exists():
            return None
        try:
            payload = json.loads(cqr_path.read_text())
            if isinstance(payload, dict) and payload.get("cqr_enabled"):
                return payload
        except Exception as exc:
            logger.debug("Could not load conformal_calibration.json: %s", exc)
        return None

    def _load_metadata_artifact(self, tft_dir: Path) -> dict:
        meta_path = tft_dir / "tft_metadata.json"
        if not meta_path.exists():
            return {}
        try:
            payload = json.loads(meta_path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception as exc:
            logger.debug("Could not load tft_metadata.json: %s", exc)
            return {}

    def _build_eval_dataframe(self, cfg, checkpoint_features: list[str]) -> pd.DataFrame:
        """Build TFT evaluation DataFrame using DB session when available, else yfinance."""
        from dataclasses import replace
        from deep_learning.data.feature_store import build_tft_dataframe

        oos_start_ts = pd.Timestamp(self.config.oos_start)
        oos_end_ts = pd.Timestamp(self.config.oos_end)
        days_needed = max(
            int(cfg.training.lookback_days),
            int((datetime.now(timezone.utc).date() - oos_start_ts.date()).days) + 180,
        )
        needs_news = any(
            name.startswith(("emb_pca_", "sent_", "sentiment_", "event_", "evt_"))
            or name in {
                "news_count",
                "material_news_count",
                "after_close_news_count",
                "days_since_last_material_news",
                "stale_sentiment_flag",
            }
            for name in checkpoint_features
        )
        eval_cfg = replace(
            cfg,
            training=replace(cfg.training, lookback_days=days_needed),
            feature_store=replace(
                cfg.feature_store,
                mrmr_top_k=0,
                include_news_features=bool(needs_news or cfg.feature_store.include_news_features),
            ),
        )

        df = None
        last_known_price = float("nan")
        try:
            from app.db import SessionLocal

            with SessionLocal() as session:
                df, _, _, _, last_known_price = build_tft_dataframe(session, eval_cfg, drop_missing_target=True)
        except Exception as db_exc:
            logger.warning("DB-backed TFT feature build unavailable (%s); trying sessionless build", db_exc)

        if df is None or df.empty:
            df, _, _, _, last_known_price = build_tft_dataframe(None, eval_cfg, drop_missing_target=True)

        self._last_known_price = float(last_known_price) if last_known_price is not None else float("nan")

        if "target" not in df.columns or "time_idx" not in df.columns:
            raise RuntimeError("Evaluation DataFrame missing required 'target' or 'time_idx'")

        missing_cols = [col for col in checkpoint_features if col not in df.columns]
        if missing_cols:
            raise RuntimeError(
                "Backtest evaluation DataFrame missing required checkpoint features: "
                f"{missing_cols[:10]}"
            )

        idx_dt = pd.to_datetime(df.index).tz_localize(None)
        df = df.copy()
        df.index = idx_dt
        df = df[df.index <= oos_end_ts].sort_index()
        return df

    def _resolve_target_close_series(
        self,
        cfg,
        df: pd.DataFrame,
        last_known_price: float = float("nan"),
    ) -> pd.Series:
        """Resolve historical target close prices aligned to ``df`` without altering TFT inputs."""
        idx_norm = pd.to_datetime(df.index).tz_localize(None).normalize()
        target_symbol = cfg.feature_store.target_symbol
        end_ts = pd.Timestamp(self.config.oos_end)

        # 1. Prefer Postgres DB prices
        try:
            from app.db import SessionLocal
            from app.features import load_price_data

            start_dt = (pd.Timestamp(idx_norm.min()) - pd.Timedelta(days=10)).to_pydatetime()
            end_dt = (end_ts + pd.Timedelta(days=1)).to_pydatetime()
            with SessionLocal() as session:
                px_df = load_price_data(session, target_symbol, start_dt, end_dt)
            if px_df is not None and not px_df.empty and "close" in px_df.columns:
                series = pd.to_numeric(px_df["close"], errors="coerce").dropna()
                if not series.empty:
                    series.index = pd.to_datetime(series.index).tz_localize(None).normalize()
                    series = series[~series.index.duplicated(keep="last")].sort_index()
                    return series[series.index <= end_ts]
        except Exception as db_exc:
            logger.debug("DB target close resolution skipped (%s)", db_exc)

        # 2. Fallback: yfinance via BacktestRunner cache/loader
        try:
            start_str = str(pd.Timestamp(idx_norm.min()).date())
            px_table = BacktestRunner(self.config).fetch_prices([target_symbol], start_str, self.config.oos_end)
            if px_table is not None and not px_table.empty:
                sub = px_table[px_table["symbol"] == target_symbol]
                if not sub.empty and "close" in sub.columns:
                    sub_idx = pd.to_datetime(sub["date"]).dt.tz_localize(None).dt.normalize()
                    series = pd.Series(
                        pd.to_numeric(sub["close"], errors="coerce").to_numpy(),
                        index=sub_idx,
                        name="close",
                    ).dropna()
                    if not series.empty:
                        series = series[~series.index.duplicated(keep="last")].sort_index()
                        return series[series.index <= end_ts]
        except Exception as yf_exc:
            logger.debug("yfinance target close resolution skipped (%s)", yf_exc)

        # 3. Deterministic offline fallback from causal log returns anchored to last_known_price
        targets = pd.to_numeric(df["target"], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
        anchor = (
            float(last_known_price)
            if (last_known_price is not None and np.isfinite(last_known_price) and last_known_price > 0.0)
            else 5.0
        )
        cum_logs = np.cumsum(targets)
        log_prices = np.log(anchor) + (cum_logs - cum_logs[-1]) if cum_logs.size else np.array([], dtype=np.float64)
        series = pd.Series(np.exp(log_prices), index=idx_norm, name="close")
        return series[~series.index.duplicated(keep="last")].sort_index()

    def run_backtest(self, symbols: list[str]) -> tuple[BacktestMetrics, pd.DataFrame]:
        """Run authentic promoted TFT-ASRO walk-forward 5-day cumulative backtest."""
        logger.info("Running authentic TFT-ASRO backtest (%d symbol universe)", len(symbols))
        from pytorch_forecasting import TimeSeriesDataSet

        from deep_learning.calibration.conformal import apply_weekly_cqr
        from deep_learning.config import get_tft_config
        from deep_learning.inference.predictor import (
            _checkpoint_dataset_parameters,
            _checkpoint_feature_contract,
        )
        from deep_learning.models.monotonic_quantiles import aggregate_weekly_quantiles_np
        from deep_learning.training.metrics import monotonic_quantiles_np
        from deep_learning.models.tft_copper import load_tft_model
        from deep_learning.validation.backtest import compare_with_baseline, run_backtest as wf_backtest
        from deep_learning.baselines.theta_model import theta_backtest

        cfg = get_tft_config()
        ckpt_path = self._ensure_tft_artifacts(cfg)
        cqr_artifact = self._load_cqr_artifact(ckpt_path.parent)
        metadata = self._load_metadata_artifact(ckpt_path.parent)
        split_dates = metadata.get("split_dates") if isinstance(metadata.get("split_dates"), dict) else {}
        hybrid_blend = metadata.get("hybrid_blend") if isinstance(metadata.get("hybrid_blend"), dict) else {}
        self.last_split_dates = dict(split_dates)
        self.last_strict_oos_report = {}
        self.last_strict_oos_theta_comparison = {}

        model = load_tft_model(ckpt_path)
        ensemble_models = [model]
        raw_ensemble_names = metadata.get("ensemble_checkpoints") or []
        if isinstance(raw_ensemble_names, list):
            for name in raw_ensemble_names:
                if not name or str(name) == ckpt_path.name:
                    continue
                extra_ckpt = ckpt_path.parent / str(name)
                if not extra_ckpt.exists():
                    raise RuntimeError(
                        f"Missing required TFT ensemble checkpoint for backtest: {extra_ckpt}"
                    )
                try:
                    ensemble_models.append(load_tft_model(extra_ckpt))
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load TFT ensemble checkpoint {extra_ckpt}: {exc}"
                    ) from exc

        meta_test_metrics = metadata.get("test_metrics") if isinstance(metadata.get("test_metrics"), dict) else {}
        expected_ensemble_size = meta_test_metrics.get("ensemble_size")
        if expected_ensemble_size is not None and int(expected_ensemble_size) > len(ensemble_models):
            raise RuntimeError(
                f"Incomplete TFT ensemble for backtest: expected {int(expected_ensemble_size)} "
                f"checkpoints, loaded {len(ensemble_models)}"
            )

        dataset_params = _checkpoint_dataset_parameters(model)
        checkpoint_unknown, checkpoint_known = _checkpoint_feature_contract(dataset_params)
        encoder_length = int(dataset_params["max_encoder_length"])
        horizon = int(dataset_params["max_prediction_length"])
        quantiles = list(getattr(getattr(model, "loss", None), "quantiles", cfg.model.quantiles))

        required_features = list(checkpoint_unknown + checkpoint_known)
        prior_candidates: tuple[str, ...] = ()
        if hybrid_blend.get("enabled"):
            prior_feat = str(hybrid_blend.get("prior_feature", "HG=F_theta_5d_prior"))
            prior_candidates = (
                prior_feat,
                prior_feat.replace(".", "_").replace("-", "_"),
                prior_feat.replace(".", "_").replace("-", "_").replace("=", "_"),
            )

        df = self._build_eval_dataframe(cfg, required_features)

        oos_start_ts = pd.Timestamp(self.config.oos_start)
        oos_mask = df.index >= oos_start_ts
        if not oos_mask.any():
            raise RuntimeError(f"No rows available on/after oos_start={self.config.oos_start}")

        first_oos_pos = int(np.argmax(oos_mask))
        warmup_start = max(0, first_oos_pos - encoder_length)
        eval_df = df.iloc[warmup_start:].copy()
        eval_df["time_idx"] = np.arange(len(eval_df), dtype=int)

        if len(eval_df) < encoder_length + horizon:
            raise RuntimeError(
                f"Insufficient rows ({len(eval_df)}) for TFT backtest (need {encoder_length + horizon})"
            )

        ds = TimeSeriesDataSet.from_parameters(
            dataset_params,
            eval_df,
            predict=False,
            stop_randomization=True,
        )
        dl = ds.to_dataloader(train=False, batch_size=64, num_workers=0)

        pred_out = model.predict(dl, mode="quantiles", return_y=True)
        if hasattr(pred_out, "output"):
            pred_t = pred_out.output
            y_val = pred_out.y[0] if isinstance(pred_out.y, (list, tuple)) else pred_out.y
        elif isinstance(pred_out, (list, tuple)) and len(pred_out) == 2:
            pred_t, y_val = pred_out
            y_val = y_val[0] if isinstance(y_val, (list, tuple)) else y_val
        else:
            pred_t = pred_out
            y_val = None

        import inspect
        from deep_learning.training.metrics import (
            apply_direction_sign_multiplier_np,
            apply_weekly_interval_scale_np,
            apply_weekly_magnitude_scale_np,
            apply_weekly_median_cap_np,
            apply_weekly_sign_correction_np,
        )

        quantiles_tuple = tuple(quantiles)
        median_idx = quantiles.index(0.50) if 0.50 in quantiles else len(quantiles) // 2
        meta_config = metadata.get("config") if isinstance(metadata.get("config"), dict) else {}
        monotonic_transform = bool(
            meta_config.get("monotonic_quantile_transform")
            or metadata.get("monotonic_quantile_transform")
            or getattr(getattr(model, "loss", None), "__class__", type(None)).__name__
            == "WeeklyASROPFLoss"
        )

        def _member_to_ordered_quantiles(mdl, raw_arr: np.ndarray) -> np.ndarray:
            arr = np.asarray(raw_arr, dtype=np.float64)
            if getattr(getattr(mdl, "loss", None), "emits_ordered_quantiles", False):
                return arr
            if monotonic_transform or (
                arr.ndim >= 2 and arr.shape[-1] > 1 and np.any(np.diff(arr, axis=-1) < -1e-12)
            ):
                return monotonic_quantiles_np(arr, median_idx=median_idx)
            return arr

        pred_np = pred_t.detach().cpu().numpy() if hasattr(pred_t, "detach") else np.asarray(pred_t)
        ordered_members = [_member_to_ordered_quantiles(model, pred_np)]
        if len(ensemble_models) > 1:
            for extra_mdl in ensemble_models[1:]:
                extra_out = extra_mdl.predict(dl, mode="quantiles")
                extra_t = extra_out.output if hasattr(extra_out, "output") else extra_out
                extra_np = extra_t.detach().cpu().numpy() if hasattr(extra_t, "detach") else np.asarray(extra_t)
                ordered_members.append(_member_to_ordered_quantiles(extra_mdl, extra_np))
            pred_np = np.mean(np.stack(ordered_members, axis=0), axis=0)
            logger.info(
                "TFT backtest averaged %d ensemble checkpoints in ordered quantile space",
                len(ordered_members),
            )
        else:
            pred_np = ordered_members[0]

        if y_val is not None:
            y_np = y_val.detach().cpu().numpy() if hasattr(y_val, "detach") else np.asarray(y_val)
        else:
            import torch

            y_np = torch.cat(
                [b[1][0] if isinstance(b[1], (list, tuple)) else b[1] for b in dl]
            ).cpu().numpy()

        # Align decoded sample index -> forecast origin timestamp (`index_end - horizon`)
        origin_pos = np.clip(
            ds.index["index_end"].to_numpy(dtype=int) - horizon,
            0,
            len(eval_df) - 1,
        )
        anchor_dates = eval_df.index[origin_pos]

        if hybrid_blend.get("enabled"):
            from deep_learning.calibration.hybrid import apply_theta_hybrid_shift_to_path

            prior_col = next((c for c in prior_candidates if c in eval_df.columns), None)
            if prior_col is not None:
                theta_vals = eval_df[prior_col].to_numpy(dtype=float)[origin_pos]
                if "already_ordered" in inspect.signature(apply_theta_hybrid_shift_to_path).parameters:
                    pred_np = apply_theta_hybrid_shift_to_path(
                        pred_np,
                        theta_vals,
                        hybrid_blend=hybrid_blend,
                        quantiles=quantiles_tuple,
                        horizon=horizon,
                        already_ordered=True,
                    )
                else:
                    w_theta = float(np.clip(float(hybrid_blend.get("w_theta", 0.0) or 0.0), 0.0, 1.0))
                    w_tft = 1.0 - w_theta
                    h = min(int(horizon), pred_np.shape[1])
                    valid = np.isfinite(theta_vals)
                    if w_theta > 0.0 and valid.any() and h > 0:
                        tft_med = pred_np[:, :h, median_idx].sum(axis=1)
                        blended = np.where(valid, w_tft * tft_med + w_theta * theta_vals, tft_med)
                        delta_per_day = (blended - tft_med) / float(max(h, 1))
                        pred_np[:, :h, :] = pred_np[:, :h, :] + delta_per_day[:, None, None]

        foundation_prior = metadata.get("foundation_prior") or {}
        if foundation_prior.get("enabled"):
            try:
                from deep_learning.calibration.foundation_prior import (
                    CHRONOS2_MODEL_ID,
                    apply_foundation_prior_to_ordered_path,
                    predict_chronos2_weekly_log_quantiles,
                )

                if "target_close" in df.columns:
                    full_closes = pd.to_numeric(df["target_close"], errors="coerce").to_numpy(dtype=np.float64)
                else:
                    close_for_fnd = self._resolve_target_close_series(
                        cfg,
                        df,
                        last_known_price=float(getattr(self, "_last_known_price", float("nan"))),
                    )
                    full_closes = (
                        pd.to_numeric(
                            close_for_fnd.reindex(df.index, method="ffill").bfill(),
                            errors="coerce",
                        ).to_numpy(dtype=np.float64)
                        if close_for_fnd is not None and not close_for_fnd.empty
                        else np.empty(0, dtype=np.float64)
                    )
                if len(full_closes) == len(df) and len(origin_pos) > 0:
                    df_origin_pos = np.asarray(origin_pos, dtype=int) + int(warmup_start)
                    fnd_q = predict_chronos2_weekly_log_quantiles(
                        full_closes,
                        df_origin_pos,
                        horizon=horizon,
                        context_len=int(foundation_prior.get("context_len", 252)),
                        batch_size=32,
                        model_id=str(foundation_prior.get("model_id", CHRONOS2_MODEL_ID)),
                    )
                    if fnd_q is not None:
                        pred_np = apply_foundation_prior_to_ordered_path(
                            pred_np,
                            fnd_q,
                            foundation_prior=foundation_prior,
                            quantiles=quantiles_tuple,
                            horizon=horizon,
                        )
            except Exception as fnd_bt_exc:
                logger.warning("Backtest foundation prior skipped (fail-open): %s", fnd_bt_exc)

        # Apply persisted validation-fitted calibration stages in exact trainer/predictor order.
        direction_cal = metadata.get("direction_calibration") or {}
        dir_mult = int(direction_cal.get("direction_sign_multiplier", 1))
        if dir_mult != 1:
            pred_np = apply_direction_sign_multiplier_np(pred_np, dir_mult)
        weekly_sign_threshold = float(direction_cal.get("weekly_sign_threshold", 0.0) or 0.0)
        if abs(weekly_sign_threshold) > 1e-12:
            pred_np = apply_weekly_sign_correction_np(
                pred_np,
                weekly_sign_threshold,
                horizon=horizon,
            )

        weekly_direction_model = metadata.get("weekly_direction_model") or {}
        if weekly_direction_model.get("enabled"):
            from deep_learning.training.direction_model import (
                apply_weekly_direction_model,
                predict_weekly_direction,
            )

            origin_times = eval_df["time_idx"].to_numpy(dtype=int)[origin_pos]
            probs = np.concatenate(
                [
                    predict_weekly_direction(
                        weekly_direction_model,
                        eval_df,
                        start_exclusive=int(t) - 1,
                        end_inclusive=int(t),
                    )
                    for t in origin_times
                ]
            )
            pred_np = apply_weekly_direction_model(
                pred_np,
                probs,
                threshold=float(weekly_direction_model.get("decision_threshold", 0.50)),
                horizon=horizon,
            )

        magnitude_cal = metadata.get("magnitude_calibration") or {}
        mag_scale = float(
            direction_cal.get("weekly_magnitude_scale", magnitude_cal.get("scale", 1.0)) or 1.0
        )
        if abs(mag_scale - 1.0) > 1e-6:
            pred_np = apply_weekly_magnitude_scale_np(
                pred_np,
                mag_scale,
                quantiles=quantiles_tuple,
            )

        if len(pred_np) > 0 and "realized_vol_20d" in eval_df.columns:
            vols = eval_df["realized_vol_20d"].to_numpy(dtype=float)[origin_pos]
        else:
            vols = None

        interval_cal = metadata.get("interval_calibration") or {}
        interval_scale = float(interval_cal.get("weekly_interval_scale", 1.0) or 1.0)
        if interval_cal.get("weekly_interval_conditioning_enabled"):
            cond_feat = str(interval_cal.get("weekly_interval_conditioning_feature", "")).strip()
            if cond_feat not in eval_df.columns:
                raise RuntimeError(
                    f"Backtest missing required interval conditioning feature: {cond_feat}"
                )
            cond_vals = eval_df[cond_feat].to_numpy(dtype=float)[origin_pos]
            pred_np = apply_weekly_interval_scale_np(
                pred_np,
                interval_scale,
                quantiles=quantiles_tuple,
                conditioning_values=cond_vals,
                conditioning_reference=float(
                    interval_cal.get("weekly_interval_conditioning_reference", 0.0)
                ),
                conditioning_power=float(
                    interval_cal.get("weekly_interval_conditioning_power", 0.35)
                ),
                conditioning_min_factor=float(
                    interval_cal.get("weekly_interval_conditioning_min_factor", 0.5)
                ),
                conditioning_max_factor=float(
                    interval_cal.get("weekly_interval_conditioning_max_factor", 2.0)
                ),
                conditioning_min_scale=float(
                    interval_cal.get("weekly_interval_conditioning_min_scale", 0.20)
                ),
                conditioning_max_scale=float(
                    interval_cal.get("weekly_interval_conditioning_max_scale", 2.50)
                ),
            )
        elif abs(interval_scale - 1.0) > 1e-6:
            pred_np = apply_weekly_interval_scale_np(
                pred_np,
                interval_scale,
                quantiles=quantiles_tuple,
            )

        raw_cap = meta_config.get(
            "weekly_median_cap",
            (metadata.get("target_scale_audit") or {}).get(
                "weekly_median_cap",
                metadata.get("weekly_median_cap"),
            ),
        )
        if raw_cap is not None and float(raw_cap) > 0.0:
            pred_np, _ = apply_weekly_median_cap_np(
                pred_np,
                weekly_median_cap=float(raw_cap),
                quantiles=quantiles_tuple,
                horizon=horizon,
                already_ordered=True,
            )

        # Already in ordered quantile space -> aggregate RSS 5-day cumulative quantiles directly.
        weekly_q = aggregate_weekly_quantiles_np(
            pred_np[:, :horizon, :],
            median_idx=median_idx,
        )
        actual_weekly = y_np[:, :horizon].sum(axis=1)

        if cqr_artifact is not None:
            weekly_q = apply_weekly_cqr(
                weekly_q,
                quantiles=quantiles_tuple,
                cqr_params=cqr_artifact,
                realized_vol=vols,
            )

        # Restrict final scoring to OOS start onward (excluding warmup anchor overlaps if any)
        keep_mask = anchor_dates >= oos_start_ts
        if not keep_mask.any():
            raise RuntimeError("No TFT predictions fell inside OOS window")

        anchor_dates = anchor_dates[keep_mask]
        weekly_q = weekly_q[keep_mask]
        actual_weekly = actual_weekly[keep_mask]

        q_indices = {round(float(q), 2): i for i, q in enumerate(quantiles)}
        idx_q10 = q_indices.get(0.10, 1)
        idx_q50 = q_indices.get(0.50, len(quantiles) // 2)
        idx_q90 = q_indices.get(0.90, max(len(quantiles) - 2, 0))

        pred_weekly_med = weekly_q[:, idx_q50]
        pred_weekly_q10 = weekly_q[:, idx_q10]
        pred_weekly_q90 = weekly_q[:, idx_q90]

        close_series = self._resolve_target_close_series(
            cfg,
            df,
            last_known_price=float(getattr(self, "_last_known_price", float("nan"))),
        )
        anchor_norm = pd.to_datetime(anchor_dates).tz_localize(None).normalize()
        aligned_px = (
            close_series.reindex(anchor_norm, method="ffill").bfill()
            if close_series is not None and not close_series.empty
            else pd.Series(dtype=float)
        )
        predictions = []
        for idx, dt in enumerate(anchor_dates):
            px_val = aligned_px.iloc[idx] if idx < len(aligned_px) else 0.0
            px_now = float(px_val) if pd.notna(px_val) and np.isfinite(px_val) else 0.0
            p_ret = float(pred_weekly_med[idx])
            a_ret = float(actual_weekly[idx])
            predictions.append(
                {
                    "date": dt,
                    "y_current": px_now,
                    "y_pred": float(px_now * np.exp(p_ret)) if px_now > 0 else 0.0,
                    "y_actual": float(px_now * np.exp(a_ret)) if px_now > 0 else 0.0,
                    "pred_return": p_ret,
                    "actual_return": a_ret,
                    "q10_return": float(pred_weekly_q10[idx]),
                    "q90_return": float(pred_weekly_q90[idx]),
                }
            )

        pred_df = pd.DataFrame(predictions)
        metrics = _compute_backtest_metrics(
            pred_df,
            annualisation=52.0,
            use_hac=True,
            metric_space="cumulative_log_return_5d",
        )

        # Rolling window breakdown + Horizon-matched 5D cumulative Theta baseline
        wf = wf_backtest(
            actual_weekly,
            pred_weekly_med,
            y_pred_q10=pred_weekly_q10,
            y_pred_q90=pred_weekly_q90,
            window_size=min(50, max(15, len(pred_df) // 3)),
            step_size=10,
            annualisation=52.0,
            use_hac=True,
        )
        self.last_walkforward_report = wf if isinstance(wf, dict) else {}

        theta_input = close_series.dropna() if close_series is not None else pd.Series(dtype=float)
        if len(theta_input) >= 80:
            eval_dates = pd.DatetimeIndex(pred_df["date"]) if not pred_df.empty else None
            theta_res = theta_backtest(
                theta_input,
                horizon=horizon,
                n_windows=len(pred_df),
                return_mode="cumulative",
                use_log_return=True,
                annualisation=52.0,
                eval_dates=eval_dates,
            )
            if "error" not in theta_res:
                full_oos_for_baseline = {
                    "summary": {
                        "mean_da": metrics.directional_accuracy,
                        "mean_sharpe": metrics.sharpe_ratio,
                        "mean_mae": metrics.mae,
                    }
                }
                comp = compare_with_baseline(full_oos_for_baseline, theta_res)
                comp["theta"] = theta_res
                self.last_theta_comparison = comp
            else:
                self.last_theta_comparison = {}
        else:
            self.last_theta_comparison = {}

        val_end_date = split_dates.get("val_end_date") if isinstance(split_dates, dict) else None
        if val_end_date and not pred_df.empty:
            try:
                val_end_ts = pd.Timestamp(val_end_date).tz_localize(None).normalize()
                pred_dates_norm = pd.to_datetime(pred_df["date"]).dt.tz_localize(None).dt.normalize()
                strict_df = pred_df.loc[pred_dates_norm > val_end_ts].copy()
                if len(strict_df) >= 15:
                    strict_metrics = _compute_backtest_metrics(
                        strict_df,
                        annualisation=52.0,
                        use_hac=True,
                        metric_space="cumulative_log_return_5d",
                    )
                    strict_dates = pd.DatetimeIndex(strict_df["date"])
                    self.last_strict_oos_report = {
                        "n_predictions": int(len(strict_df)),
                        "overlap_excluded_windows": int(len(pred_df) - len(strict_df)),
                        "val_end_date": str(val_end_date),
                        "start_date": strict_dates.min().date().isoformat(),
                        "end_date": strict_dates.max().date().isoformat(),
                        "metrics": asdict(strict_metrics),
                    }
                    if len(theta_input) >= 80:
                        strict_theta = theta_backtest(
                            theta_input,
                            horizon=horizon,
                            n_windows=len(strict_df),
                            return_mode="cumulative",
                            use_log_return=True,
                            annualisation=52.0,
                            eval_dates=strict_dates,
                        )
                        if "error" not in strict_theta:
                            strict_base = {
                                "summary": {
                                    "mean_da": strict_metrics.directional_accuracy,
                                    "mean_sharpe": strict_metrics.sharpe_ratio,
                                    "mean_mae": strict_metrics.mae,
                                }
                            }
                            strict_comp = compare_with_baseline(strict_base, strict_theta)
                            strict_comp["theta"] = strict_theta
                            self.last_strict_oos_theta_comparison = strict_comp
            except Exception as strict_exc:
                logger.debug("Strict OOS segmentation skipped: %s", strict_exc)

        return metrics, pred_df

    def compare_with_xgboost(
        self,
        symbol_set: SymbolSet,
        xgb_metrics: Optional[BacktestMetrics] = None,
    ) -> dict:
        """
        Evaluate XGBoost (1D diagnostic) and promoted TFT-ASRO (5D primary) separately.

        Note: XGBoost operates on T+1 simple returns while TFT-ASRO operates on 5-day
        cumulative log returns; ``delta_mae_pct`` therefore compares the scale-invariant
        ``mae_vs_naive_zero`` skill ratio across horizons.
        """
        if xgb_metrics is None:
            xgb_runner = BacktestRunner(self.config)
            logger.info("=== XGBoost 1D Diagnostic Backtest ===")
            xgb_metrics, _xgb_preds = xgb_runner.run_backtest(symbol_set.symbols)

        logger.info("=== TFT-ASRO 5D Primary Walk-Forward Backtest ===")
        tft_metrics, _tft_preds = self.run_backtest(symbol_set.symbols)

        delta_mae = (
            (tft_metrics.mae_vs_naive_zero - xgb_metrics.mae_vs_naive_zero)
            / max(xgb_metrics.mae_vs_naive_zero, 1e-12)
        ) * 100.0
        delta_dir = (
            (tft_metrics.directional_accuracy - xgb_metrics.directional_accuracy)
            / max(xgb_metrics.directional_accuracy, 1e-9)
        ) * 100.0

        theta_comp = dict(self.last_theta_comparison)
        verdict = theta_comp.get("verdict")
        if not verdict:
            verdict = "TFT_SUPERIOR" if tft_metrics.mae_vs_naive_zero < 1.0 and tft_metrics.sharpe_ratio > 0 else "MIXED"

        return {
            "run_id": self.run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "symbol_set": asdict(symbol_set),
            "horizon_note": (
                "XGBoost evaluates T+1 simple returns (sqrt(252)); TFT-ASRO and Theta "
                "evaluate 5-day cumulative log returns (sqrt(52) HAC)."
            ),
            "xgboost": asdict(xgb_metrics),
            "tft_asro": asdict(tft_metrics),
            "tft_walkforward_summary": self.last_walkforward_report.get("summary", {}),
            "windows": self.last_walkforward_report.get("windows", []),
            "theta_comparison": theta_comp,
            "split_dates": self.last_split_dates,
            "strict_oos": self.last_strict_oos_report,
            "strict_oos_theta_comparison": self.last_strict_oos_theta_comparison,
            "verdict": verdict,
            "delta_mae_pct": round(delta_mae, 2),
            "delta_mae_basis": "mae_vs_naive_zero",
            "delta_dir_acc_pct": round(delta_dir, 2),
            "tft_better_mae": bool(tft_metrics.mae_vs_naive_zero < xgb_metrics.mae_vs_naive_zero),
            "tft_better_dir": bool(delta_dir > 0),
        }


# ---------------------------------------------------------------------------
# DB persistence helpers
# ---------------------------------------------------------------------------


def _persist_report_to_db(result: BacktestResult, kind: str = "xgb_challenger") -> None:
    """Write a BacktestResult into the backtest_reports table."""
    from app.db import SessionLocal
    from app.models import BacktestReport

    summary = {
        "champion": result.champion.get("metrics", {}),
        "challenger": result.challenger.get("metrics", {}),
        "decision": result.decision,
        "decision_reason": result.decision_reason,
        "delta_mae_pct": result.delta_mae_pct,
        "delta_rmse_pct": result.delta_rmse_pct,
        "delta_dir_acc_pct": result.delta_dir_acc_pct,
        "improvement_mae_pct": result.improvement_mae_pct,
        "ci95_improvement_mae_pct": result.ci95_improvement_mae_pct,
    }

    safe_summary = _to_json_safe(summary)
    safe_raw = _to_json_safe(asdict(result))

    with SessionLocal() as session:
        row = BacktestReport(
            symbol="HG=F",
            run_id=result.run_id,
            report_kind=kind,
            summary_json=safe_summary,
            windows_json=[],
            theta_comparison_json={},
            raw_json=safe_raw,
            verdict=result.decision,
        )
        session.add(row)
        session.commit()
        logger.info("Backtest report persisted to DB: id=%s", row.id)


def _persist_tft_report_to_db(tft_comparison: dict) -> None:
    """Write authentic TFT walk-forward & Theta baseline comparison into backtest_reports."""
    from app.db import SessionLocal
    from app.models import BacktestReport

    tft_asro = tft_comparison.get("tft_asro") or {}
    wf_summary = tft_comparison.get("tft_walkforward_summary") or {}
    windows = _to_json_safe(tft_comparison.get("windows") or [])

    summary = _to_json_safe(
        {
            "directional_accuracy": tft_asro.get("directional_accuracy"),
            "sharpe_ratio": tft_asro.get("sharpe_ratio"),
            "mae": tft_asro.get("mae"),
            "rmse": tft_asro.get("rmse"),
            "variance_ratio": wf_summary.get("mean_vr"),
            "mae_vs_naive_zero": tft_asro.get("mae_vs_naive_zero"),
            "mean_da": wf_summary.get("mean_da", tft_asro.get("directional_accuracy")),
            "mean_sharpe": wf_summary.get("mean_sharpe", tft_asro.get("sharpe_ratio")),
            "mean_mae": wf_summary.get("mean_mae", tft_asro.get("mae")),
            "mean_vr": wf_summary.get("mean_vr"),
            "mean_pi80_coverage": wf_summary.get("mean_pi80_coverage"),
            "n_windows": wf_summary.get("n_windows", len(windows)),
            "total_samples": wf_summary.get("total_samples", tft_asro.get("n_predictions")),
            "xgboost": tft_comparison.get("xgboost", {}),
            "tft_asro": tft_asro,
            "tft_walkforward": wf_summary,
            "horizon_note": tft_comparison.get("horizon_note"),
            "delta_mae_pct": tft_comparison.get("delta_mae_pct"),
            "delta_mae_basis": tft_comparison.get("delta_mae_basis"),
            "delta_dir_acc_pct": tft_comparison.get("delta_dir_acc_pct"),
            "tft_better_mae": tft_comparison.get("tft_better_mae"),
            "tft_better_dir": tft_comparison.get("tft_better_dir"),
            "split_dates": tft_comparison.get("split_dates") or {},
            "strict_oos": tft_comparison.get("strict_oos") or {},
            "strict_oos_theta_comparison": tft_comparison.get("strict_oos_theta_comparison") or {},
        }
    )
    theta_comp = _to_json_safe(tft_comparison.get("theta_comparison") or {})
    raw_payload = _to_json_safe(tft_comparison)
    verdict = str(tft_comparison.get("verdict") or "MIXED")

    with SessionLocal() as session:
        row = BacktestReport(
            symbol="HG=F",
            run_id=tft_comparison.get("run_id"),
            report_kind="tft_vs_xgb",
            summary_json=summary,
            windows_json=windows,
            theta_comparison_json=theta_comp,
            raw_json=raw_payload,
            verdict=verdict,
        )
        session.add(row)
        session.commit()
        logger.info("TFT backtest report persisted to DB: id=%s (windows=%d)", row.id, len(windows))


def main():
    parser = argparse.ArgumentParser(description="Champion/Challenger Backtest")
    parser.add_argument("--champion", required=True, help="Path to champion symbol set JSON")
    parser.add_argument("--challenger", required=True, help="Path to challenger symbol set JSON")
    parser.add_argument("--output-dir", default="backend/artifacts/backtests", help="Output directory")
    parser.add_argument("--oos-start", default="", help="OOS start date (YYYY-MM-DD, empty = 365d lookback)")
    parser.add_argument("--oos-end", default="", help="OOS end date (YYYY-MM-DD, empty = today UTC)")
    parser.add_argument("--include-tft", action="store_true", help="Include authentic TFT-ASRO walk-forward evaluation")
    args = parser.parse_args()

    champion = load_symbol_set(args.champion)
    challenger = load_symbol_set(args.challenger)
    config = BacktestConfig(oos_start=args.oos_start, oos_end=args.oos_end)

    runner = BacktestRunner(config)
    result, predictions = runner.compare(champion, challenger)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_path = output_dir / f"{result.run_id}_report.json"
    report_path.write_text(json.dumps(_to_json_safe(asdict(result)), indent=2))
    logger.info("Report saved: %s", report_path)

    try:
        _persist_report_to_db(result, kind="xgb_challenger")
    except Exception as db_err:
        logger.warning("Could not persist backtest report to DB: %s", db_err)

    preds_path = output_dir / f"{result.run_id}_predictions.csv"
    predictions.to_csv(preds_path, index=False)
    logger.info("Predictions saved: %s", preds_path)

    print("\n" + "=" * 60)
    print(f"BACKTEST RESULT: {result.decision}")
    print("=" * 60)
    print(f"OOS Window:      {config.oos_start} -> {config.oos_end}")
    print(f"Champion MAE:    {result.champion['metrics']['mae']:.6f} (vs naive0={result.champion['metrics'].get('mae_vs_naive_zero', 1.0):.4f})")
    print(f"Challenger MAE:  {result.challenger['metrics']['mae']:.6f} (vs naive0={result.challenger['metrics'].get('mae_vs_naive_zero', 1.0):.4f})")
    print(f"Improvement:     {result.improvement_mae_pct:+.2f}% (95% CI {result.ci95_improvement_mae_pct})")
    print(f"Decision:        {result.decision}")
    print(f"Reason:          {result.decision_reason}")
    print("=" * 60)

    if getattr(args, "include_tft", False):
        print("\n" + "=" * 60)
        print("TFT-ASRO 5D PRIMARY WALK-FORWARD EVALUATION")
        print("=" * 60)
        try:
            tft_runner = TFTBacktestRunner(config)
            champ_metrics = (
                BacktestMetrics(**result.champion["metrics"])
                if isinstance(result.champion, dict) and isinstance(result.champion.get("metrics"), dict)
                else None
            )
            tft_comparison = tft_runner.compare_with_xgboost(champion, xgb_metrics=champ_metrics)

            tft_report_path = output_dir / f"{tft_comparison['run_id']}_tft_report.json"
            tft_report_path.write_text(json.dumps(_to_json_safe(tft_comparison), indent=2))

            try:
                _persist_tft_report_to_db(tft_comparison)
            except Exception as db_err:
                print(f"::warning::Could not persist TFT backtest report to DB: {db_err}")

            print(f"TFT 5D MAE:      {tft_comparison['tft_asro']['mae']:.6f} (vs naive0={tft_comparison['tft_asro'].get('mae_vs_naive_zero', 1.0):.4f})")
            print(f"TFT 5D DA:       {tft_comparison['tft_asro']['directional_accuracy']:.4f}")
            print(f"TFT 5D Sharpe:   {tft_comparison['tft_asro']['sharpe_ratio']:.4f}")
            print(f"Theta Verdict:   {tft_comparison.get('verdict')} ({len(tft_comparison.get('windows', []))} windows)")
            print("=" * 60)
        except Exception as exc:
            print(f"::warning::TFT backtest skipped/unavailable: {exc}")
            print("=" * 60)


if __name__ == "__main__":
    main()
