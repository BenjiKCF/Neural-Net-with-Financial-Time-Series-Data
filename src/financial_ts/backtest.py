"""Delayed-close, long/cash SPY exposure simulations with explicit trading costs.

Signals observed after session t trade at the next observed session's close and
earn the following close-to-close adjusted-price return. The supplied price
index is the session calendar; it must be complete. This module never downloads
prices, fills missing observations, tunes thresholds, or interprets SPX as a
tradable security. Adjusted SPY closes are a total-return proxy, not fill quotes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import numpy as np
import pandas as pd

DEFAULT_COSTS_BPS = (0.0, 5.0, 10.0, 20.0)
CONTROLS = ("cash", "buy_and_hold")


def _daily_dates(index, name: str) -> pd.DatetimeIndex:
    if pd.api.types.is_numeric_dtype(index.dtype):
        raise ValueError(f"{name} must contain daily dates, not numeric offsets.")
    try:
        dates = pd.DatetimeIndex(index)
    except (ValueError, TypeError) as error:
        raise ValueError(f"{name} must contain valid daily dates.") from error
    if (
        dates.empty
        or dates.hasnans
        or dates.tz is not None
        or not dates.equals(dates.normalize())
        or not dates.is_unique
        or not dates.is_monotonic_increasing
    ):
        raise ValueError(f"{name} must be unique, sorted, timezone-naive daily dates at midnight.")
    return dates


def _inputs(predictions, adjusted_close):
    if not isinstance(predictions, pd.DataFrame) or predictions.empty:
        raise ValueError("predictions must be a nonempty DataFrame of model scores.")
    if (
        not predictions.columns.is_unique
        or any(not isinstance(name, str) or not name.strip() for name in predictions.columns)
        or set(predictions.columns) & set(CONTROLS)
    ):
        raise ValueError(
            "Model columns must be unique nonempty names; cash/buy_and_hold are reserved."
        )
    index = predictions.index
    instrument = None
    if isinstance(index, pd.MultiIndex):
        date_names = set(index.names) & {"date", "datetime"}
        if index.nlevels != 2 or len(date_names) != 1 or "instrument" not in index.names:
            raise ValueError("A MultiIndex must have date/datetime and instrument levels.")
        instruments = index.get_level_values("instrument")
        if instruments.hasnans or len(instruments.unique()) != 1:
            raise ValueError("The backtest requires exactly one signal instrument.")
        instrument = str(instruments[0])
        index = index.get_level_values(next(iter(date_names)))
    dates = _daily_dates(index, "Prediction index")
    if not isinstance(adjusted_close, pd.Series):
        raise ValueError("adjusted_close must be a Series indexed by every observed session.")
    calendar = _daily_dates(adjusted_close.index, "Price index")
    try:
        scores = predictions.to_numpy(dtype=np.float64)
        prices = adjusted_close.to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError("Forecast scores and adjusted closes must be numeric.") from error
    if not np.isfinite(scores).all():
        raise ValueError("Every model must have one finite score for every prediction date.")
    if not np.isfinite(prices).all() or (prices <= 0).any():
        raise ValueError("Every adjusted close must be finite and positive; no prices are filled.")
    locations = calendar.get_indexer(dates)
    if (locations < 0).any() or (np.diff(locations) != 1).any():
        raise ValueError(
            "Prediction dates must be consecutive observed sessions in the price index."
        )
    if locations[-1] + 2 >= len(calendar):
        raise ValueError("Two subsequent observed sessions are required after the last signal.")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        asset_returns = prices[locations + 2] / prices[locations + 1] - 1
    if not np.isfinite(asset_returns).all() or (asset_returns <= -1).any():
        raise ValueError("Adjusted prices must produce finite returns strictly above -100%.")
    return (
        dates,
        calendar[locations + 1],
        calendar[locations + 2],
        scores,
        asset_returns,
        instrument,
    )


def _costs(costs_bps: Sequence[float]) -> tuple[float, ...]:
    try:
        values = tuple(costs_bps)
    except TypeError as error:
        raise ValueError(
            "costs_bps must be a nonempty sequence of one-way basis-point costs."
        ) from error
    if not values or any(
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or not 0 <= value < 10000
        for value in values
    ):
        raise ValueError("One-way costs must be finite numbers in [0, 10000) basis points.")
    values = tuple(float(value) for value in values)
    if len(set(values)) != len(values):
        raise ValueError("Cost scenarios must be unique.")
    return values


def _ledger(model, scores, dates, entries, exits, asset_returns, cost_bps):
    rate = cost_bps / 10000
    equity = gross_equity = peak = 1.0
    previous_position = 0
    rows = []
    for i, (score, asset_return) in enumerate(zip(scores, asset_returns)):
        position = int(score > 0)
        entry_trade = int(position > previous_position)
        exit_trade = int(position < previous_position)
        liquidation_trade = int(i == len(scores) - 1 and position == 1)
        starting_equity = equity
        # A full purchase spends notional * (1 + rate), so it cannot borrow
        # cash to pay its fee. A full sale pays rate * sale notional.
        traded_notional = (
            starting_equity / (1 + rate) if entry_trade else starting_equity if exit_trade else 0.0
        )
        trade_cost = traded_notional * rate
        cost_factor = 1 / (1 + rate) if entry_trade else 1 - rate if exit_trade else 1.0
        gross_return = position * asset_return
        before_liquidation = starting_equity * cost_factor * (1 + gross_return)
        liquidation_cost = before_liquidation * rate if liquidation_trade else 0.0
        liquidation_factor = 1 - rate if liquidation_trade else 1.0
        equity = before_liquidation * liquidation_factor
        gross_equity *= 1 + gross_return
        if not math.isfinite(equity) or equity <= 0 or not math.isfinite(gross_equity):
            raise ValueError("Portfolio compounding overflowed or produced nonpositive equity.")
        peak = max(peak, equity)
        rows.append(
            {
                "model": model,
                "cost_bps": cost_bps,
                "signal_date": dates[i],
                "entry_date": entries[i],
                "exit_date": exits[i],
                "score": float(score),
                "previous_position": previous_position,
                "position": position,
                "asset_return": float(asset_return),
                "gross_return": float(gross_return),
                "entry_trade": entry_trade,
                "exit_trade": exit_trade,
                "liquidation_trade": liquidation_trade,
                "exposure_turnover": entry_trade + exit_trade + liquidation_trade,
                "traded_notional": traded_notional,
                "liquidation_notional": before_liquidation if liquidation_trade else 0.0,
                "trade_cost": trade_cost,
                "liquidation_cost": liquidation_cost,
                "cost_factor": cost_factor,
                "liquidation_factor": liquidation_factor,
                "starting_equity": starting_equity,
                "net_return": equity / starting_equity - 1,
                "gross_equity": gross_equity,
                "net_equity": equity,
                "drawdown": equity / peak - 1,
            }
        )
        previous_position = position
    return pd.DataFrame(rows)


def _metrics(ledger):
    returns = ledger["net_return"].to_numpy()
    elapsed_days = (ledger["exit_date"].iloc[-1] - ledger["entry_date"].iloc[0]).days
    final_equity = float(ledger["net_equity"].iloc[-1])
    try:
        cagr = math.expm1(math.log(final_equity) * 365.25 / elapsed_days)
    except OverflowError as error:
        raise ValueError("Annualized return overflowed for the supplied prices.") from error
    deviation = float(np.std(returns, ddof=1)) if len(returns) > 1 else None
    annualized_volatility = deviation * math.sqrt(252) if deviation is not None else None
    sharpe = (
        float(np.mean(returns)) / deviation * math.sqrt(252)
        if deviation is not None and deviation > 0
        else None
    )
    entries = int(ledger["entry_trade"].sum())
    exits = int(ledger["exit_trade"].sum() + ledger["liquidation_trade"].sum())
    return {
        "model": ledger["model"].iloc[0],
        "cost_bps": float(ledger["cost_bps"].iloc[0]),
        "periods": len(ledger),
        "signal_start": ledger["signal_date"].iloc[0].strftime("%Y-%m-%d"),
        "signal_end": ledger["signal_date"].iloc[-1].strftime("%Y-%m-%d"),
        "entry_start": ledger["entry_date"].iloc[0].strftime("%Y-%m-%d"),
        "exit_end": ledger["exit_date"].iloc[-1].strftime("%Y-%m-%d"),
        "elapsed_days": elapsed_days,
        "ending_equity": final_equity,
        "total_return": final_equity - 1,
        "gross_total_return": float(ledger["gross_equity"].iloc[-1]) - 1,
        "cagr": cagr,
        "annualized_volatility": annualized_volatility,
        "sharpe_zero_cash": sharpe,
        "max_drawdown": float(ledger["drawdown"].min()),
        "mean_exposure": float(ledger["position"].mean()),
        "entries": entries,
        "exits": exits,
        "trade_count": entries + exits,
        "completed_round_trips": entries,
        "exposure_turnover": float(ledger["exposure_turnover"].sum()),
        "fees_paid_per_initial_equity": float(
            ledger["trade_cost"].sum() + ledger["liquidation_cost"].sum()
        ),
    }


def run_backtest(
    predictions: pd.DataFrame,
    adjusted_close: pd.Series,
    *,
    costs_bps: Sequence[float] = DEFAULT_COSTS_BPS,
) -> dict:
    """Compare fixed-sign long/cash strategies over an identical observed period.

    ``predictions`` has one score column per model and a daily date index, or
    a date/datetime + instrument MultiIndex containing one signal instrument.
    ``adjusted_close`` supplies complete observed sessions including the signal
    dates and two subsequent sessions. The caller identifies these prices as
    SPY and preserves their data-source provenance. No alignment gaps are filled.

    Return ``ledger`` and ``metrics`` DataFrames plus JSON-ready ``settings``.
    The ledger is long-form, with one row for each model/cost/signal. Equity
    starts at 1 in cash. Cash and buy-and-hold controls are included. Sharpe uses
    zero cash yield and 252 sessions; undefined Sharpe/volatility are None.
    Maximum drawdown is a nonpositive fraction, including the initial NAV.
    Exposure turnover counts changes in binary exposure, including liquidation;
    actual fee notionals and fees are recorded separately in initial-NAV units.
    """
    scenarios = _costs(costs_bps)
    dates, entries, exits, scores, asset_returns, instrument = _inputs(predictions, adjusted_close)
    models = {name: scores[:, i] for i, name in enumerate(predictions.columns)}
    models.update(cash=np.zeros(len(dates)), buy_and_hold=np.ones(len(dates)))
    ledgers, metrics = [], []
    for model, model_scores in models.items():
        for cost in scenarios:
            frame = _ledger(model, model_scores, dates, entries, exits, asset_returns, cost)
            ledgers.append(frame)
            metrics.append(_metrics(frame))
    metrics_frame = pd.DataFrame(metrics)
    for column in ("sharpe_zero_cash", "annualized_volatility"):
        metrics_frame[column] = (
            metrics_frame[column].astype(object).where(metrics_frame[column].notna(), None)
        )
    return {
        "ledger": pd.concat(ledgers, ignore_index=True),
        "metrics": metrics_frame,
        "settings": {
            "policy": "long if score > 0, otherwise cash; fixed for every model",
            "signal_instrument": instrument,
            "execution_asset": "SPY",
            "price_basis": "supplied dividend/split-adjusted close; total-return proxy",
            "signal_time": "after close t",
            "entry_time": "close of next observed session t+1",
            "exit_time": "close of following observed session t+2",
            "session_calendar": "supplied price index; completeness is a data-source requirement",
            "costs_bps": list(scenarios),
            "primary_cost_bps": 5.0 if 5.0 in scenarios else None,
            "cost_assumption": "one-way proportional cost on traded notional, not a quoted fee",
            "initial_equity": 1.0,
            "initial_position": "cash",
            "terminal_liquidation": True,
            "cash_return": 0.0,
            "shorting": False,
            "leverage": False,
            "taxes": "not modeled",
            "annualization": "CAGR uses actual elapsed days/365.25; volatility/Sharpe use 252",
            "threshold_selection": "none; threshold fixed at zero",
        },
    }
