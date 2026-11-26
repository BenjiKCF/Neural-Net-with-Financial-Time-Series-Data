# Modernization verification — 27 November 2026

The supported Qlib workflow was executed on Linux with Python 3.12 and CPU PyTorch.
The machine-readable evidence is in [modernization_validation.json](../reports/modernization_validation.json).

## Software verification

- 60 automated tests passed, including actual Qlib ingestion, training, and backtesting.
- LightGBM and the Transformer each generated 576 finite, aligned predictions on the offline synthetic demo.
- Both synthetic portfolios completed with explicit transaction-cost deductions.
- A separate integration test verified the zero-return cash benchmark.
- Reloading the saved Transformer and fitted dataset reproduced all 576 scores exactly, with maximum absolute difference 0.0 and without refitting preprocessing.
- Changing only future prices/volume left all training features and fitted scaler statistics exactly unchanged.
- Lint checks passed.

Synthetic return numbers are excluded here: the synthetic runs check software behavior, not investment performance.

## Existing S&P 500 data

Both models were trained using the bundled `^GSPC.csv`: 7,252 observations from 2 January 1990 through 10 October 2018. They used the same chronological split, raw delayed-return target, and 157 OHLCV features. The Transformer additionally used 20-session historical windows.

There were **1,438 test predictions per model**. RMSE is in decimal return units; lower is better.

| Forecast | Test RMSE | Test MAE | Directional accuracy |
| --- | ---: | ---: | ---: |
| Zero return | 0.00773293 | — | — |
| LightGBM | 0.00773053 | 0.00544931 | 54.24% |
| Temporal Transformer | 0.00779371 | 0.00553373 | 49.72% |

LightGBM is essentially tied with the zero-return forecast on this holdout; the Transformer is worse. Directional accuracy alone does not establish useful prediction or profitability, and a constant-direction baseline is not evaluated in this table. These results do not establish statistical significance or current-market generalization. No portfolio backtest was attached to this index example because the index is not directly tradable.

No model settings were tuned to improve these test numbers. The default configurations were run once, with model selection/early stopping restricted to validation. Modern software and valid evaluation are the result of this migration; profitable performance remains an empirical research question.

## Reproduction

Follow the installation instructions in the README, then use new output directories:

```bash
python -m pytest -q
fts demo --model both --epochs 2 --output runs/recheck-demo
fts import-csv --csv "^GSPC.csv" --symbol SPX --provider .data/recheck-spx
fts train --config configs/lightgbm.yaml --provider .data/recheck-spx --output runs/recheck-lightgbm --no-backtest
fts train --config configs/transformer.yaml --provider .data/recheck-spx --output runs/recheck-transformer --device cpu --no-backtest
```

The JSON evidence includes source hashes, exact package versions, model settings, and date ranges. GPU behavior and current market data have not been tested by this verification run. Normal third-party deprecation/backtest warnings were present; all asserted checks completed.
