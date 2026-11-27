# Getting started: your first successful run

This guide installs the project, checks all eight trainable models on generated
example data, then runs the real-data CSI300 benchmark using the latest verified
chenditc Qlib release. You need internet access for installation and the market
dataset download. The initial demo needs **no API keys, market data, or GPU**.
Copy commands into Terminal on macOS/Linux or PowerShell on Windows, one line at
a time. These are terminal commands, not Python statements to enter after `>>>`.

## 1. Install Python 3.12

Use **Python 3.12**. This project supports Python 3.10–3.12, so installing the
newest Python release automatically can give you an unsupported version.

- **Windows:** install the [Python install manager](https://www.python.org/downloads/),
  following [Python's Windows instructions](https://docs.python.org/3/using/windows.html).
  Open PowerShell and run `py install 3.12`, then `py -3.12 --version`.
- **macOS:** install [Homebrew](https://brew.sh/) if needed, then run
  `brew install python@3.12` followed by `python3.12 --version`.
  See the [Python 3.12 formula](https://formulae.brew.sh/formula/python@3.12) and
  [official macOS installation guide](https://docs.python.org/3.12/using/mac.html).
- **Linux:** install Python 3.12 and its virtual-environment package through your
  distribution. On Ubuntu 24.04, use `sudo apt update`, followed by
  `sudo apt install python3.12 python3.12-venv`. Check `python3.12 --version`.
  Other distributions have different package commands; see
  [Python's Linux guidance](https://docs.python.org/3.12/using/unix.html).

The version check should print `Python 3.12.x`.

## 2. Download the project

**Git is optional.** Download the [master branch ZIP](https://github.com/BenjiKCF/Neural-Net-with-Financial-Time-Series-Data/archive/refs/heads/master.zip),
extract it completely, and open the extracted folder. Do not work inside the ZIP.

Open your terminal in the folder containing `pyproject.toml`, `README.md`,
`configs`, and `src`. This is the **repository root**. On Windows, open that folder
in File Explorer, type `powershell` into its address bar, and press Enter.
On macOS, type `cd ` in Terminal, drag the extracted folder into the window,
and press Enter. On Linux, use your file manager's “Open in Terminal” action.

If Git is already installed, you can use these commands instead:

```bash
git clone https://github.com/BenjiKCF/Neural-Net-with-Financial-Time-Series-Data.git
cd Neural-Net-with-Financial-Time-Series-Data
```

The default branch is named `master`. Keep running the remaining commands from
the repository root, including after opening a new terminal.

## 3. Create an isolated Python environment

A virtual environment keeps this project's packages together in `.venv`.

**macOS/Linux:**

```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

**Windows PowerShell:**

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Now check the interpreter:

```bash
python --version
python -c "import sys; print(sys.executable)"
```

Expect Python 3.12 and a path containing this project's `.venv`. Your prompt may
also show `(.venv)`. Repeat the activation command whenever you open a new terminal.
If PowerShell blocks activation, use the alternative in the troubleshooting table.

## 4. Install the project and CPU PyTorch

First update the installer:

```bash
python -m pip install --upgrade pip
```

**Windows/Linux:** install the CPU build of PyTorch:

```bash
python -m pip install "torch>=2.5,<3" --index-url https://download.pytorch.org/whl/cpu
```

**macOS:** install LightGBM's OpenMP runtime, then the standard macOS PyTorch build:

```bash
brew install libomp
python -m pip install "torch>=2.5,<3"
```

[LightGBM's installation instructions](https://pypi.org/project/lightgbm/4.7.0/)
require this OpenMP library. Use CPU here; this project's CLI does not expose
Apple's MPS backend. [Qlib 0.9.7](https://pypi.org/project/pyqlib/0.9.7/#files)
provides a prebuilt macOS package for Python 3.12, so the normal install should
not require compiling Qlib or separately installing Cython. These macOS steps
follow upstream requirements; the repository's executed validation used Linux CPU.

Then, on **every platform**, install the project from the folder you downloaded:

```bash
python -m pip install -e ".[neural,visualization]"
fts models
```

The dot means “this folder”; this command does not depend on the project being
published on PyPI. The `neural` extra installs the trainable models;
`visualization` adds Qlib's native charts and PNG/HTML export.
Keep the quotes around `.[neural,visualization]`. For platform-specific PyTorch requirements,
use the [official installation selector](https://pytorch.org/get-started/locally/).

## 5. Run the quick offline check

```bash
fts demo --model all --epochs 1 --device cpu --output runs/my-first-demo
```

Leave the terminal open until the command finishes. It generates synthetic
prices, trains LightGBM, the Transformer, four mini models, and two new financial
architectures, then exports predictions and simulated portfolio results.
Runtime depends on your computer.

Open `runs/my-first-demo` in your file manager:

| File or folder | What to look at |
| --- | --- |
| `comparison.json` | Forecast metrics for all eight models. |
| Each model's `predictions.csv` | Dates, instruments, predictions, and observed targets. |
| Each model's `metrics.json` | Forecast errors and portfolio statistics. |
| Each model's `portfolio.csv` | Simulated daily returns and transaction costs. |
| `synthetic.csv` and `provider/` | Generated input data and its Qlib representation. |

JSON files open in a text editor; CSV files open in a spreadsheet. Completing
this check means the software runs. Small errors or positive simulated returns
on generated data do **not** establish useful real-market predictions.

To try one model, change `--model all` to `deepseek_v41_mini`, `glm54_mini`,
`qwen4_mini`, or `timesfm3_mini`, and choose a new output folder. These small models
train from random initialization; they do not contain the flagship pretrained weights.

## 6. Download the latest chenditc data and rerun the models

Download and verify the real daily market dataset:

```bash
fts download-chenditc --provider .data/cn --release latest
```

The command creates a **provider**: the local data folder Qlib reads. It resolves
the latest release once, downloads the archive and its matching provenance
manifest, verifies the bytes, and writes `.data/cn/metadata.json`. That file
states the actual release date and the last observed market date. A future
trading calendar in the archive is not treated as future prices.

The checked 26 November 2026 archive is approximately **573 MB compressed**;
extraction and model outputs require additional space. The archive contains a
broad China-market universe, while this experiment uses **historical CSI300
membership**. This is a different dataset and task from the old SPX example.
See the [dataset and protocol guide](chenditc-benchmark.md).

Run the eight trainable models:

```bash
fts benchmark-cn --provider .data/cn --output runs/csi300
```

Leave the terminal open while it runs. This trains on all eligible training
rows, evaluates a full validation/test panel, and backtests every model using
the same Top50/Drop5 strategy. It is substantially longer than the offline
demo. The default uses CPU, seed **42**, and at most **six neural epochs**.

| Open this file | What it tells you |
| --- | --- |
| `runs/csi300/protocol.json` | Exact data release, split dates, model roster, budgets, and selection rule |
| `runs/csi300/comparison.csv` | Completed models' forecast and portfolio results |
| `runs/csi300/comparison.json` | Results plus any models still pending |
| `runs/csi300/lightgbm/run.json` | Whether that model finished and which source it used |
| `runs/csi300/lightgbm/predictions.csv` | Every exported test score beside its stock, date, and label |
| `runs/csi300/lightgbm/portfolio.csv` | Qlib's daily portfolio returns, costs, benchmark, and turnover |

The registered primary model is selected by **validation Rank IC**, not test
profit. The results include all declared models; the single seed does not
provide a multi-seed robustness estimate. With a newer data release the test
endpoint can change. To reproduce the published snapshot exactly, use
`--release 2026-11-26` and a new folder such as `.data/cn-20261126`.

### Open Qlib's native charts

Once LightGBM has completed, run:

```bash
fts qlib-report --run-dir runs/csi300/lightgbm --output runs/csi300/lightgbm-plots
```

Open `runs/csi300/lightgbm-plots/report.html` in a browser. It contains interactive
charts and works offline. The same folder contains PNGs for a README and JSON
data for inspection. To chart another completed model, change `lightgbm` in
both paths. For the main comparison, use the validation-selected model named
in `comparison.json`.

| Chart | Beginner's interpretation |
| --- | --- |
| Portfolio report | How the portfolio and CSI300 benchmark changed, with trading costs and drawdowns |
| Daily IC/rank IC | Whether higher scores tended to identify higher-return stocks on each date |
| Monthly IC heatmap | Whether that ranking relationship persisted across months |
| Quintile returns | Returns of five groups sorted by score; a forecast diagnostic before costs |
| Risk summary | Excess-return risk and the effect of costs |

Native Qlib plots **add daily returns**; they are not compounded account-value
curves. Their default daily risk annualization uses **238 sessions/year**.
The comparison table separately reports compounded portfolio returns and
252-session annualization. The [chart guide](chenditc-benchmark.md#read-the-native-qlib-charts)
explains these conventions.

The recorded LightGBM replay also has a known stale valuation after an
instrument's price series ends: the affected holding is approximately 1.94%
of its final simulated account. Read the [terminated-series finding](chenditc-benchmark.md#terminated-series-and-portfolio-valuation)
before interpreting portfolio figures as achievable returns. The repository
retains the historical universe and reports the issue explicitly.

## Archived example: the bundled index data

The repository includes `^GSPC.csv`, so you can import it and train one model:

```bash
fts import-csv --csv "^GSPC.csv" --symbol SPX --provider .data/my-spx
fts train --config configs/timesfm3_mini.yaml --provider .data/my-spx --output runs/my-spx-timesfm --device cpu --no-backtest
```

The importer creates a **provider**: a local data folder that Qlib can read.
The second command uses it to train and evaluate against historical observations.
This old index dataset is a research example; `--no-backtest` avoids treating the
index itself as a directly tradable security.

For a longer comparison, `fts benchmark --provider .data/my-spx --output runs/my-benchmark`
runs all eight trainable models under three seeds, with up to 12 neural epochs.
That is 24 training runs, substantially more work than the one-epoch demo.
See the [benchmark instructions](../README.md#reproduce-the-three-seed-architecture-comparison)
and [recorded validation](architecture-validation.md) before interpreting results.

## 7. Replay the recorded SPY backtest

You can inspect a real historical simulation without training any model or
downloading more data. The repository includes the recorded forecasts and the
SPY adjusted-price window used in the [research report](financial-research.md).
Run this command from the repository root after activating your environment:

```bash
fts backtest --predictions reports/financial_research/predictions.csv --spy-csv reports/financial_research/spy_prices.csv --output runs/my-backtest
```

| Output | How to use it |
| --- | --- |
| `runs/my-backtest/metrics.csv` | Compare each strategy's return, annualized return, Sharpe ratio, drawdown, exposure and trade count. |
| `runs/my-backtest/ledger.csv` | Inspect every signal, execution date, position, fee and portfolio value. |
| `runs/my-backtest/settings.json` | Read the timing rule, price convention and cost assumptions. |

The default runs four cost scenarios: **0, 5, 10 and 20 basis points each way**.
Five basis points is 0.05% of traded notional on a purchase or sale. The rule holds
SPY after a positive prediction and holds cash otherwise; a signal after today's
close trades at the next session's close. Cash earns zero, and the simulation
charges for the final sale. It uses adjusted-price returns as a historical proxy,
not actual broker fills.

The README's archived SPX results use the **5-basis-point** rows. SPY buy-and-hold
outperformed every model strategy there, even though the newly trained TimesFM
mini ensemble had the best forecast RMSE. This is why both the forecast table
and the backtest matter. Choose a new output folder if you repeat the command.

## If something goes wrong

| Problem | Next step |
| --- | --- |
| `py` or `python3.12` is not found | Finish the Python installation, reopen the terminal, and repeat its version check. |
| `fts` is not found | Activate `.venv` again. You can also replace `fts` with `python -m financial_ts.cli`. |
| Import errors or the wrong Python version | Check `python -c "import sys; print(sys.executable)"`; use the interpreter inside this project's `.venv`, then repeat step 4. |
| PowerShell says scripts are disabled | Activation is optional. Replace `python` with `.\.venv\Scripts\python.exe`; replace `fts` with `.\.venv\Scripts\python.exe -m financial_ts.cli`. |
| `pyproject.toml`, a config, or the CSV cannot be found | Return to the repository root; check that the ZIP was extracted. |
| Provider or output already exists | Reuse the imported provider for training, or choose a new import destination. For another run, choose a new output such as `runs/my-second-demo`. |
| Downloading `latest` into an existing provider fails | Choose a new provider folder for the new release. Existing data are not silently overwritten. |
| Qlib charts report a missing Plotly, Kaleido, or statsmodels package | Run `python -m pip install -e ".[visualization]"` inside the same environment. |
| A chart output folder already contains files | Choose a new chart output path so the prior report stays inspectable. |
| No compatible PyTorch package | Check Python 3.12 and your operating system/processor against the official PyTorch installation selector above. |
| macOS reports `libomp.dylib` missing | Run `brew install libomp`, then retry in the same `.venv`. |

## Optional: Google's pretrained TimesFM 3

The regular demo needs no external weights. `timesfm3_pretrained` is a separate
option requiring an approximately **1.32 GB** checkpoint, an extra dependency,
and Google's **noncommercial, nonproduction** weight license. Follow the
[dedicated instructions](architectures/timesfm.md#timesfm3_pretrained) when you
want that experiment. It has different inputs and an unknown pretraining overlap
with historical evaluation data, so its scores require separate interpretation.

To add frozen 128/256-close forecasts, loop + adapter fine-tuning, and the matched
adapter-only control to the CSI300 campaign, use the locally downloaded checkpoint:

```bash
python -m pip install -e ".[timesfm,visualization]"
fts benchmark-cn --provider .data/cn --output runs/csi300-with-timesfm --timesfm-checkpoint .data/timesfm3-checkpoint
```

This starts a new twelve-model campaign. Official adaptation uses 1,024 training
examples and 512 validation-selection examples for three epochs, then evaluates
the complete declared validation/test panel. These models use raw adjusted-close
contexts; the eight scratch architectures use Qlib factors.
