# Algorithmic Trading Assignment 2

## Strategy Overview
Our strategy progresses through three sequential stages, each designed to isolate and then compound improvements in execution quality:
- **Stage 1(A)**: Optimizes timing filters with a fixed spend per fill to isolate signal gating effects.
- **Stage 1(B)**: Locks the Stage 1(A) timing parameters and introduces dynamic, gate-based position sizing so fills scale with conviction.
- **Stage 1(C)**: Layers an Average True Range (ATR) multiplier on top of Stage 1(B) sizing logic for volatility-aware scaling.

The working hypothesis is that gating sharpens execution timing, while dynamic sizing and ATR-based adjustments further enhance risk-adjusted performance.

## Stage 1: In-Sample Training
- **Universe & Data**: Uses adjusted hourly bars for the SPY ETF from 2017-01-01 through 2024-01-01. SPY’s liquidity mirrors mainstream paycheck investing behaviour and anchors the parameter search on robust data.
- **Optimization Loop**: For each indicator family, Stage 1(A) tunes filter thresholds under a constant spend assumption. Stage 1(B) then locks timing parameters and sweeps dynamic sizing gates that modulate spending per trigger. Stage 1(C) adds an ATR multiplier to the selected Stage 1(B) configuration, allowing fills to expand or contract with recent volatility.
- **Cost & Execution Assumptions**: Commission costs follow QuantConnect’s default Interactive Brokers (IBKR) fee model; slippage rails are disabled given the high liquidity of index ETFs. All executions are market buy orders for whole shares.

## Stage 2: Out-of-Sample Evaluation
- **Assets**: Applies the winning Stage 1 model for each indicator family to SPY, IWM, RWM, and SH.
- **Windowing Scheme**: Evaluates rolling windows beginning 2023-01-01—2024-01-01 and extending both the start and end date forward by one month until the final window of 2024-01-01—2025-01-01. This mitigates overfitting and highlights seasonal robustness.
- **Benchmark**: Compares each candidate against a baseline dollar-cost averaging (DCA) model that deposits on schedule without timing filters.

## Cash Flow & Scheduling
- **Deposit Cadence**: Deposits occur every month (`DcaFrequencyMonths`) with a tranche size of USD 1,000 (`DcaCashAmount`). Deposits are treated as external cash flows so that performance metrics reflect market returns only.
- **Decision Frequency**: All assets are sampled hourly. On each bar, timing gates evaluate whether the indicator value is at/below its threshold. If true, the model executes at the close of that bar; otherwise funds roll forward.

## Dynamic Sizing & ATR Layering
- **Gate-Based Sizing**: Stage 1(B) replaces fixed spending with gate tiers. Each open gate allocates a different fraction of the available DCA tranche, allowing the model to scale exposure when multiple signals align.
- **ATR Multiplier**: Stage 1(C) multiplies the gate-derived allocation by an ATR-based factor. Higher ATR expands position size within defined caps to exploit high conviction regimes, while lower ATR contracts the spend to limit risk in quieter markets.

## Performance Measurement
- **Metrics**: Reports Compounded Annual Growth Rate (CAGR), time-weighted return (TWR), Sharpe ratio, and maximum drawdown. Metrics are sampled daily after the close, while drawdowns are plotted weekly to stay within platform limits.
- **TWR Adjustments**: QuantConnect’s default statistics assume a single initial deposit. We override this to compute true time-weighted returns that accommodate recurring external cash flows.

## Code Workflow
1. **Stage Configuration**: YAML/JSON parameter grids define gates, thresholds, ATR multipliers, and DCA settings to iterate through during Stage 1 training.
2. **Optimization Harness**: The Stage 1 runner loads SPY data, applies filters, and records performance metrics per configuration. Winners per indicator family are serialized for reuse.
3. **Evaluation Runner**: Stage 2 loads the saved configurations and replays them across the multi-asset OOS set, logging equity curves, TWR statistics, and drawdown profiles.
4. **Reporting**: Scripts aggregate the results into comparative tables and plots used to select the final candidate strategies.

## Repository Usage
- **`Quantconnect code/dca_baseline.py`**: Upload to a QuantConnect project and run in the cloud to produce the benchmark DCA equity curve that anchors all comparisons.
- **`Quantconnect code/twr_tracker.py`**: Import alongside any stage script inside QuantConnect so the live run emits corrected TWR, Sharpe, and drawdown statistics during execution.
- **QC Optimizer Outputs**: The `Results/` directory captures cloud optimizer runs (QuantConnect wizard supports up to three simultaneous parameters), including JSON logs and exported images for the top configurations.
- **`Results/json_qc_fetch.ipynb`**: Configure API credentials at the top, then run to fetch backtest statistics from QuantConnect via REST, flatten the JSON into CSV tables, and render heatmaps per asset/shift so optimizer winners can be compared visually.
- **`Results/hypothesis_testing.ipynb`**: Load the saved model vs benchmark JSON pairs to compute Newey–West HAC t-tests on daily log-return spreads, annotate the results with Durbin–Watson and AR(1) diagnostics, and output a summary table plus CSV/JSON artifacts for reporting.
- **Stage Archives**: The remaining stage folders store intermediate gates, ATR sweeps, plots, and logs so you can trace how each stage’s choices lead to the reported performance.

## Limitations
- QC’s cloud optimizer wizard accepts at most three parameters per job, constraining the grid search and forcing staged sweeps for wider parameter combinations.
- Local optimization via the Lean engine Docker container failed periodically, so all calibration and validation relied on QuantConnect’s managed cloud runs.

## References
OpenAI. (2025). *ChatGPT (GPT-5)* [Large language model]. https://chat.openai.com/

QuantConnect. (2025). *Lean algorithmic trading engine documentation*. https://www.quantconnect.com/docs/v2

Google. (2025). *Gemini API documentation*. https://ai.google.dev/gemini-api
