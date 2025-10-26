# region imports
from AlgorithmImports import *
from QuantConnect.Indicators import StochasticRelativeStrengthIndex
from QuantConnect.Securities import MarketHoursState, Security
from System import DayOfWeek
# endregion

import math
import calendar
from collections import deque
from datetime import datetime, timedelta
from typing import Optional, Sequence

try:
    from twr_tracker import TimeWeightedReturnTracker
except ImportError as error:
    raise ImportError(
        "twr_tracker.py must be deployed alongside dca_baseline.py; "
        "unable to import TimeWeightedReturnTracker."
    ) from error


DEFAULT_START_DATE = datetime(2019, 1, 1)
DEFAULT_END_DATE = datetime(2025, 6, 30)
MIN_START_DATE = datetime(2007, 1, 1)
DEFAULT_DISTRIBUTION_LOOKBACK_DAYS = 252
MIN_DISTRIBUTION_LOOKBACK_DAYS = 10
MIN_WARMUP_DAYS = 30
DEFAULT_RAMP_QUANTILE = 0.75
DEFAULT_RAMP_BARS_FLOOR = 2
DEFAULT_RAMP_BARS_CAP = 64
HOUR_SECONDS = 3600.0
DEFAULT_INITIAL_CASH = 1000.0
DEFAULT_DCA_CASH_AMOUNT = 1000.0
DEFAULT_DCA_FREQUENCY_MONTHS = 1
DEFAULT_DCA_FREQUENCY_DAYS = 0
DEFAULT_DCA_FREQUENCY_HOURS = 0
DEFAULT_BACKTEST_SHIFT_MONTHS = 0
TRANCHE_UNIT_FLOOR = 1e-9
SPAN_FLOOR = 1e-9
ATR_HISTORY_MIN_SAMPLES = 10
PERCENTILE_TIE_WEIGHT = 0.5


class DollarCostAverageEnhanced(QCAlgorithm):
    """Dollar-cost averaging baseline with gate-driven sizing plus
    SMI/ATR extensions.
    """

    # Initialize
    def Initialize(self) -> None:
        self.SetTimeZone(TimeZones.NewYork)
        default_end = DEFAULT_END_DATE
        default_start = DEFAULT_START_DATE
        start_year = self.get_int_parameter("StartYear", default_start.year)
        start_month = self.get_int_parameter("StartMonth", default_start.month)
        start_day = self.get_int_parameter("StartDay", default_start.day)
        try:
            requested_start = datetime(start_year, start_month, start_day)
        except ValueError as error:
            message = (
                "StartYear/StartMonth/StartDay produce invalid date: "
                f"{error}"
            )
            raise ValueError(message) from error
        min_start_date = MIN_START_DATE
        start_date = max(requested_start, min_start_date)

        end_year = self.get_int_parameter("EndYear", default_end.year)
        end_month = self.get_int_parameter("EndMonth", default_end.month)
        end_day = self.get_int_parameter("EndDay", default_end.day)
        try:
            requested_end = datetime(end_year, end_month, end_day)
        except ValueError as error:
            message = (
                "EndYear/EndMonth/EndDay produce invalid date: "
                f"{error}"
            )
            raise ValueError(message) from error
        if requested_end > default_end:
            requested_end = default_end
        end_date = requested_end

        if start_date > end_date:
            message = (
                f"Start date {start_date.date()} must be on or before "
                f"end date {end_date.date()}."
            )
            raise ValueError(message)

        shift_months = self.get_int_parameter(
            "BacktestShiftMonths", DEFAULT_BACKTEST_SHIFT_MONTHS)
        safe_shift = shift_months
        if safe_shift > 0:
            while (
                safe_shift > 0
                and self.add_months(end_date, safe_shift) > DEFAULT_END_DATE
            ):
                safe_shift -= 1
        elif safe_shift < 0:
            while (
                safe_shift < 0
                and self.add_months(start_date, safe_shift) < MIN_START_DATE
            ):
                safe_shift += 1

        self.backtest_shift_months = safe_shift
        if safe_shift != shift_months:
            original_shift = shift_months
            self.Log(
                "BacktestShiftMonths adjusted from "
                f"{original_shift} to {safe_shift} to stay within the "
                "supported data window."
            )

        if safe_shift != 0:
            start_date = self.add_months(start_date, safe_shift)
            end_date = self.add_months(end_date, safe_shift)

        if start_date < MIN_START_DATE or end_date > DEFAULT_END_DATE:
            start_date = max(start_date, MIN_START_DATE)
            end_date = min(end_date, DEFAULT_END_DATE)
            if start_date > end_date:
                message = (
                    "BacktestShiftMonths adjustment collapsed the backtest "
                    "window. Review the configured dates."
                )
                raise ValueError(message)

        self.SetStartDate(start_date.year, start_date.month, start_date.day)
        self.SetEndDate(end_date.year, end_date.month, end_date.day)

        # Start with zero cash; deposits are pushed via the CashBook during
        # backtests to simulate DCA funding.
        self.SetCash(DEFAULT_INITIAL_CASH)

        equity_param = self.GetParameter("Equity") or "SPY"
        ticker = equity_param.strip().upper() or "SPY"
        self.data_resolution = Resolution.Hour
        self.bar_interval = self.get_resolution_timedelta(self.data_resolution)
        security = self.AddEquity(ticker, self.data_resolution)
        security.SetDataNormalizationMode(DataNormalizationMode.Adjusted)
        self.symbol = security.Symbol
        self.SetBenchmark(self.symbol)
        self.bars_per_day = self.estimate_bars_per_day(
            security, self.data_resolution)

        # Core cash/cadence
        self.dca_cash_amount = self.get_float_parameter(
            "DcaCashAmount", DEFAULT_DCA_CASH_AMOUNT, minimum=0.0)
        self.dca_frequency_months = max(
            0,
            self.get_int_parameter(
                "DcaFrequencyMonths", DEFAULT_DCA_FREQUENCY_MONTHS),
        )
        self.dca_frequency_days = max(
            0,
            self.get_int_parameter(
                "DcaFrequencyDays", DEFAULT_DCA_FREQUENCY_DAYS),
        )
        self.dca_frequency_hours = max(
            0,
            self.get_int_parameter(
                "DcaFrequencyHours", DEFAULT_DCA_FREQUENCY_HOURS),
        )
        self.deploy_all_on_gate = self.get_bool_parameter(
            "DeployAllOnGate", False)

        # RSI gate
        self.use_rsi_gate = self.get_bool_parameter("UseRsiGate", False)
        self.rsi_period = max(2, self.get_int_parameter("RsiPeriod", 21))
        self.rsi_threshold = self.get_float_parameter(
            "RsiThreshold", 40.0, minimum=0.0, maximum=100.0)

        # Bollinger gate
        self.use_bb_gate = self.get_bool_parameter("UseBollingerGate", False)
        self.bb_period = max(2, self.get_int_parameter("BbPeriod", 20))
        self.bb_std_dev = self.get_float_parameter(
            "BbStdDeviation", 2.0, minimum=0.1)
        self.bb_z_threshold = self.get_float_parameter("BbZThreshold", -1.0)

        # Moving average gates
        self.use_ema_gate = self.get_bool_parameter("UseEmaGate", False)
        self.ema_period = max(2, self.get_int_parameter("EmaPeriod", 80))
        self.ema_diff_threshold = self.get_float_parameter(
            "EmaDiffThreshold", -0.01)

        self.use_sma_gate = self.get_bool_parameter("UseSmaGate", False)
        self.sma_period = max(2, self.get_int_parameter("SmaPeriod", 60))
        self.sma_diff_threshold = self.get_float_parameter(
            "SmaDiffThreshold", -0.01)

        # New per-gate sizing caps (with sensible fallbacks)
        self.rsi_min_mult = self.get_float_parameter(
            "RsiMinMultiplier", 0.85, minimum=0.0)
        self.rsi_max_mult = self.get_float_parameter(
            "RsiMaxMultiplier", 1.5, minimum=0.0)
        self.bb_min_mult = self.get_float_parameter(
            "BbMinMultiplier",  1, minimum=0.0)
        self.bb_max_mult = self.get_float_parameter(
            "BbMaxMultiplier",  1, minimum=0.0)
        self.ema_min_mult = self.get_float_parameter(
            "EmaMinMultiplier", 0.05, minimum=0.0)
        self.ema_max_mult = self.get_float_parameter(
            "EmaMaxMultiplier", 1, minimum=0.0)
        self.sma_min_mult = self.get_float_parameter(
            "SmaMinMultiplier", 0.75, minimum=0.0)
        self.sma_max_mult = self.get_float_parameter(
            "SmaMaxMultiplier", 1.5, minimum=0.0)

        # Composite controls
        self.overall_min_mult = self.get_float_parameter(
            "OverallMinMultiplier", 0.125, minimum=0.0)
        self.overall_max_mult = self.get_float_parameter(
            "OverallMaxMultiplier", 4.0,  minimum=0.0)
        self.combine_method = (self.GetParameter(
            "CombineMultipliers") or "product").strip().lower()
        if self.combine_method not in {"product", "sum", "max"}:
            self.combine_method = "product"

        # Stochastic RSI sizing
        self.use_srsi_sizing = self.get_bool_parameter("UseSrsiSizing", False)
        self.srsi_period = max(2, self.get_int_parameter("SrsiPeriod", 14))
        self.srsi_stochastic_period = max(
            2, self.get_int_parameter("SrsiStochasticPeriod", 14))
        self.srsi_smooth_period = max(
            1, self.get_int_parameter("SrsiSmoothPeriod", 3))
        self.srsi_threshold = self.get_float_parameter(
            "SrsiThreshold", 0.20, minimum=0.0, maximum=1.0)
        self.srsi_min_mult = self.get_float_parameter(
            "SrsiMinMultiplier", 0.85, minimum=0.0)
        self.srsi_max_mult = self.get_float_parameter(
            "SrsiMaxMultiplier", 1.5, minimum=0.0)

        # ATR sizing controls
        self.use_atr_sizing = self.get_bool_parameter("UseAtrSizing", False)
        self.atr_period = max(1, self.get_int_parameter("AtrPeriod", 14))
        self.atr_min_mult = self.get_float_parameter(
            "AtrMinMultiplier", 0.5, minimum=0.0)
        self.atr_max_mult = self.get_float_parameter(
            "AtrMaxMultiplier", 1.5, minimum=0.0)
        if self.atr_min_mult > self.atr_max_mult:
            self.atr_min_mult, self.atr_max_mult = (
                self.atr_max_mult,
                self.atr_min_mult,
            )
        atr_mode_param = self.GetParameter("AtrSizingMode") or "composite"
        self.atr_mode = atr_mode_param.strip().lower()
        if self.atr_mode not in {"composite", "standalone"}:
            self.atr_mode = "composite"

        # Ramp behavior
        ramp_override = (self.GetParameter("RampBarsToMax") or "").strip()
        self.dynamic_ramp_enabled = ramp_override == ""
        if self.dynamic_ramp_enabled:
            self.ramp_quantile = self.get_float_parameter(
                "RampBarsQuantile",
                DEFAULT_RAMP_QUANTILE,
                minimum=0.0,
                maximum=0.99,
            )
            floor_param = self.get_int_parameter(
                "RampBarsFloor",
                DEFAULT_RAMP_BARS_FLOOR,
            )
            self.ramp_floor_bars = max(1, floor_param)
            cap_param = self.get_int_parameter(
                "RampBarsCap",
                DEFAULT_RAMP_BARS_CAP,
            )
            self.ramp_cap_bars = max(self.ramp_floor_bars, cap_param)
            self.ramp_bars_to_max = float(self.ramp_floor_bars)
        else:
            try:
                ramp_override_value = max(1, int(round(float(ramp_override))))
            except ValueError as error:
                message = (
                    f"RampBarsToMax parameter '{ramp_override}' must be "
                    "numeric."
                )
                raise ValueError(message) from error
            self.dynamic_ramp_enabled = False
            self.ramp_quantile = None
            self.ramp_floor_bars = ramp_override_value
            self.ramp_cap_bars = ramp_override_value
            self.ramp_bars_to_max = float(ramp_override_value)
        self.reset_pressure_on_close = self.get_bool_parameter(
            "ResetPressureOnClose", True)

        # Distribution windows
        lookback_default = DEFAULT_DISTRIBUTION_LOOKBACK_DAYS
        self.dist_lookback_days = max(
            MIN_DISTRIBUTION_LOOKBACK_DAYS,
            self.get_int_parameter(
                "DistributionLookbackDays", lookback_default),
        )
        self.dist_lookback_bars = max(
            1, int(self.dist_lookback_days * self.bars_per_day))
        gate_samples_param = (self.GetParameter(
            "MinSamplesForGate") or "").strip()
        if gate_samples_param:
            try:
                self.min_gate_samples = max(0, int(float(gate_samples_param)))
            except ValueError as error:
                message = (
                    f"MinSamplesForGate parameter '{gate_samples_param}' "
                    "must be integer."
                )
                raise ValueError(message) from error
        else:
            self.min_gate_samples = max(
                0, self.get_int_parameter("MinSamplesForQuantile", 60))

        # Indicators (hourly)
        self.rsi: Optional[RelativeStrengthIndex] = None
        if self.use_rsi_gate:
            self.rsi = self.RSI(self.symbol, self.rsi_period,
                                MovingAverageType.Wilders, Resolution.Hour)

        self.bollinger_bands: Optional[BollingerBands] = None
        if self.use_bb_gate:
            self.bollinger_bands = self.BB(
                self.symbol,
                self.bb_period,
                self.bb_std_dev,
                MovingAverageType.Simple,
                Resolution.Hour,
            )

        self.ema_gate: Optional[ExponentialMovingAverage] = None
        if self.use_ema_gate:
            self.ema_gate = self.EMA(
                self.symbol, self.ema_period, Resolution.Hour)

        self.sma_gate: Optional[SimpleMovingAverage] = None
        if self.use_sma_gate:
            self.sma_gate = self.SMA(
                self.symbol, self.sma_period, Resolution.Hour)

        self.srsi: Optional[StochasticRelativeStrengthIndex] = None
        if self.use_srsi_sizing:
            self.srsi = StochasticRelativeStrengthIndex(
                int(self.srsi_period),
                int(self.srsi_stochastic_period),
                int(self.srsi_smooth_period),
                int(self.srsi_smooth_period),
            )
            self.RegisterIndicator(self.symbol, self.srsi, Resolution.Hour)

        self.atr: Optional[AverageTrueRange] = None
        if self.use_atr_sizing:
            self.atr = self.ATR(
                self.symbol,
                self.atr_period,
                MovingAverageType.Wilders,
                Resolution.Hour,
            )

        # Rolling distributions
        self.rsi_values = deque(maxlen=self.dist_lookback_bars)
        self.bb_z_values = deque(maxlen=self.dist_lookback_bars)
        self.ema_diff_values = deque(maxlen=self.dist_lookback_bars)
        self.sma_diff_values = deque(maxlen=self.dist_lookback_bars)
        self.srsi_values = deque(maxlen=self.dist_lookback_bars)
        self.atr_ratios = deque(maxlen=self.dist_lookback_bars)

        # Progressive ramp state (per gate)
        self.rsi_pressure = 0.0
        self.bb_pressure = 0.0
        self.ema_pressure = 0.0
        self.sma_pressure = 0.0
        self.srsi_pressure = 0.0
        self.gate_open_streaks = {
            "rsi": 0,
            "bb": 0,
            "ema": 0,
            "sma": 0,
            "srsi": 0,
        }
        self.gate_open_lengths = deque(maxlen=self.dist_lookback_bars)

        # Schedule state
        self.next_base_cycle_time: Optional[datetime] = None
        self.last_purchase_time: Optional[datetime] = None

        # Account currency used for deposits (backtests)
        self.deposit_ccy = getattr(self, "AccountCurrency", "USD")

        # Warmup horizon (bars)
        indicator_warmup = [
            self.rsi_period if self.use_rsi_gate else 0,
            self.bb_period if self.use_bb_gate else 0,
            self.ema_period if self.use_ema_gate else 0,
            self.sma_period if self.use_sma_gate else 0,
            self.srsi_period if self.use_srsi_sizing else 0,
            self.atr_period if self.use_atr_sizing else 0,
        ]
        indicator_max_bars = max(indicator_warmup) if indicator_warmup else 0
        min_warmup_bars = max(1, int(MIN_WARMUP_DAYS * self.bars_per_day))
        warmup_bars = max(
            min_warmup_bars, self.dist_lookback_bars, indicator_max_bars)
        warmup_period = self.bar_interval * warmup_bars
        self.SetWarmUp(warmup_period)

        if self.dynamic_ramp_enabled and self.ramp_quantile is not None:
            ramp_desc = (
                f"{self.ramp_bars_to_max:.0f} "
                f"(q={self.ramp_quantile:0.2f})"
            )
        else:
            ramp_desc = f"{self.ramp_bars_to_max:.0f}"

        debug_parts = [
            f"DCA ready {self.symbol}",
            f"| ${self.dca_cash_amount:,.0f}/tranche",
            (
                f"| cadence: {self.dca_frequency_months}m/"
                f"{self.dca_frequency_days}d/"
                f"{self.dca_frequency_hours}h"
            ),
            f"| shift={self.backtest_shift_months}m",
            (
                f"| hist={self.dist_lookback_days}d"
                f"≈{self.dist_lookback_bars} bars"
            ),
            (
                f"| RSI gate {self.use_rsi_gate} "
                f"≤{self.rsi_threshold:.2f} "
                f"[{self.rsi_min_mult:.2f},{self.rsi_max_mult:.2f}]"
            ),
            (
                f"| BB gate {self.use_bb_gate} "
                f"≤{self.bb_z_threshold:.2f} "
                f"[{self.bb_min_mult:.2f},{self.bb_max_mult:.2f}]"
            ),
            (
                f"| EMA gate {self.use_ema_gate} "
                f"≤{self.ema_diff_threshold:.2%} "
                f"[{self.ema_min_mult:.2f},{self.ema_max_mult:.2f}]"
            ),
            (
                f"| SMA gate {self.use_sma_gate} "
                f"≤{self.sma_diff_threshold:.2%} "
                f"[{self.sma_min_mult:.2f},{self.sma_max_mult:.2f}]"
            ),
            (
                f"| SRSI sizing {self.use_srsi_sizing} "
                f"≤{self.srsi_threshold:.2f} "
                f"[{self.srsi_min_mult:.2f},{self.srsi_max_mult:.2f}]"
            ),
            (
                f"| ATR sizing {self.use_atr_sizing} "
                f"mode={self.atr_mode} "
                f"[{self.atr_min_mult:.2f},{self.atr_max_mult:.2f}]"
            ),
            f"| ramp_to_max={ramp_desc}",
            f"| combiner={self.combine_method}",
        ]
        self.Debug(" ".join(debug_parts))

        self.twr = TimeWeightedReturnTracker(self, self.symbol)

    # Hourly driver
    def estimate_bars_per_day(
        self,
        security: Security,
        resolution: Resolution,
    ) -> int:
        interval = self.get_resolution_timedelta(resolution)
        interval_seconds = max(interval.total_seconds(), 1.0)
        trading_seconds = 0.0
        exchange_hours = security.Exchange.Hours
        open_states = {getattr(MarketHoursState, "Open", None)}
        market_state = getattr(MarketHoursState, "Market", None)
        if market_state is not None:
            open_states.add(market_state)
        business_days = [
            DayOfWeek.Monday,
            DayOfWeek.Tuesday,
            DayOfWeek.Wednesday,
            DayOfWeek.Thursday,
            DayOfWeek.Friday,
        ]
        for day in business_days:
            try:
                local_hours = exchange_hours.MarketHours[day]
            except KeyError:
                continue
            day_seconds = 0.0
            segments = getattr(local_hours, "Segments", None)
            if not segments:
                continue
            for segment in segments:
                state = getattr(segment, "State", None)
                if state not in open_states:
                    continue
                delta = segment.End - segment.Start
                day_seconds += max(delta.total_seconds(), 0.0)
            if day_seconds > 0:
                trading_seconds = max(trading_seconds, day_seconds)
        if trading_seconds <= 0.0:
            trading_seconds = 24.0 * HOUR_SECONDS
        bars = int(math.ceil(trading_seconds / interval_seconds))
        return max(1, bars)

    @staticmethod
    def get_resolution_timedelta(resolution: Resolution) -> timedelta:
        if resolution == Resolution.Second:
            return timedelta(seconds=1)
        if resolution == Resolution.Minute:
            return timedelta(minutes=1)
        if resolution == Resolution.Hour:
            return timedelta(hours=1)
        if resolution == Resolution.Daily:
            return timedelta(days=1)
        return timedelta(hours=1)

    def OnData(self, data: Slice) -> None:
        if self.symbol not in data.Bars:
            return

        # keep distributions and ramp stats updated even during warmup
        self.update_distributions()
        self.update_gate_pressures()

        if self.IsWarmingUp:
            return

        self.handle_scheduled()

    # Scheduler (now called hourly)
    def handle_scheduled(self) -> None:
        now = self.Time

        # Update rolling distributions
        # Initialize cadence anchor
        if self.next_base_cycle_time is None:
            self.next_base_cycle_time = now

        # Add tranche cash when due (deposit into CashBook; backtests only)
        if now >= self.next_base_cycle_time:
            self.Portfolio.CashBook[self.deposit_ccy].AddAmount(
                self.dca_cash_amount)
            self.twr.register_cash_flow(self.dca_cash_amount)
            self.next_base_cycle_time = self.compute_next_cycle_time(now)
            deposit_message = (
                f"{now}: tranche DEPOSIT +${self.dca_cash_amount:,.2f} "
                f"| cash=${float(self.Portfolio.Cash):,.2f}"
            )
            self.Debug(deposit_message)
            if not self.any_gates_enabled():
                self.deploy_tranches(1, reason="Baseline DCA (no gates)")
                return

        # Gate-aware deployment
        if self.any_gates_enabled() and self.filters_allow_entry():
            # Base tranche count inferred from available cash
            cash_available = float(self.Portfolio.Cash)
            tranche_unit = max(self.dca_cash_amount, TRANCHE_UNIT_FLOOR)
            pending = int(cash_available // tranche_unit)
            if pending <= 0:
                return
            tranches = pending if self.deploy_all_on_gate else 1
            self.deploy_tranches(tranches, reason="Gate-approved DCA")

    # Core deploy
    def deploy_tranches(self, num_tranches: int, reason: str) -> None:
        if num_tranches <= 0:
            return

        # Composite multiplier from active gates (continuous ramp)
        size_mult = self.get_composite_size_multiplier()

        intended = self.dca_cash_amount * num_tranches * size_mult
        spend_cash = min(intended, float(self.Portfolio.Cash))
        if spend_cash <= 0:
            return

        self.deploy_cash(spend_cash, reason, size_mult)

    def deploy_cash(
        self,
        target_cash: float,
        reason: str,
        size_mult: float,
    ) -> None:
        security = self.Securities[self.symbol]
        price = security.Price
        if not math.isfinite(price) or price <= 0:
            return

        available_cash = float(self.Portfolio.Cash)
        max_cash = min(target_cash, available_cash)
        if max_cash <= 0:
            return

        raw_quantity = max_cash / price
        quantity = int(math.floor(raw_quantity))
        if quantity <= 0:
            return

        order_ticket = self.MarketOrder(self.symbol, quantity)
        if order_ticket is None:
            return

        spent_value = quantity * price
        self.last_purchase_time = self.Time

        # Diagnostics
        msg = (
            f"{self.Time}: {reason} buy {quantity} @ {price:.2f} "
            f"spent ${spent_value:,.2f}"
        )

        # Per-gate diagnostics
        if (
            self.use_rsi_gate
            and self.rsi
            and self.rsi.IsReady
            and len(self.rsi_values) > 0
        ):
            cur_rsi = self.rsi.Current.Value
            rsi_i = self.gate_intensity_rsi()
            rsi_mult = self.scale_pressure_to_multiplier(
                self.rsi_pressure, self.rsi_min_mult, self.rsi_max_mult)
            msg += (
                f" | RSI {cur_rsi:.2f}≤{self.rsi_threshold:.2f}"
                f" i={rsi_i:.2f} P={self.rsi_pressure:.1f} m={rsi_mult:.2f}"
            )

        if (
            self.use_bb_gate
            and self.bollinger_bands
            and self.bollinger_bands.IsReady
            and len(self.bb_z_values) > 0
        ):
            bollinger_z_score = self.current_bb_z()
            bb_i = self.gate_intensity_bb()
            bb_mult = self.scale_pressure_to_multiplier(
                self.bb_pressure, self.bb_min_mult, self.bb_max_mult)
            msg += (
                f" | BBz {bollinger_z_score:.2f}≤{self.bb_z_threshold:.2f}"
                f" i={bb_i:.2f} P={self.bb_pressure:.1f} m={bb_mult:.2f}"
            )

        if (
            self.use_ema_gate
            and self.ema_gate
            and self.ema_gate.IsReady
            and len(self.ema_diff_values) > 0
        ):
            diff = self.current_ema_diff()
            ema_i = self.gate_intensity_ema()
            ema_mult = self.scale_pressure_to_multiplier(
                self.ema_pressure, self.ema_min_mult, self.ema_max_mult)
            msg += (
                f" | EMAΔ {diff:.2%}≤{self.ema_diff_threshold:.2%}"
                f" i={ema_i:.2f} P={self.ema_pressure:.1f} m={ema_mult:.2f}"
            )

        if (
            self.use_sma_gate
            and self.sma_gate
            and self.sma_gate.IsReady
            and len(self.sma_diff_values) > 0
        ):
            diff = self.current_sma_diff()
            sma_i = self.gate_intensity_sma()
            sma_mult = self.scale_pressure_to_multiplier(
                self.sma_pressure, self.sma_min_mult, self.sma_max_mult)
            msg += (
                f" | SMAΔ {diff:.2%}≤{self.sma_diff_threshold:.2%}"
                f" i={sma_i:.2f} P={self.sma_pressure:.1f} m={sma_mult:.2f}"
            )

        if (
            self.use_srsi_sizing
            and self.srsi
            and self.srsi.IsReady
            and len(self.srsi_values) > 0
        ):
            srsi_val = self.srsi.Current.Value
            srsi_i = self.srsi_intensity()
            srsi_mult = self.scale_pressure_to_multiplier(
                self.srsi_pressure, self.srsi_min_mult, self.srsi_max_mult)
            msg += (
                f" | SRSI {srsi_val:.2f}≤{self.srsi_threshold:.2f}"
                f" i={srsi_i:.2f} P={self.srsi_pressure:.1f} m={srsi_mult:.2f}"
            )

        if (
            self.use_atr_sizing
            and self.atr
            and self.atr.IsReady
            and len(self.atr_ratios) > 0
        ):
            atr_ratio = self.current_atr_ratio()
            if math.isfinite(atr_ratio):
                atr_percentile = self.percentile_rank(
                    list(self.atr_ratios),
                    atr_ratio,
                )
                atr_multiplier = self.atr_size_multiplier()
                msg += (
                    f" | ATR ratio={atr_ratio:.4f}"
                    f" pct={atr_percentile:.2f} m={atr_multiplier:.2f}"
                )

        msg += (
            f" | total_mult={size_mult:.2f}"
            f" | cash_left=${float(self.Portfolio.Cash):,.2f}"
        )
        self.Debug(msg)

    # Gates
    def any_gates_enabled(self) -> bool:
        return (
            self.use_rsi_gate
            or self.use_bb_gate
            or self.use_ema_gate
            or self.use_sma_gate
        )

    def indicators_ready(self) -> bool:
        if (
            self.use_rsi_gate
            and (self.rsi is None or not self.rsi.IsReady)
        ):
            return False
        if (
            self.use_bb_gate
            and (
                self.bollinger_bands is None
                or not self.bollinger_bands.IsReady
            )
        ):
            return False
        if (
            self.use_ema_gate
            and (self.ema_gate is None or not self.ema_gate.IsReady)
        ):
            return False
        if (
            self.use_sma_gate
            and (self.sma_gate is None or not self.sma_gate.IsReady)
        ):
            return False
        return True

    def filters_allow_entry(self) -> bool:
        if not self.indicators_ready():
            return False

        if self.use_rsi_gate:
            if len(self.rsi_values) < self.min_gate_samples:
                return False
            cur_rsi = self.rsi.Current.Value
            if not math.isfinite(cur_rsi) or cur_rsi > self.rsi_threshold:
                return False

        if self.use_bb_gate:
            if len(self.bb_z_values) < self.min_gate_samples:
                return False
            cur_z = self.current_bb_z()
            if not math.isfinite(cur_z) or cur_z > self.bb_z_threshold:
                return False

        if self.use_ema_gate:
            if len(self.ema_diff_values) < self.min_gate_samples:
                return False
            cur_diff = self.current_ema_diff()
            if not math.isfinite(cur_diff):
                return False
            if cur_diff > self.ema_diff_threshold:
                return False

        if self.use_sma_gate:
            if len(self.sma_diff_values) < self.min_gate_samples:
                return False
            cur_diff = self.current_sma_diff()
            if not math.isfinite(cur_diff):
                return False
            if cur_diff > self.sma_diff_threshold:
                return False

        return True

    # Continuous, progressive per-gate intensities & ramp
    def update_gate_pressures(self) -> None:
        """Accumulate per-gate pressure while gate is open (using intensity in
        [0, 1]); reset or decay when closed.
        """

        def handle_gate(
            name: str,
            enabled: bool,
            indicator_ready: bool,
            sample_count: int,
            pressure_attr: str,
            intensity_func,
        ) -> None:
            meets_requirements = (
                enabled
                and indicator_ready
                and sample_count >= self.min_gate_samples
            )
            if not meets_requirements:
                self.record_gate_open_state(name, False)
                setattr(self, pressure_attr, 0.0)
                return

            intensity = intensity_func()
            is_open = intensity > 0
            self.record_gate_open_state(name, is_open)

            pressure = getattr(self, pressure_attr)
            if is_open:
                updated = min(self.ramp_bars_to_max, pressure + intensity)
            else:
                if self.reset_pressure_on_close:
                    updated = 0.0
                else:
                    updated = max(0.0, pressure - 1.0)
            setattr(self, pressure_attr, updated)

        handle_gate(
            "rsi",
            self.use_rsi_gate,
            bool(self.rsi and self.rsi.IsReady),
            len(self.rsi_values),
            "rsi_pressure",
            self.gate_intensity_rsi,
        )
        handle_gate(
            "bb",
            self.use_bb_gate,
            bool(self.bollinger_bands and self.bollinger_bands.IsReady),
            len(self.bb_z_values),
            "bb_pressure",
            self.gate_intensity_bb,
        )
        handle_gate(
            "ema",
            self.use_ema_gate,
            bool(self.ema_gate and self.ema_gate.IsReady),
            len(self.ema_diff_values),
            "ema_pressure",
            self.gate_intensity_ema,
        )
        handle_gate(
            "sma",
            self.use_sma_gate,
            bool(self.sma_gate and self.sma_gate.IsReady),
            len(self.sma_diff_values),
            "sma_pressure",
            self.gate_intensity_sma,
        )
        handle_gate(
            "srsi",
            self.use_srsi_sizing,
            bool(self.srsi and self.srsi.IsReady),
            len(self.srsi_values),
            "srsi_pressure",
            self.srsi_intensity,
        )

        self.update_dynamic_ramp()

    def record_gate_open_state(self, gate: str, is_open: bool) -> None:
        streak = int(self.gate_open_streaks.get(gate, 0))
        if is_open:
            streak += 1
        else:
            if streak > 0:
                self.gate_open_lengths.append(streak)
            streak = 0
        self.gate_open_streaks[gate] = streak

    def update_dynamic_ramp(self) -> None:
        if not self.dynamic_ramp_enabled:
            return
        lengths = list(self.gate_open_lengths)
        for streak in self.gate_open_streaks.values():
            if streak > 0:
                lengths.append(streak)
        if not lengths:
            self.ramp_bars_to_max = float(self.ramp_floor_bars)
            return
        quantile_value = self.select_quantile(lengths, self.ramp_quantile)
        if not math.isfinite(quantile_value):
            self.ramp_bars_to_max = float(self.ramp_floor_bars)
            return
        target = int(math.ceil(quantile_value))
        target = max(self.ramp_floor_bars, min(self.ramp_cap_bars, target))
        self.ramp_bars_to_max = float(max(1, target))

    @staticmethod
    def select_quantile(values: Sequence[float], quantile: float) -> float:
        if not values:
            return float("nan")
        finite_values = [
            float(value) for value in values if math.isfinite(value)
        ]
        if not finite_values:
            return float("nan")
        clamped_quantile = min(max(quantile, 0.0), 1.0)
        finite_values.sort()
        if clamped_quantile <= 0.0:
            return finite_values[0]
        if clamped_quantile >= 1.0:
            return finite_values[-1]
        position = clamped_quantile * (len(finite_values) - 1)
        lower_index = int(math.floor(position))
        upper_index = int(math.ceil(position))
        if lower_index == upper_index:
            return finite_values[lower_index]
        interpolation_weight = position - lower_index
        lower_value = finite_values[lower_index]
        upper_value = finite_values[upper_index]
        return lower_value + (upper_value - lower_value) * interpolation_weight

    def gate_intensity_rsi(self) -> float:
        """RSI gate intensity in [0, 1]; stronger when RSI sits below the
        threshold.
        """
        ready = (
            self.use_rsi_gate
            and self.rsi
            and self.rsi.IsReady
            and len(self.rsi_values) >= self.min_gate_samples
        )
        if not ready:
            return 0.0
        cur = self.rsi.Current.Value
        if not (math.isfinite(cur) and cur <= self.rsi_threshold):
            return 0.0
        span = max(self.rsi_threshold, SPAN_FLOOR)
        return float(min(1.0, (self.rsi_threshold - cur) / span))

    def gate_intensity_bb(self) -> float:
        """BB gate intensity in [0, 1]; stronger when the Z-score falls below
        the threshold.
        """
        ready = (
            self.use_bb_gate
            and self.bollinger_bands
            and self.bollinger_bands.IsReady
            and len(self.bb_z_values) >= self.min_gate_samples
        )
        if not ready:
            return 0.0
        bollinger_z_score = self.current_bb_z()
        if not (
            math.isfinite(bollinger_z_score)
            and bollinger_z_score <= self.bb_z_threshold
        ):
            return 0.0
        span = max(abs(self.bb_z_threshold), SPAN_FLOOR)
        return float(
            min(
                1.0,
                (
                    self.bb_z_threshold
                    - bollinger_z_score
                )
                / span,
            )
        )

    def gate_intensity_ema(self) -> float:
        """EMA gate intensity in [0, 1]; stronger when price trades deeper
        below the EMA discount threshold.
        """
        ready = (
            self.use_ema_gate
            and self.ema_gate
            and self.ema_gate.IsReady
            and len(self.ema_diff_values) >= self.min_gate_samples
        )
        if not ready:
            return 0.0
        cur = self.current_ema_diff()
        if not (math.isfinite(cur) and cur <= self.ema_diff_threshold):
            return 0.0
        span = max(abs(self.ema_diff_threshold), SPAN_FLOOR)
        return float(min(1.0, (self.ema_diff_threshold - cur) / span))

    def gate_intensity_sma(self) -> float:
        """SMA gate intensity in [0, 1]; stronger when price trades deeper
        below the SMA discount threshold.
        """
        ready = (
            self.use_sma_gate
            and self.sma_gate
            and self.sma_gate.IsReady
            and len(self.sma_diff_values) >= self.min_gate_samples
        )
        if not ready:
            return 0.0
        cur = self.current_sma_diff()
        if not (math.isfinite(cur) and cur <= self.sma_diff_threshold):
            return 0.0
        span = max(abs(self.sma_diff_threshold), SPAN_FLOOR)
        return float(min(1.0, (self.sma_diff_threshold - cur) / span))

    def srsi_intensity(self) -> float:
        """SRSI sizing intensity; lower absolute readings drive larger
        multipliers.
        """
        ready = (
            self.use_srsi_sizing
            and self.srsi
            and self.srsi.IsReady
            and len(self.srsi_values) >= self.min_gate_samples
        )
        if not ready:
            return 0.0
        cur = self.srsi.Current.Value
        if not (math.isfinite(cur) and cur <= self.srsi_threshold):
            return 0.0
        span = max(self.srsi_threshold, SPAN_FLOOR)
        return float(min(1.0, (self.srsi_threshold - cur) / span))

    def scale_pressure_to_multiplier(
        self,
        pressure: float,
        min_mult: float,
        max_mult: float,
    ) -> float:
        """Map pressure in [0, ramp_bars_to_max] into [min_mult, max_mult]."""
        normalized_pressure = max(
            0.0, min(1.0, pressure / float(self.ramp_bars_to_max))
        )
        return float(
            min_mult + (max_mult - min_mult) * normalized_pressure
        )

    def atr_size_multiplier(self) -> float:
        """ATR-based sizing multiplier; lower volatility → larger tranches."""
        if not (self.use_atr_sizing and self.atr and self.atr.IsReady):
            return 1.0
        if len(self.atr_ratios) < max(
            ATR_HISTORY_MIN_SAMPLES, self.min_gate_samples
        ):
            return 1.0
        ratio = self.current_atr_ratio()
        if not math.isfinite(ratio):
            return 1.0
        atr_ratio_history = list(self.atr_ratios)
        if not atr_ratio_history:
            return 1.0
        volatility_percentile = self.percentile_rank(
            atr_ratio_history, ratio
        )
        if not math.isfinite(volatility_percentile):
            return 1.0
        # Low volatility (low ATR/price) increases size toward atr_max_mult.
        volatility_position = max(
            0.0, min(1.0, 1.0 - volatility_percentile)
        )
        multiplier_range = self.atr_max_mult - self.atr_min_mult
        return float(
            self.atr_min_mult + multiplier_range * volatility_position
        )

    def get_composite_size_multiplier(self) -> float:
        """Combine per-gate multipliers with the selected combiner and apply
        overall caps.
        """
        multipliers = []

        include_gates = not (
            self.use_atr_sizing and self.atr_mode == "standalone"
        )

        if include_gates:
            if self.use_rsi_gate:
                multipliers.append(
                    self.scale_pressure_to_multiplier(
                        self.rsi_pressure,
                        self.rsi_min_mult,
                        self.rsi_max_mult,
                    )
                )
            if self.use_bb_gate:
                multipliers.append(
                    self.scale_pressure_to_multiplier(
                        self.bb_pressure,
                        self.bb_min_mult,
                        self.bb_max_mult,
                    )
                )
            if self.use_ema_gate:
                multipliers.append(
                    self.scale_pressure_to_multiplier(
                        self.ema_pressure,
                        self.ema_min_mult,
                        self.ema_max_mult,
                    )
                )
            if self.use_sma_gate:
                multipliers.append(
                    self.scale_pressure_to_multiplier(
                        self.sma_pressure,
                        self.sma_min_mult,
                        self.sma_max_mult,
                    )
                )

        if self.use_srsi_sizing:
            multipliers.append(
                self.scale_pressure_to_multiplier(
                    self.srsi_pressure,
                    self.srsi_min_mult,
                    self.srsi_max_mult,
                )
            )
        if self.use_atr_sizing:
            multipliers.append(self.atr_size_multiplier())

        if not multipliers:
            return 1.0

        if self.combine_method == "max":
            combined_multiplier = max(multipliers)
        elif self.combine_method == "sum":
            combined_multiplier = 1.0 + sum(
                multiplier - 1.0 for multiplier in multipliers
            )
        else:  # "product"
            combined_multiplier = 1.0
            for multiplier in multipliers:
                combined_multiplier *= multiplier

        capped_multiplier = max(
            self.overall_min_mult,
            min(self.overall_max_mult, combined_multiplier),
        )
        return float(capped_multiplier)

    # Utilities
    def compute_next_cycle_time(self, reference_time: datetime) -> datetime:
        if self.dca_frequency_months > 0:
            return self.add_months(reference_time, self.dca_frequency_months)
        if self.dca_frequency_days > 0:
            return reference_time + timedelta(days=self.dca_frequency_days)
        if self.dca_frequency_hours > 0:
            return reference_time + timedelta(hours=self.dca_frequency_hours)
        return reference_time + timedelta(days=30)

    def add_months(
        self,
        reference_datetime: datetime,
        months_to_add: int,
    ) -> datetime:
        month_index = reference_datetime.month - 1 + months_to_add
        target_year = reference_datetime.year + month_index // 12
        target_month = month_index % 12 + 1
        days_in_target_month = calendar.monthrange(
            target_year,
            target_month,
        )[1]
        target_day = min(reference_datetime.day, days_in_target_month)
        return datetime(
            target_year,
            target_month,
            target_day,
            reference_datetime.hour,
            reference_datetime.minute,
            reference_datetime.second,
        )

    def update_distributions(self) -> None:
        rsi_ready = (
            self.use_rsi_gate
            and self.rsi is not None
            and self.rsi.IsReady
        )
        if rsi_ready:
            rsi_val = self.rsi.Current.Value
            if math.isfinite(rsi_val):
                self.rsi_values.append(float(rsi_val))

        bb_ready = (
            self.use_bb_gate
            and self.bollinger_bands is not None
            and self.bollinger_bands.IsReady
        )
        if bb_ready:
            bollinger_z_score = self.current_bb_z()
            if math.isfinite(bollinger_z_score):
                self.bb_z_values.append(float(bollinger_z_score))

        ema_ready = (
            self.use_ema_gate
            and self.ema_gate is not None
            and self.ema_gate.IsReady
        )
        if ema_ready:
            diff = self.current_ema_diff()
            if math.isfinite(diff):
                self.ema_diff_values.append(float(diff))

        sma_ready = (
            self.use_sma_gate
            and self.sma_gate is not None
            and self.sma_gate.IsReady
        )
        if sma_ready:
            diff = self.current_sma_diff()
            if math.isfinite(diff):
                self.sma_diff_values.append(float(diff))

        srsi_ready = (
            self.use_srsi_sizing
            and self.srsi is not None
            and self.srsi.IsReady
        )
        if srsi_ready:
            srsi_val = self.srsi.Current.Value
            if math.isfinite(srsi_val):
                self.srsi_values.append(float(srsi_val))

        atr_ready = (
            self.use_atr_sizing
            and self.atr is not None
            and self.atr.IsReady
        )
        if atr_ready:
            ratio = self.current_atr_ratio()
            if math.isfinite(ratio):
                self.atr_ratios.append(float(ratio))

    def current_bb_z(self) -> float:
        price = self.Securities[self.symbol].Price
        if not math.isfinite(price) or not self.bollinger_bands:
            return float("nan")
        upper = self.bollinger_bands.UpperBand.Current.Value
        lower = self.bollinger_bands.LowerBand.Current.Value
        mid = self.bollinger_bands.MiddleBand.Current.Value
        if not math.isfinite(mid):
            return float("nan")
        half_bandwidth = float("nan")
        if math.isfinite(upper):
            half_bandwidth = upper - mid
        insufficient_band = (
            not math.isfinite(half_bandwidth)
            or half_bandwidth <= 0
        )
        if insufficient_band and math.isfinite(lower):
            half_bandwidth = mid - lower
        valid_band = math.isfinite(half_bandwidth) and half_bandwidth > 0
        if not valid_band:
            std_dev = self.bollinger_bands.StandardDeviation.Current.Value
            if math.isfinite(std_dev) and self.bb_std_dev > 0:
                half_bandwidth = self.bb_std_dev * std_dev
        valid_band = math.isfinite(half_bandwidth) and half_bandwidth > 0
        if not valid_band:
            return float("nan")
        return (price - mid) / half_bandwidth

    def current_ema_diff(self) -> float:
        price = self.Securities[self.symbol].Price
        ema_value = (
            self.ema_gate.Current.Value
            if self.ema_gate
            else float("nan")
        )
        conditions_met = (
            math.isfinite(price)
            and math.isfinite(ema_value)
            and ema_value > 0
        )
        if not conditions_met:
            return float("nan")
        return (price / ema_value) - 1.0

    def current_sma_diff(self) -> float:
        price = self.Securities[self.symbol].Price
        sma_value = (
            self.sma_gate.Current.Value
            if self.sma_gate
            else float("nan")
        )
        conditions_met = (
            math.isfinite(price)
            and math.isfinite(sma_value)
            and sma_value > 0
        )
        if not conditions_met:
            return float("nan")
        return (price / sma_value) - 1.0

    def current_atr_ratio(self) -> float:
        if not self.atr or not self.atr.IsReady:
            return float("nan")
        price = self.Securities[self.symbol].Price
        atr_value = self.atr.Current.Value
        atr_conditions = (
            math.isfinite(price)
            and price > 0
            and math.isfinite(atr_value)
            and atr_value >= 0
        )
        if not atr_conditions:
            return float("nan")
        return atr_value / price

    @staticmethod
    def percentile_rank(values: list, target_value: float) -> float:
        if not values or not math.isfinite(target_value):
            return float("nan")
        finite_values = sorted(
            candidate_value
            for candidate_value in values
            if math.isfinite(candidate_value)
        )
        if not finite_values:
            return float("nan")
        below_count = 0
        tie_count = 0
        for candidate_value in finite_values:
            if candidate_value < target_value:
                below_count += 1
            elif candidate_value == target_value:
                tie_count += 1
        value_count = len(finite_values)
        return (
            below_count + PERCENTILE_TIE_WEIGHT * tie_count
        ) / value_count

    # Param helpers
    def get_bool_parameter(self, name: str, default: bool) -> bool:
        raw_value = (self.GetParameter(name) or "").strip().lower()
        if raw_value == "":
            return default
        truthy = {"true", "1", "yes", "on"}
        falsy = {"false", "0", "no", "off"}
        if raw_value in truthy:
            return True
        if raw_value in falsy:
            return False
        allowed_values = sorted(truthy | falsy)
        message = (
            f"{name} parameter '{raw_value}' must be one of {allowed_values}."
        )
        raise ValueError(message)

    def get_int_parameter(self, name: str, default: int) -> int:
        raw_value = (self.GetParameter(name) or "").strip()
        if raw_value == "":
            return int(default)
        try:
            return int(float(raw_value))
        except ValueError as error:
            message = f"{name} parameter '{raw_value}' must be integer."
            raise ValueError(message) from error

    def get_float_parameter(
        self,
        name: str,
        default: float,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
    ) -> float:
        raw_value = (self.GetParameter(name) or "").strip()
        if raw_value == "":
            value = float(default)
        else:
            try:
                value = float(raw_value)
            except ValueError as error:
                message = f"{name} parameter '{raw_value}' must be numeric."
                raise ValueError(message) from error
        if minimum is not None and value < minimum:
            message = f"{name} parameter {value} below minimum {minimum}."
            raise ValueError(message)
        if maximum is not None and value > maximum:
            message = f"{name} parameter {value} above maximum {maximum}."
            raise ValueError(message)
        return value
