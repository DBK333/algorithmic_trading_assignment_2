import math
import statistics

from AlgorithmImports import *
from QuantConnect.Data import RiskFreeInterestRateModelExtensions


DEFAULT_INITIAL_NAV = 1.0
NAV_UNIT_FLOOR = 1e-9
NAV_FLOOR = 1e-12
PERCENT_SCALE = 100.0
DEFAULT_TRADING_DAYS_PER_YEAR = 252
FRIDAY_WEEKDAY_INDEX = 4
DRAW_DOWN_PLOT_INTERVAL_DAYS = 7
MIN_SHARPE_SAMPLES = 2
MARKET_CLOSE_OFFSET_MINUTES = 0


class TimeWeightedReturnTracker:
    """Track NAV and drawdown telemetry for the algorithm.
    Manages time-weighted return math, plotting cadence, and runtime statistic.
    """
    def __init__(self, algorithm: QCAlgorithm, symbol: Symbol) -> None:
        self.algorithm = algorithm
        self.symbol = symbol
        self.nav = DEFAULT_INITIAL_NAV
        self.prev_nav = None
        self.units = 0.0
        self.peak_nav = DEFAULT_INITIAL_NAV
        self.max_drawdown = 0.0
        self.start_nav = None
        self.daily_nav = []
        self.daily_returns = []
        self.last_sample_date = None
        self.last_drawdown_plot = None
        self.current_drawdown = 0.0
        self.trading_days_per_year = self._get_trading_days_per_year()
        self.daily_risk_free = self._get_daily_risk_free_rate()

        nav_chart = Chart("TWR NAV")
        nav_chart.AddSeries(Series("NAV", SeriesType.Line, "", Color.Blue))
        algorithm.AddChart(nav_chart)

        dd_chart = Chart("TWR Drawdown %")
        dd_chart.AddSeries(
            Series("Drawdown %", SeriesType.Line, "%", Color.Red))
        algorithm.AddChart(dd_chart)

        algorithm.Schedule.On(
            algorithm.DateRules.EveryDay(symbol),
            algorithm.TimeRules.AfterMarketClose(
                symbol,
                MARKET_CLOSE_OFFSET_MINUTES,
            ),
            self.sample_end_of_day,
        )

        cash = float(algorithm.Portfolio.Cash)
        if cash > 0:
            self.register_cash_flow(cash)
        self.update()

    def register_cash_flow(self, amount: float) -> None:
        if amount == 0:
            return
        nav = max(self.nav, NAV_UNIT_FLOOR)
        self.units = max(0.0, self.units + (amount / nav))
        if self.start_nav is None and self.units > 0:
            self.start_nav = nav

    def update(self) -> None:
        algo = self.algorithm
        if self.units <= 0:
            self.nav = DEFAULT_INITIAL_NAV
            self.current_drawdown = 0.0
            return

        equity = float(algo.Portfolio.TotalPortfolioValue)
        prev_nav = self.nav
        self.nav = max(NAV_FLOOR, equity / self.units)
        self.prev_nav = prev_nav
        self.peak_nav = max(self.peak_nav, self.nav)
        self.current_drawdown = (self.nav / self.peak_nav) - 1.0
        self.max_drawdown = min(self.max_drawdown, self.current_drawdown)

    def sample_end_of_day(self) -> None:
        self.update()
        today = self.algorithm.Time.date()
        if self.units <= 0:
            self.last_sample_date = today
            return
        if self.last_sample_date == today:
            return
        self.last_sample_date = today
        self.daily_nav.append(self.nav)
        if len(self.daily_nav) >= MIN_SHARPE_SAMPLES:
            prev = self.daily_nav[-2]
            if prev > 0:
                self.daily_returns.append((self.nav / prev) - 1.0)
        self.plot_daily_nav()
        self.plot_weekly_drawdown(today)
        self.update_runtime_statistics()

    def update_runtime_statistics(self) -> None:
        algo = self.algorithm
        if self.daily_nav and self.start_nav is not None:
            twr_return = (self.nav / self.daily_nav[0]) - 1.0
            return_pct = twr_return * PERCENT_SCALE
            algo.SetRuntimeStatistic("Return % (TWR)", f"{return_pct:0.2f}%")
        drawdown_pct = self.max_drawdown * PERCENT_SCALE
        algo.SetRuntimeStatistic("Max Drawdown (TWR)", f"{drawdown_pct:0.2f}%")
        if len(self.daily_returns) >= MIN_SHARPE_SAMPLES:
            excess_returns = [
                r - self.daily_risk_free for r in self.daily_returns
            ]
            mean = statistics.mean(excess_returns)
            stdev = statistics.stdev(excess_returns)
            if stdev > 0:
                ann_mean = mean * self.trading_days_per_year
                ann_std = stdev * math.sqrt(self.trading_days_per_year)
                if ann_std > 0:
                    sharpe_twr = ann_mean / ann_std
                    formatted_sharpe = f"{sharpe_twr:0.2f}"
                    algo.SetRuntimeStatistic("Sharpe (TWR, daily)",
                                             formatted_sharpe)
                    algo.SetSummaryStatistic("Sharpe (TWR, daily)",
                                             formatted_sharpe)

    def plot_daily_nav(self) -> None:
        nav_value = self.nav if self.units > 0 else DEFAULT_INITIAL_NAV
        self.algorithm.Plot("TWR NAV", "NAV", nav_value)

    def plot_weekly_drawdown(self, today) -> None:
        if self.units <= 0:
            value = 0.0
        else:
            value = self.current_drawdown * PERCENT_SCALE
        should_plot = False
        if self.last_drawdown_plot is None:
            should_plot = True
        elif today.weekday() == FRIDAY_WEEKDAY_INDEX:
            should_plot = True
        elif (
            (today - self.last_drawdown_plot).days
            >= DRAW_DOWN_PLOT_INTERVAL_DAYS
        ):
            should_plot = True
        if not should_plot:
            return
        self.algorithm.Plot("TWR Drawdown %", "Drawdown %", value)
        self.last_drawdown_plot = today

    def _get_trading_days_per_year(self) -> int:
        settings = getattr(self.algorithm, "Settings", None)
        trading_days = getattr(
            settings, "TradingDaysPerYear", DEFAULT_TRADING_DAYS_PER_YEAR)
        try:
            value = int(trading_days)
        except (TypeError, ValueError):
            value = DEFAULT_TRADING_DAYS_PER_YEAR
        return max(1, value)

    def _get_daily_risk_free_rate(self) -> float:
        model = getattr(self.algorithm, "RiskFreeInterestRateModel", None)
        start = getattr(self.algorithm, "StartDate", None)
        end = getattr(self.algorithm, "EndDate", None)
        if not model or not start or not end:
            return 0.0
        try:
            annualized = RiskFreeInterestRateModelExtensions.GetRiskFreeRate(
                model,
                start,
                end,
            )
        except Exception:
            annualized = 0.0
        return float(annualized) / float(self.trading_days_per_year)
