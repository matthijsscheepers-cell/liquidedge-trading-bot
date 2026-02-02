import numpy as np
import pandas as pd
from typing import Dict, Optional
from datetime import datetime


class PerformanceCalculator:
    """
    Calculate trading performance metrics.

    Metrics included:
    - Total return
    - Sharpe ratio
    - Sortino ratio
    - Maximum drawdown
    - Win rate
    - Profit factor
    - Average R-multiple
    - Expectancy
    - Recovery factor
    - Calmar ratio

    Example:
        calc = PerformanceCalculator(equity_df, trades_df, 10000)
        results = calc.calculate_all()
        print(f"Sharpe: {results['sharpe_ratio']:.2f}")
    """

    def __init__(self,
                 equity_curve: pd.DataFrame,
                 trades: pd.DataFrame,
                 initial_capital: float):
        """
        Initialize performance calculator

        Args:
            equity_curve: DataFrame with 'equity' column
            trades: DataFrame with trade records
            initial_capital: Starting capital
        """
        self.equity_curve = equity_curve
        self.trades = trades
        self.initial_capital = initial_capital

        # Calculate returns
        self.equity_curve['returns'] = self.equity_curve['equity'].pct_change()
        self.equity_curve['cumulative_returns'] = (1 + self.equity_curve['returns']).cumprod()

    def calculate_all(self) -> Dict:
        """
        Calculate all performance metrics

        Returns:
            Dict with all metrics
        """

        results = {}

        # Basic metrics
        results['initial_capital'] = self.initial_capital
        results['final_capital'] = self.equity_curve['equity'].iloc[-1]
        results['total_return'] = self.calculate_total_return()
        results['total_return_pct'] = results['total_return'] * 100

        # Risk-adjusted returns
        results['sharpe_ratio'] = self.calculate_sharpe_ratio()
        results['sortino_ratio'] = self.calculate_sortino_ratio()
        results['calmar_ratio'] = self.calculate_calmar_ratio()

        # Drawdown metrics
        dd_metrics = self.calculate_drawdown_metrics()
        results.update(dd_metrics)

        # Trade metrics
        if len(self.trades) > 0:
            trade_metrics = self.calculate_trade_metrics()
            results.update(trade_metrics)
        else:
            results['total_trades'] = 0
            results['win_rate'] = 0
            results['profit_factor'] = 0

        # Time metrics
        results['total_days'] = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        results['trading_days'] = len(self.equity_curve)

        return results

    def calculate_total_return(self) -> float:
        """Calculate total return"""
        return (self.equity_curve['equity'].iloc[-1] - self.initial_capital) / self.initial_capital

    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio

        Args:
            risk_free_rate: Annual risk-free rate (default 2%)

        Returns:
            Sharpe ratio (annualized)
        """

        returns = self.equity_curve['returns'].dropna()

        if len(returns) == 0 or returns.std() == 0:
            return 0.0

        # Annualize (assuming hourly bars → 252 trading days * 6.5 hours)
        periods_per_year = 252 * 6.5

        excess_returns = returns.mean() - (risk_free_rate / periods_per_year)
        sharpe = excess_returns / returns.std() * np.sqrt(periods_per_year)

        return sharpe

    def calculate_sortino_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sortino ratio (uses downside deviation)

        Args:
            risk_free_rate: Annual risk-free rate

        Returns:
            Sortino ratio (annualized)
        """

        returns = self.equity_curve['returns'].dropna()

        if len(returns) == 0:
            return 0.0

        # Downside deviation (only negative returns)
        downside_returns = returns[returns < 0]

        if len(downside_returns) == 0:
            return float('inf')

        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0.0

        periods_per_year = 252 * 6.5
        excess_returns = returns.mean() - (risk_free_rate / periods_per_year)
        sortino = excess_returns / downside_std * np.sqrt(periods_per_year)

        return sortino

    def calculate_drawdown_metrics(self) -> Dict:
        """
        Calculate drawdown metrics

        Returns:
            Dict with drawdown stats
        """

        equity = self.equity_curve['equity']

        # Calculate running maximum
        running_max = equity.expanding().max()

        # Calculate drawdown
        drawdown = (equity - running_max) / running_max

        # Maximum drawdown
        max_drawdown = drawdown.min()
        max_dd_pct = max_drawdown * 100

        # Drawdown duration
        # Find periods in drawdown
        in_drawdown = drawdown < 0

        if in_drawdown.any():
            # Find longest drawdown period
            drawdown_periods = []
            current_period = 0

            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0

            if current_period > 0:
                drawdown_periods.append(current_period)

            max_dd_duration = max(drawdown_periods) if drawdown_periods else 0
        else:
            max_dd_duration = 0

        # Recovery factor (total return / max drawdown)
        total_return = self.calculate_total_return()
        recovery_factor = abs(total_return / max_drawdown) if max_drawdown != 0 else 0

        return {
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_dd_pct,
            'max_drawdown_duration': max_dd_duration,
            'recovery_factor': recovery_factor
        }

    def calculate_calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (return / max drawdown)

        Returns:
            Calmar ratio
        """

        total_return = self.calculate_total_return()
        max_dd = self.calculate_drawdown_metrics()['max_drawdown']

        if max_dd == 0:
            return 0.0

        # Annualize return
        days = (self.equity_curve.index[-1] - self.equity_curve.index[0]).days
        years = days / 365.25
        annual_return = (1 + total_return) ** (1 / years) - 1

        calmar = annual_return / abs(max_dd)

        return calmar

    def calculate_trade_metrics(self) -> Dict:
        """
        Calculate trade-specific metrics

        Returns:
            Dict with trade stats
        """

        if len(self.trades) == 0:
            return {}

        # Basic counts
        total_trades = len(self.trades)
        wins = self.trades[self.trades['outcome'] == 'WIN']
        losses = self.trades[self.trades['outcome'] == 'LOSS']

        # Win rate
        win_rate = len(wins) / total_trades if total_trades > 0 else 0

        # Average win/loss
        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0

        avg_win_pct = (wins['pnl'] / self.initial_capital * 100).mean() if len(wins) > 0 else 0
        avg_loss_pct = (losses['pnl'] / self.initial_capital * 100).mean() if len(losses) > 0 else 0

        # Profit factor
        total_wins = wins['pnl'].sum()
        total_losses = abs(losses['pnl'].sum())

        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')

        # R-multiples (if available)
        if 'r_multiple' in self.trades.columns:
            avg_r = self.trades['r_multiple'].mean()
            win_r = wins['r_multiple'].mean() if len(wins) > 0 else 0
            loss_r = losses['r_multiple'].mean() if len(losses) > 0 else 0
        else:
            # Estimate from P&L
            avg_r = 0
            win_r = 0
            loss_r = 0

        # Expectancy
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
        expectancy_pct = expectancy / self.initial_capital * 100

        # Consecutive wins/losses
        consecutive_wins = self._calculate_max_consecutive(self.trades, 'WIN')
        consecutive_losses = self._calculate_max_consecutive(self.trades, 'LOSS')

        # Largest win/loss
        largest_win = wins['pnl'].max() if len(wins) > 0 else 0
        largest_loss = losses['pnl'].min() if len(losses) > 0 else 0

        return {
            'total_trades': total_trades,
            'winning_trades': len(wins),
            'losing_trades': len(losses),
            'win_rate': win_rate * 100,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_win_pct': avg_win_pct,
            'avg_loss_pct': avg_loss_pct,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'expectancy_pct': expectancy_pct,
            'avg_r_multiple': avg_r,
            'avg_win_r': win_r,
            'avg_loss_r': loss_r,
            'max_consecutive_wins': consecutive_wins,
            'max_consecutive_losses': consecutive_losses,
            'largest_win': largest_win,
            'largest_loss': largest_loss
        }

    def _calculate_max_consecutive(self, trades: pd.DataFrame, outcome: str) -> int:
        """Calculate maximum consecutive wins or losses"""

        if len(trades) == 0:
            return 0

        max_consecutive = 0
        current_consecutive = 0

        for _, trade in trades.iterrows():
            if trade['outcome'] == outcome:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def print_summary(self, results: Dict):
        """
        Print formatted summary

        Args:
            results: Results dict from calculate_all()
        """

        print("\n" + "="*60)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("="*60)

        print(f"\n📊 RETURNS:")
        print(f"   Initial Capital:     ${results['initial_capital']:,.2f}")
        print(f"   Final Capital:       ${results['final_capital']:,.2f}")
        print(f"   Total Return:        {results['total_return_pct']:+.2f}%")
        print(f"   Total Days:          {results['total_days']}")

        print(f"\n📈 RISK-ADJUSTED:")
        print(f"   Sharpe Ratio:        {results['sharpe_ratio']:.2f}")
        print(f"   Sortino Ratio:       {results['sortino_ratio']:.2f}")
        print(f"   Calmar Ratio:        {results['calmar_ratio']:.2f}")

        print(f"\n📉 DRAWDOWN:")
        print(f"   Max Drawdown:        {results['max_drawdown_pct']:.2f}%")
        print(f"   Max DD Duration:     {results['max_drawdown_duration']} bars")
        print(f"   Recovery Factor:     {results['recovery_factor']:.2f}")

        if results['total_trades'] > 0:
            print(f"\n🎯 TRADES:")
            print(f"   Total Trades:        {results['total_trades']}")
            print(f"   Win Rate:            {results['win_rate']:.1f}%")
            print(f"   Profit Factor:       {results['profit_factor']:.2f}")
            print(f"   Expectancy:          {results['expectancy_pct']:+.3f}%")

            print(f"\n💰 WIN/LOSS:")
            print(f"   Avg Win:             ${results['avg_win']:+,.2f} ({results['avg_win_pct']:+.2f}%)")
            print(f"   Avg Loss:            ${results['avg_loss']:+,.2f} ({results['avg_loss_pct']:+.2f}%)")
            print(f"   Largest Win:         ${results['largest_win']:+,.2f}")
            print(f"   Largest Loss:        ${results['largest_loss']:+,.2f}")

            print(f"\n🔄 STREAKS:")
            print(f"   Max Consecutive Wins:   {results['max_consecutive_wins']}")
            print(f"   Max Consecutive Losses: {results['max_consecutive_losses']}")

        print("\n" + "="*60)
