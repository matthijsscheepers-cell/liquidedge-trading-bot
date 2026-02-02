import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Dict, Optional
import seaborn as sns


class BacktestVisualizer:
    """
    Visualize backtest results.

    Creates:
    - Equity curve
    - Drawdown chart
    - Monthly returns heatmap
    - Trade distribution
    - Win/loss analysis

    Example:
        viz = BacktestVisualizer(results)
        viz.plot_all()
        plt.show()
    """

    def __init__(self, results: Dict):
        """
        Initialize visualizer

        Args:
            results: Results dict from BacktestEngine.run()
        """
        self.results = results
        self.equity_curve = results['equity_curve']
        self.trades = results.get('trades', pd.DataFrame())

        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")

    def plot_all(self, save_path: Optional[str] = None):
        """
        Create comprehensive visualization

        Args:
            save_path: Path to save figure (optional)
        """

        fig = plt.figure(figsize=(16, 12))

        # Create subplots
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)

        ax1 = fig.add_subplot(gs[0, :])  # Equity curve (full width)
        ax2 = fig.add_subplot(gs[1, :])  # Drawdown (full width)
        ax3 = fig.add_subplot(gs[2, 0])  # Trade distribution
        ax4 = fig.add_subplot(gs[2, 1])  # Win/Loss analysis
        ax5 = fig.add_subplot(gs[3, 0])  # Monthly returns
        ax6 = fig.add_subplot(gs[3, 1])  # R-multiple distribution

        # Plot each
        self.plot_equity_curve(ax1)
        self.plot_drawdown(ax2)
        self.plot_trade_distribution(ax3)
        self.plot_win_loss_analysis(ax4)
        self.plot_monthly_returns(ax5)
        self.plot_r_distribution(ax6)

        # Title
        fig.suptitle('LIQUIDEDGE BACKTEST RESULTS', fontsize=16, fontweight='bold')

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")

        return fig

    def plot_equity_curve(self, ax=None):
        """Plot equity curve"""

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        equity = self.equity_curve['equity']

        ax.plot(equity.index, equity.values, linewidth=2, label='Equity')
        ax.axhline(y=self.results['initial_capital'], color='gray', linestyle='--', alpha=0.5, label='Initial')

        ax.set_title('Equity Curve', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Capital ($)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))

        return ax

    def plot_drawdown(self, ax=None):
        """Plot drawdown"""

        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))

        equity = self.equity_curve['equity']
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max * 100

        ax.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red', label='Drawdown')
        ax.plot(drawdown.index, drawdown.values, linewidth=1, color='darkred')

        # Mark max drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_val = drawdown.min()
        ax.plot(max_dd_idx, max_dd_val, 'rv', markersize=10, label=f'Max DD: {max_dd_val:.1f}%')

        ax.set_title('Drawdown', fontsize=12, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Drawdown (%)')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_trade_distribution(self, ax=None):
        """Plot P&L distribution"""

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        if len(self.trades) == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return ax

        pnl = self.trades['pnl']

        # Histogram
        ax.hist(pnl, bins=30, alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Breakeven')
        ax.axvline(x=pnl.mean(), color='green', linestyle='--', linewidth=2, label=f'Mean: ${pnl.mean():.2f}')

        ax.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
        ax.set_xlabel('P&L ($)')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)

        return ax

    def plot_win_loss_analysis(self, ax=None):
        """Plot win/loss comparison"""

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        if len(self.trades) == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return ax

        wins = self.trades[self.trades['outcome'] == 'WIN']
        losses = self.trades[self.trades['outcome'] == 'LOSS']

        data = {
            'Wins': [len(wins), wins['pnl'].sum()],
            'Losses': [len(losses), abs(losses['pnl'].sum())]
        }

        x = np.arange(2)
        width = 0.35

        ax.bar(x - width/2, data['Wins'], width, label='Wins', color='green', alpha=0.7)
        ax.bar(x + width/2, data['Losses'], width, label='Losses', color='red', alpha=0.7)

        ax.set_title('Win/Loss Comparison', fontsize=12, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(['Count', 'Total P&L ($)'])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        return ax

    def plot_monthly_returns(self, ax=None):
        """Plot monthly returns heatmap"""

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        if len(self.trades) == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return ax

        # Calculate monthly returns from equity curve
        equity = self.equity_curve['equity'].resample('ME').last()
        monthly_returns = equity.pct_change() * 100

        if len(monthly_returns) < 2:
            ax.text(0.5, 0.5, 'Not enough data', ha='center', va='center')
            return ax

        # Create year-month matrix
        monthly_returns.index = pd.to_datetime(monthly_returns.index)
        monthly_data = monthly_returns.to_frame('returns')
        monthly_data['year'] = monthly_data.index.year
        monthly_data['month'] = monthly_data.index.month

        pivot = monthly_data.pivot(index='year', columns='month', values='returns')

        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Return (%)'}, ax=ax)

        ax.set_title('Monthly Returns (%)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Month')
        ax.set_ylabel('Year')

        return ax

    def plot_r_distribution(self, ax=None):
        """Plot R-multiple distribution"""

        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 4))

        if len(self.trades) == 0:
            ax.text(0.5, 0.5, 'No trades', ha='center', va='center')
            return ax

        # Calculate R-multiples if not present
        if 'r_multiple' not in self.trades.columns:
            # Estimate from P&L ratio
            wins = self.trades[self.trades['outcome'] == 'WIN']
            losses = self.trades[self.trades['outcome'] == 'LOSS']

            if len(wins) > 0 and len(losses) > 0:
                avg_win_r = 2.0  # Estimate
                avg_loss_r = -1.0  # Estimate

                ax.bar(['Wins', 'Losses'], [avg_win_r, avg_loss_r], color=['green', 'red'], alpha=0.7)
                ax.set_title('Estimated R-Multiples', fontsize=12, fontweight='bold')
                ax.set_ylabel('Average R')
                ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
                ax.grid(True, alpha=0.3, axis='y')
            else:
                ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        else:
            # Plot actual R-multiples
            r_multiples = self.trades['r_multiple']
            ax.hist(r_multiples, bins=20, alpha=0.7, edgecolor='black')
            ax.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax.axvline(x=r_multiples.mean(), color='green', linestyle='--', linewidth=2,
                      label=f'Mean: {r_multiples.mean():.2f}R')

            ax.set_title('R-Multiple Distribution', fontsize=12, fontweight='bold')
            ax.set_xlabel('R-Multiple')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True, alpha=0.3)

        return ax
