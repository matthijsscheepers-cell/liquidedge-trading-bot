"""
LiquidEdge Backtest met Databento Micro Futures Data

Test de strategie op 16 jaar professionele futures data!
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.backtest.engine import BacktestEngine
from src.backtest.performance import PerformanceCalculator
from src.backtest.visualizer import BacktestVisualizer
from src.regime.detector import RegimeDetector
import pandas as pd

print("=" * 70)
print("LIQUIDEDGE BACKTEST - DATABENTO MICRO FUTURES")
print("=" * 70)
print()

# ============================================================================
# CONFIGURATIE
# ============================================================================

# Backtest periode
START_DATE = '2025-01-01'  # Start: 1 jaar recent
END_DATE = '2026-01-29'    # Tot nu

# Of test op langere periode (kies één):
# START_DATE = '2024-01-01'  # 2 jaar
# START_DATE = '2020-01-01'  # 5 jaar
# START_DATE = '2015-01-01'  # 10 jaar
# START_DATE = '2010-09-12'  # VOLLEDIGE 16 JAAR!

INITIAL_CAPITAL = 10000.0
TIMEFRAME = '15min'  # 15-minuten bars (zoals je strategie verwacht)

ASSETS = ['GOLD', 'SILVER', 'US100', 'US500']

print("Backtest configuratie:")
print(f"  Periode: {START_DATE} tot {END_DATE}")
print(f"  Kapitaal: ${INITIAL_CAPITAL:,.0f}")
print(f"  Timeframe: {TIMEFRAME}")
print(f"  Assets: {', '.join(ASSETS)}")
print()

# ============================================================================
# STAP 1: DATA LADEN
# ============================================================================

print("=" * 70)
print("STAP 1: DATA LADEN")
print("=" * 70)
print()

loader = DatabentoMicroFuturesLoader()

# Laad data voor alle assets
all_data = loader.load_all_symbols(
    start_date=START_DATE,
    end_date=END_DATE,
    resample=TIMEFRAME
)

print()
print("Data geladen:")
for symbol, df in all_data.items():
    if not df.empty:
        days = (df.index[-1] - df.index[0]).days
        print(f"  {symbol:10s}: {len(df):6d} bars, {days:4d} dagen, "
              f"${df['close'].iloc[-1]:8.2f} huidige prijs")

if not all_data or all(df.empty for df in all_data.values()):
    print()
    print("⚠ GEEN DATA GELADEN - Controleer de datums")
    sys.exit(1)

print()

# ============================================================================
# STAP 2: INDICATOREN TOEVOEGEN
# ============================================================================

print("=" * 70)
print("STAP 2: INDICATOREN TOEVOEGEN")
print("=" * 70)
print()

detector = RegimeDetector()

print("Berekenen van indicatoren (ADX, EMA, ATR, RSI, etc.)...")
for symbol in all_data.keys():
    if not all_data[symbol].empty:
        print(f"  {symbol}...", end=" ")
        all_data[symbol] = detector.add_all_indicators(all_data[symbol])
        print("✓")

print()
print("Indicatoren toegevoegd:")
print(f"  {list(all_data['GOLD'].columns)}")
print()

# ============================================================================
# STAP 3: BACKTEST UITVOEREN
# ============================================================================

print("=" * 70)
print("STAP 3: BACKTEST UITVOEREN")
print("=" * 70)
print()

print(f"[BACKTEST] Geïnitialiseerd met ${INITIAL_CAPITAL:,.2f}")
print(f"[BACKTEST] Assets: {', '.join([k for k, v in all_data.items() if not v.empty])}")
print(f"[BACKTEST] Data range: {START_DATE} tot {END_DATE}")
print()

# Maak backtest engine
engine = BacktestEngine(
    data=all_data,
    initial_capital=INITIAL_CAPITAL,
    assets=[k for k, v in all_data.items() if not v.empty],
    commission_pct=0.001,
    slippage_pct=0.001
)

print()
print("=" * 60)
print("BACKTEST GESTART")
print("=" * 60)
print()

# Run backtest
results = engine.run()

print()
print("=" * 60)
print("BACKTEST VOLTOOID")
print("=" * 60)
print()

# ============================================================================
# STAP 4: RESULTATEN
# ============================================================================

print()
print("=" * 70)
print("BACKTEST RESULTATEN")
print("=" * 70)
print()

# Print resultaten
calc = PerformanceCalculator(
    results['equity_curve'],
    results['trades'],
    INITIAL_CAPITAL
)
calc.print_summary(results)

# ============================================================================
# STAP 5: VISUALISATIE
# ============================================================================

print()
print("=" * 70)
print("VISUALISATIE")
print("=" * 70)
print()

# Genereer charts
output_file = "data/backtest_results/databento_backtest.png"
print(f"Genereren van grafieken...")

try:
    viz = BacktestVisualizer(results)
    viz.plot_all(save_path=output_file)
    print(f"✓ Chart opgeslagen: {output_file}")
except Exception as e:
    print(f"⚠ Fout bij genereren chart: {e}")

print()
print("=" * 70)
print("SAMENVATTING")
print("=" * 70)
print()

if results:
    print(f"📊 Return: {results['metrics']['total_return']:.2f}%")
    print(f"🎯 Trades: {results['metrics']['total_trades']}")
    print(f"✅ Win Rate: {results['metrics']['win_rate']:.1f}%")
    print(f"💰 Profit Factor: {results['metrics']['profit_factor']:.2f}")
    print(f"📉 Max Drawdown: {results['metrics']['max_drawdown']:.2f}%")
    print(f"📈 Sharpe Ratio: {results['metrics']['sharpe_ratio']:.2f}")

print()
print("=" * 70)
print("BACKTEST COMPLEET!")
print("=" * 70)
print()

# ============================================================================
# EXTRA: TRADE ANALYSE
# ============================================================================

if results and results['trades']:
    print()
    print("=" * 70)
    print("TRADE DETAILS (Laatste 10)")
    print("=" * 70)
    print()

    trades_df = pd.DataFrame(results['trades'])

    # Toon laatste 10 trades
    if len(trades_df) > 0:
        # Selecteer belangrijke kolommen
        cols = ['entry_time', 'exit_time', 'symbol', 'direction',
                'entry_price', 'exit_price', 'pnl', 'pnl_pct', 'bars_held']

        display_cols = [col for col in cols if col in trades_df.columns]

        print(trades_df[display_cols].tail(10).to_string(index=False))
        print()

        # Analyse per asset
        print()
        print("Prestaties per asset:")
        print("-" * 70)

        for symbol in ASSETS:
            symbol_trades = trades_df[trades_df['symbol'] == symbol]

            if len(symbol_trades) > 0:
                wins = symbol_trades[symbol_trades['pnl'] > 0]
                win_rate = len(wins) / len(symbol_trades) * 100
                avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
                avg_loss = symbol_trades[symbol_trades['pnl'] < 0]['pnl'].mean()

                print(f"{symbol:10s}: {len(symbol_trades):3d} trades, "
                      f"{win_rate:5.1f}% win rate, "
                      f"Avg win: ${avg_win:7.2f}, "
                      f"Avg loss: ${avg_loss:7.2f}")

print()
print("=" * 70)
print("TIP: Wijzig START_DATE in het script om andere periodes te testen:")
print("  - '2025-01-01' = 1 jaar (snel)")
print("  - '2020-01-01' = 5 jaar (goed overzicht)")
print("  - '2010-09-12' = 16 JAAR (volledige dataset!)")
print("=" * 70)
