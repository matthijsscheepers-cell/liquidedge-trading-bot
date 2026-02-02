"""
Vergelijk alle data bronnen voor backtesting
"""

import sys
sys.path.insert(0, '.')

from src.execution.databento_loader import DatabentoMicroFuturesLoader
from src.execution.capital_connector import CapitalConnector
import pandas as pd

print("=" * 70)
print("DATA BRONNEN VERGELIJKING")
print("=" * 70)
print()

# ============================================================================
# 1. DATABENTO MICRO FUTURES (NIEUWE DATA!)
# ============================================================================

print("1. DATABENTO MICRO FUTURES")
print("-" * 70)
print()

db_loader = DatabentoMicroFuturesLoader()
stats = db_loader.summary_stats()

print("Dataset informatie:")
print(f"  Periode: {stats['start_date']} tot {stats['end_date']}")
print(f"  Duur: {stats['duration_years']:.1f} jaar")
print(f"  Grootte: {stats['file_size_mb']:.1f} MB")
print(f"  Symbolen: {', '.join(stats['symbols'])}")
print()

# Laad laatste maand voor elk symbool
print("Laden laatste 30 dagen (15-min bars)...")
all_data = db_loader.load_all_symbols(
    start_date='2026-01-01',
    end_date='2026-01-29',
    resample='15min'
)

print()
for symbol, df in all_data.items():
    if not df.empty:
        print(f"  {symbol:10s}: {len(df):5d} bars, "
              f"${df['close'].iloc[-1]:8.2f} huidige prijs, "
              f"${df['close'].min():8.2f} - ${df['close'].max():8.2f} range")

print()
print("Voordelen:")
print("  ✓ 16 jaar historie (2010-2026)")
print("  ✓ 1-minuut granulariteit (resample naar 15m, 1H, 1D)")
print("  ✓ 16 miljoen data punten")
print("  ✓ Alle belangrijke assets (GOLD, SILVER, US100, US500)")
print("  ✓ Professionele futures data van CME Globex")
print("  ✓ Geen API rate limits")
print()
print("Nadelen:")
print("  ✗ Grote bestand (204 MB)")
print("  ✗ Laden duurt ~30 seconden")
print()

# ============================================================================
# 2. CAPITAL.COM API
# ============================================================================

print()
print("2. CAPITAL.COM API")
print("-" * 70)
print()

config = {
    'api_key': 'jvJoOhauq6w7Yub0',
    'password': 'Vergeten22!',
    'identifier': 'matthijsscheepers@gmail.com'
}

connector = CapitalConnector(config)
try:
    connector.connect()

    # Laad GOLD data
    gold_capital = connector.get_historical_data('GOLD', '15m', 500)

    print(f"GOLD data geladen:")
    print(f"  Bars: {len(gold_capital)}")
    print(f"  Periode: {gold_capital.index[0]} tot {gold_capital.index[-1]}")
    print(f"  Prijs: ${gold_capital['close'].iloc[-1]:.2f}")
    print()

    print("Voordelen:")
    print("  ✓ Real-time data (altijd actueel)")
    print("  ✓ Snel laden (< 5 seconden)")
    print("  ✓ Meerdere assets beschikbaar")
    print("  ✓ Geschikt voor live trading")
    print()
    print("Nadelen:")
    print("  ✗ Beperkte historie (500 bars = ~5 dagen op 15m)")
    print("  ✗ API key vereist")
    print("  ✗ Rate limits mogelijk")
    print()

except Exception as e:
    print(f"⚠ Capital.com API fout: {e}")
    print()

# ============================================================================
# 3. KAGGLE DAILY DATA
# ============================================================================

print()
print("3. KAGGLE DAILY DATA")
print("-" * 70)
print()

try:
    from src.execution.kaggle_daily_loader import KaggleDailyGoldLoader

    kaggle_loader = KaggleDailyGoldLoader()
    kaggle_data = kaggle_loader.load_data()

    print(f"Dataset informatie:")
    print(f"  Bars: {len(kaggle_data)}")
    print(f"  Periode: {kaggle_data.index[0]} tot {kaggle_data.index[-1]}")
    print(f"  Prijs: ${kaggle_data['close'].iloc[-1]:.2f}")
    print()

    print("Voordelen:")
    print("  ✓ Dagelijkse OHLCV data")
    print("  ✓ 1 jaar recente data (2025-2026)")
    print("  ✓ Gratis en snel")
    print()
    print("Nadelen:")
    print("  ✗ Alleen dagelijks (geen intraday)")
    print("  ✗ Beperkte periode (1 jaar)")
    print("  ✗ Alleen GOLD")
    print()

except Exception as e:
    print(f"⚠ Kaggle data niet beschikbaar: {e}")
    print()

# ============================================================================
# AANBEVELING
# ============================================================================

print()
print("=" * 70)
print("AANBEVELING VOOR LIQUIDEDGE 15-MINUTEN STRATEGIE")
print("=" * 70)
print()

print("🥇 PRIMAIR: DATABENTO MICRO FUTURES")
print()
print("  Waarom:")
print("  • 16 jaar historie voor uitgebreid backtesten")
print("  • 1-minuut data → perfect voor 15m resample")
print("  • Alle assets (GOLD, SILVER, US100, US500)")
print("  • Professionele kwaliteit data")
print("  • Geen API limits")
print()
print("  Gebruik voor:")
print("  • Hoofd backtesting (2010-2026)")
print("  • Strategy development en optimalisatie")
print("  • Multi-year performance analysis")
print()

print("🥈 SECUNDAIR: CAPITAL.COM API")
print()
print("  Waarom:")
print("  • Real-time data (altijd actueel)")
print("  • Snel voor recent testen")
print("  • Live trading ready")
print()
print("  Gebruik voor:")
print("  • Recente markt validatie")
print("  • Live trading")
print("  • Quick checks van strategy")
print()

print("=" * 70)
print()

# ============================================================================
# PRESTATIE VERGELIJKING
# ============================================================================

print("PRESTATIE VERGELIJKING")
print("-" * 70)
print()

comparison = pd.DataFrame({
    'Bron': ['Databento', 'Capital.com', 'Kaggle Daily'],
    'Timeframe': ['1min → 15min', '15min', '1D'],
    'Historie': ['16 jaar', '~7 dagen', '1 jaar'],
    'Bars beschikbaar': ['Miljoenen', '~500', '~250'],
    'Assets': ['4 (GOLD,SILVER,US100,US500)', '4+', '1 (GOLD)'],
    'Laadtijd': ['~30 sec', '~5 sec', '<1 sec'],
    'Geschikt voor 15m': ['✓✓✓', '✓✓', '✗'],
})

print(comparison.to_string(index=False))
print()

print("=" * 70)
print("CONCLUSIE: Gebruik Databento voor je backtests!")
print("=" * 70)
