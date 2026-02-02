"""
Full connection test for Capital.com API
"""
import os
import sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
from src.execution.capital_connector import CapitalConnector

load_dotenv()

config = {
    'api_key': os.getenv('CAPITAL_API_KEY'),
    'identifier': os.getenv('CAPITAL_IDENTIFIER'),
    'password': os.getenv('CAPITAL_PASSWORD'),
    'environment': 'demo'
}

print('=' * 70)
print('FULL CONNECTION TEST')
print('=' * 70)
print()

broker = CapitalConnector(config)
broker.connect()

print('Account Info:')
account = broker.get_account_info()
print(f'  Balance:     €{account.balance:,.2f}')
print(f'  Equity:      €{account.equity:,.2f}')
print(f'  Currency:    {account.currency}')
print()

print('Testing historical data (GOLD 15min, last 50 bars)...')
df = broker.get_historical_data('GOLD', '15m', 50)
print(f'  ✓ Received {len(df)} bars')
print(f'  Latest close: ${df["close"].iloc[-1]:.2f}')
print(f'  Latest time:  {df.index[-1]}')
print()

print('Testing historical data (US_TECH_100 1H, last 50 bars)...')
df_us100 = broker.get_historical_data('US_TECH_100', '1H', 50)
print(f'  ✓ Received {len(df_us100)} bars')
print(f'  Latest close: ${df_us100["close"].iloc[-1]:.2f}')
print()

broker.disconnect()

print('=' * 70)
print('✅ ALL SYSTEMS OPERATIONAL!')
print('=' * 70)
print()
print('Ready to start paper trading engine!')
print()
