import json
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd


class StateManager:
    """
    Manages bot state persistence.

    Saves and loads:
    - Open positions
    - Trading history
    - Risk metrics
    - Daily counters
    - Performance stats

    This ensures bot can recover after restart without losing data.

    Example:
        state = StateManager('data/live')
        state.save_state({
            'positions': {...},
            'balance': 10000,
            'daily_trades': [...]
        })

        # After restart
        loaded = state.load_state()
    """

    def __init__(self, data_dir: str = 'data/live'):
        """
        Initialize state manager

        Args:
            data_dir: Directory for state files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # File paths
        self.state_file = self.data_dir / 'bot_state.json'
        self.positions_file = self.data_dir / 'open_positions.json'
        self.history_file = self.data_dir / 'trade_history.json'
        self.metrics_file = self.data_dir / 'performance_metrics.json'
        self.backup_dir = self.data_dir / 'backups'
        self.backup_dir.mkdir(exist_ok=True)

    def save_state(self, state: Dict[str, Any]):
        """
        Save complete bot state

        Args:
            state: Dictionary with all bot state
        """
        try:
            # Add timestamp
            state['last_saved'] = datetime.now().isoformat()

            # Save to temp file first (atomic write)
            temp_file = self.state_file.with_suffix('.tmp')

            with open(temp_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)

            # Rename to actual file (atomic on most systems)
            temp_file.replace(self.state_file)

            print(f"[STATE] Saved bot state at {datetime.now()}")

        except Exception as e:
            print(f"[STATE] Error saving state: {e}")
            raise

    def load_state(self) -> Optional[Dict[str, Any]]:
        """
        Load bot state

        Returns:
            State dict or None if no state exists
        """
        try:
            if not self.state_file.exists():
                print("[STATE] No previous state found")
                return None

            with open(self.state_file, 'r') as f:
                state = json.load(f)

            print(f"[STATE] Loaded state from {state.get('last_saved', 'unknown time')}")
            return state

        except Exception as e:
            print(f"[STATE] Error loading state: {e}")
            return None

    def save_positions(self, positions: Dict[str, Any]):
        """
        Save open positions

        Args:
            positions: Dict of open positions
        """
        try:
            with open(self.positions_file, 'w') as f:
                json.dump(positions, f, indent=2, default=str)

            print(f"[STATE] Saved {len(positions)} open positions")

        except Exception as e:
            print(f"[STATE] Error saving positions: {e}")

    def load_positions(self) -> Dict[str, Any]:
        """
        Load open positions

        Returns:
            Dict of positions or empty dict
        """
        try:
            if not self.positions_file.exists():
                return {}

            with open(self.positions_file, 'r') as f:
                positions = json.load(f)

            return positions

        except Exception as e:
            print(f"[STATE] Error loading positions: {e}")
            return {}

    def save_trade_history(self, trades: list):
        """
        Append trades to history

        Args:
            trades: List of trade records
        """
        try:
            # Load existing
            history = self.load_trade_history()

            # Append new
            history.extend(trades)

            # Save
            with open(self.history_file, 'w') as f:
                json.dump(history, f, indent=2, default=str)

            print(f"[STATE] Saved {len(trades)} new trades (total: {len(history)})")

        except Exception as e:
            print(f"[STATE] Error saving trade history: {e}")

    def load_trade_history(self) -> list:
        """
        Load complete trade history

        Returns:
            List of trade records
        """
        try:
            if not self.history_file.exists():
                return []

            with open(self.history_file, 'r') as f:
                history = json.load(f)

            return history

        except Exception as e:
            print(f"[STATE] Error loading trade history: {e}")
            return []

    def save_metrics(self, metrics: Dict[str, Any]):
        """
        Save performance metrics

        Args:
            metrics: Performance metrics dict
        """
        try:
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2, default=str)

        except Exception as e:
            print(f"[STATE] Error saving metrics: {e}")

    def load_metrics(self) -> Optional[Dict[str, Any]]:
        """
        Load performance metrics

        Returns:
            Metrics dict or None
        """
        try:
            if not self.metrics_file.exists():
                return None

            with open(self.metrics_file, 'r') as f:
                metrics = json.load(f)

            return metrics

        except Exception as e:
            print(f"[STATE] Error loading metrics: {e}")
            return None

    def create_backup(self, label: str = ''):
        """
        Create backup of current state

        Args:
            label: Optional label for backup
        """
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"backup_{timestamp}"
            if label:
                backup_name += f"_{label}"

            backup_path = self.backup_dir / backup_name
            backup_path.mkdir(exist_ok=True)

            # Copy all state files
            import shutil

            if self.state_file.exists():
                shutil.copy2(self.state_file, backup_path / 'bot_state.json')

            if self.positions_file.exists():
                shutil.copy2(self.positions_file, backup_path / 'open_positions.json')

            if self.history_file.exists():
                shutil.copy2(self.history_file, backup_path / 'trade_history.json')

            if self.metrics_file.exists():
                shutil.copy2(self.metrics_file, backup_path / 'performance_metrics.json')

            print(f"[STATE] Created backup: {backup_name}")

        except Exception as e:
            print(f"[STATE] Error creating backup: {e}")

    def restore_from_backup(self, backup_name: str):
        """
        Restore state from backup

        Args:
            backup_name: Name of backup to restore
        """
        try:
            backup_path = self.backup_dir / backup_name

            if not backup_path.exists():
                print(f"[STATE] Backup not found: {backup_name}")
                return

            import shutil

            # Restore files
            for filename in ['bot_state.json', 'open_positions.json',
                           'trade_history.json', 'performance_metrics.json']:
                backup_file = backup_path / filename
                if backup_file.exists():
                    shutil.copy2(backup_file, self.data_dir / filename)

            print(f"[STATE] Restored from backup: {backup_name}")

        except Exception as e:
            print(f"[STATE] Error restoring backup: {e}")

    def list_backups(self) -> list:
        """
        List all available backups

        Returns:
            List of backup names
        """
        backups = [d.name for d in self.backup_dir.iterdir() if d.is_dir()]
        backups.sort(reverse=True)
        return backups

    def cleanup_old_backups(self, keep_last: int = 10):
        """
        Remove old backups, keep only recent ones

        Args:
            keep_last: Number of backups to keep
        """
        backups = self.list_backups()

        if len(backups) <= keep_last:
            return

        to_remove = backups[keep_last:]

        import shutil
        for backup in to_remove:
            backup_path = self.backup_dir / backup
            shutil.rmtree(backup_path)
            print(f"[STATE] Removed old backup: {backup}")
