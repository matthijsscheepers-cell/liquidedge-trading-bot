from typing import Dict, Tuple, Optional, List
from datetime import datetime, date
import pandas as pd
from dataclasses import dataclass, field

from src.risk.limits import RiskLimits, RiskProfile
from src.risk.position_sizing import PositionSizer


@dataclass
class TradeRecord:
    """Record of a completed trade"""
    timestamp: datetime
    asset: str
    direction: str
    entry_price: float
    exit_price: float
    units: float
    pnl: float
    pnl_pct: float
    outcome: str  # 'WIN' or 'LOSS'
    setup_type: str
    exit_reason: str
    capital_after: float
    dd_pct_after: float


class RiskGovernor:
    """
    Multi-layer risk management system.

    Layers:
    1. Trade-level: Can this single trade be taken?
    2. Daily-level: Have we hit daily limits?
    3. Drawdown-level: Risk scaling based on DD
    4. Streak-level: Adjust after wins/losses
    5. Portfolio-level: Concentration limits

    This is the MASTER control - overrides everything else.

    Example:
        governor = RiskGovernor(initial_capital=10000)
        allowed, reason, risk = governor.can_open_trade(
            asset='US_TECH_100',
            regime='STRONG_TREND',
            setup={'confidence': 85}
        )
    """

    def __init__(self, initial_capital: float):
        """
        Initialize risk governor

        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital

        # Get appropriate risk profile
        self.profile = RiskLimits.get_profile(initial_capital)
        self.position_sizer = PositionSizer(self.profile)

        # Daily tracking
        self.current_date: Optional[date] = None
        self.daily_pnl = 0.0
        self.daily_trades: List[TradeRecord] = []

        # Streak tracking
        self.consecutive_wins = 0
        self.consecutive_losses = 0

        # Drawdown tracking
        self.current_dd_pct = 0.0

        # Portfolio tracking
        self.open_positions: Dict[str, dict] = {}

        # Trade history
        self.trade_log: List[TradeRecord] = []

        # Risk event log
        self.risk_events: List[dict] = []

    def check_daily_reset(self, current_time: datetime):
        """
        Reset daily counters if new day

        Args:
            current_time: Current timestamp
        """
        current_date = current_time.date()

        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_pnl = 0.0
            self.daily_trades = []

            self._log_event('DAILY_RESET', {
                'date': current_date,
                'capital': self.current_capital,
                'dd_pct': self.current_dd_pct,
                'streak_w': self.consecutive_wins,
                'streak_l': self.consecutive_losses
            })

    def can_open_trade(self,
                      asset: str,
                      regime: str,
                      setup: dict) -> Tuple[bool, str, float]:
        """
        MASTER DECISION: Can we open this trade?

        Checks ALL risk layers in order:
        1. Hard stops (DD, daily loss, streak)
        2. Regime restrictions
        3. Portfolio limits
        4. Calculate adjusted risk

        Args:
            asset: Trading instrument
            regime: Market regime
            setup: Setup dict with 'confidence', 'risk_per_share', etc

        Returns:
            Tuple of (allowed, reason, adjusted_risk_pct)
        """

        # ========================================
        # LAYER 1: HARD STOPS
        # ========================================

        # Check max drawdown
        if self.current_dd_pct >= self.profile.max_drawdown_pct:
            return False, "MAX_DRAWDOWN_REACHED", 0.0

        # Check daily loss limit
        daily_loss_pct = self.daily_pnl / self.current_capital
        if daily_loss_pct <= -self.profile.max_daily_loss_pct:
            return False, "DAILY_LOSS_LIMIT", 0.0

        # Check loss streak
        MAX_LOSS_STREAK = 4  # Hardcoded limit
        if self.consecutive_losses >= MAX_LOSS_STREAK:
            return False, "LOSS_STREAK_LIMIT", 0.0

        # Check daily trade limit
        MAX_DAILY_TRADES = 5  # Hardcoded limit
        if len(self.daily_trades) >= MAX_DAILY_TRADES:
            return False, "DAILY_TRADE_LIMIT", 0.0

        # ========================================
        # LAYER 2: PORTFOLIO LIMITS
        # ========================================

        # Check max concurrent positions
        if len(self.open_positions) >= self.profile.max_concurrent_positions:
            return False, "MAX_POSITIONS_OPEN", 0.0

        # Check if asset already open
        if asset in self.open_positions:
            return False, "ASSET_ALREADY_OPEN", 0.0

        # ========================================
        # LAYER 3: REGIME RESTRICTIONS
        # ========================================

        # In deep drawdown, only STRONG_TREND allowed
        if self.current_dd_pct >= self.profile.dd_scale_threshold_2:
            if regime != 'STRONG_TREND':
                return False, "DD_REGIME_RESTRICTION", 0.0

        # ========================================
        # LAYER 4: CALCULATE ADJUSTED RISK
        # ========================================

        base_risk = self.profile.base_risk_pct
        risk_multiplier = 1.0

        # Drawdown scaling
        if self.current_dd_pct >= self.profile.dd_scale_threshold_2:
            risk_multiplier = 0.50  # Halve risk
        elif self.current_dd_pct >= self.profile.dd_scale_threshold_1:
            risk_multiplier = 0.65  # Reduce risk

        # Streak adjustments
        if self.consecutive_losses >= 2:
            risk_multiplier *= 0.80
        elif self.consecutive_wins >= 3:
            risk_multiplier *= 1.15  # Slight boost

        # Regime adjustments
        if regime == 'WEAK_TREND':
            risk_multiplier *= 0.80
        elif regime == 'STRONG_TREND' and setup.get('confidence', 70) > 80:
            risk_multiplier *= 1.10

        adjusted_risk = base_risk * risk_multiplier

        # Hard cap at 3%
        adjusted_risk = min(adjusted_risk, 0.03)

        # ========================================
        # LAYER 5: MINIMUM POSITION CHECK
        # ========================================

        # Estimate position value
        risk_capital = self.current_capital * adjusted_risk
        risk_per_share = setup.get('risk_per_share', 1.0)
        entry_price = setup.get('entry_price', 100.0)

        estimated_units = risk_capital / risk_per_share
        estimated_value = estimated_units * entry_price

        if estimated_value < self.profile.min_position_value:
            # Try increasing risk to meet minimum
            min_risk_needed = (self.profile.min_position_value * risk_per_share) / (entry_price * self.current_capital)

            if min_risk_needed > 0.04:  # Still too risky
                return False, "POSITION_TOO_SMALL", 0.0

            adjusted_risk = min_risk_needed

            self._log_event('RISK_OVERRIDE', {
                'reason': 'Minimum position size',
                'original_risk': base_risk * risk_multiplier,
                'adjusted_risk': adjusted_risk
            })

        # Log calculation
        self._log_event('RISK_CALCULATION', {
            'asset': asset,
            'profile': self.profile.profile_name,
            'base_risk': base_risk,
            'multiplier': risk_multiplier,
            'adjusted_risk': adjusted_risk,
            'dd_pct': self.current_dd_pct,
            'capital': self.current_capital,
            'streak': f"W{self.consecutive_wins}/L{self.consecutive_losses}"
        })

        return True, "APPROVED", adjusted_risk

    def calculate_position_size(self,
                               risk_pct: float,
                               risk_per_share: float,
                               price: float,
                               asset: str) -> float:
        """
        Calculate position size using position sizer

        Args:
            risk_pct: Adjusted risk percentage
            risk_per_share: Risk per unit
            price: Current price
            asset: Instrument

        Returns:
            Position size in units
        """
        return self.position_sizer.calculate_size(
            capital=self.current_capital,
            risk_pct=risk_pct,
            risk_per_share=risk_per_share,
            price=price,
            asset=asset
        )

    def add_position(self, asset: str, position_data: dict):
        """
        Add position to tracking

        Args:
            asset: Instrument
            position_data: Position details
        """
        self.open_positions[asset] = position_data

    def record_trade(self,
                    asset: str,
                    direction: str,
                    entry_price: float,
                    exit_price: float,
                    units: float,
                    outcome: str,
                    setup_type: str,
                    exit_reason: str = 'UNKNOWN'):
        """
        Record completed trade and update metrics

        Args:
            asset: Instrument
            direction: LONG or SHORT
            entry_price: Entry price
            exit_price: Exit price
            units: Position size
            outcome: WIN or LOSS
            setup_type: Type of setup
            exit_reason: Why exited
        """

        # Calculate P&L
        if direction == 'LONG':
            pnl = (exit_price - entry_price) * units
        else:
            pnl = (entry_price - exit_price) * units

        pnl_pct = pnl / self.current_capital

        # Update capital
        self.current_capital += pnl
        self.daily_pnl += pnl

        # Update peak and drawdown
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
            self.current_dd_pct = 0.0
        else:
            self.current_dd_pct = (self.peak_capital - self.current_capital) / self.peak_capital

        # Update streaks
        if outcome == 'WIN':
            self.consecutive_wins += 1
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1
            self.consecutive_wins = 0

        # Create trade record
        record = TradeRecord(
            timestamp=datetime.now(),
            asset=asset,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            units=units,
            pnl=pnl,
            pnl_pct=pnl_pct,
            outcome=outcome,
            setup_type=setup_type,
            exit_reason=exit_reason,
            capital_after=self.current_capital,
            dd_pct_after=self.current_dd_pct
        )

        self.trade_log.append(record)
        self.daily_trades.append(record)

        # Remove from open positions
        if asset in self.open_positions:
            del self.open_positions[asset]

        # Log
        self._log_event('TRADE_CLOSED', {
            'asset': asset,
            'outcome': outcome,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'capital': self.current_capital,
            'dd_pct': self.current_dd_pct
        })

    def should_pause_trading(self) -> Tuple[bool, str, str]:
        """
        Check if trading should be paused

        Returns:
            (should_pause, reason, resume_conditions)
        """

        if self.current_dd_pct >= self.profile.max_drawdown_pct:
            return True, "MAX_DRAWDOWN", "Manual review required"

        MAX_LOSS_STREAK = 4
        if self.consecutive_losses >= MAX_LOSS_STREAK:
            return True, "LOSS_STREAK", "Wait until next trading day"

        daily_loss_pct = self.daily_pnl / self.current_capital
        if daily_loss_pct <= -self.profile.max_daily_loss_pct:
            return True, "DAILY_LOSS_LIMIT", "Wait until next trading day"

        return False, "TRADING_ACTIVE", ""

    def get_health_status(self) -> dict:
        """
        Get current health status of risk system

        Returns:
            Dict with health metrics
        """

        health_score = 100
        warnings = []

        # Drawdown check
        if self.current_dd_pct > self.profile.dd_scale_threshold_2:
            health_score -= 40
            warnings.append(f"HIGH DRAWDOWN: {self.current_dd_pct*100:.1f}%")
        elif self.current_dd_pct > self.profile.dd_scale_threshold_1:
            health_score -= 20
            warnings.append(f"Elevated drawdown: {self.current_dd_pct*100:.1f}%")

        # Loss streak
        if self.consecutive_losses >= 2:
            health_score -= 15 * self.consecutive_losses
            warnings.append(f"Loss streak: {self.consecutive_losses}")

        # Daily loss
        daily_loss_pct = self.daily_pnl / self.current_capital
        if daily_loss_pct < -0.02:
            health_score -= 20
            warnings.append(f"Daily loss: {daily_loss_pct*100:.1f}%")

        # Capital check
        capital_pct = self.current_capital / self.initial_capital
        if capital_pct < 0.85:
            health_score -= 30
            warnings.append(f"Capital down {(1-capital_pct)*100:.1f}%")

        health_score = max(0, health_score)

        if health_score >= 80:
            status = "HEALTHY"
        elif health_score >= 60:
            status = "CAUTION"
        elif health_score >= 40:
            status = "WARNING"
        else:
            status = "CRITICAL"

        return {
            'score': health_score,
            'status': status,
            'warnings': warnings,
            'profile': self.profile.profile_name,
            'current_capital': self.current_capital,
            'dd_pct': self.current_dd_pct * 100,
            'positions_open': len(self.open_positions),
            'daily_pnl': self.daily_pnl,
            'streak_w': self.consecutive_wins,
            'streak_l': self.consecutive_losses
        }

    def get_statistics(self) -> Optional[dict]:
        """
        Get trading statistics

        Returns:
            Dict with performance metrics or None if no trades
        """

        if not self.trade_log:
            return None

        df = pd.DataFrame([vars(t) for t in self.trade_log])

        total_return = (self.current_capital - self.initial_capital) / self.initial_capital

        wins = df[df['outcome'] == 'WIN']
        losses = df[df['outcome'] == 'LOSS']

        win_rate = len(wins) / len(df) if len(df) > 0 else 0

        avg_win = wins['pnl_pct'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl_pct'].mean() if len(losses) > 0 else 0

        profit_factor = abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf')

        # Sharpe (simplified)
        returns = df['pnl_pct']
        sharpe = (returns.mean() / returns.std() * (252 ** 0.5)) if returns.std() > 0 else 0

        max_dd = df['dd_pct_after'].max()

        return {
            'total_trades': len(df),
            'total_return_pct': total_return * 100,
            'current_capital': self.current_capital,
            'win_rate': win_rate * 100,
            'avg_win_pct': avg_win * 100,
            'avg_loss_pct': avg_loss * 100,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_dd * 100,
            'current_dd_pct': self.current_dd_pct * 100,
            'consecutive_wins': self.consecutive_wins,
            'consecutive_losses': self.consecutive_losses
        }

    def _log_event(self, event_type: str, data: dict):
        """Log risk event"""
        self.risk_events.append({
            'timestamp': datetime.now(),
            'type': event_type,
            'data': data
        })
