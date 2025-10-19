"""
Backtest Engine Module
High-performance backtesting with realistic simulation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass, field

from config import TradingRules
from database import DatabaseManager
from scanner.daily_scanner import DailyScanner

logger = logging.getLogger(__name__)

@dataclass
class BacktestResult:
    """Backtest results container"""
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    annual_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    profitable_trades: int
    avg_win: float
    avg_loss: float
    profit_factor: float
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)
    parameters: Dict = field(default_factory=dict)

class BacktestEngine:
    """Advanced backtesting engine"""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self.scanner = None
        self.commission = 0.001  # 0.1% commission
        self.slippage = 0.0005  # 0.05% slippage
        
    def run(self, start_date: str, end_date: str, 
            initial_capital: float = 100000,
            strategy: TradingRules = None) -> BacktestResult:
        """Run backtest simulation"""
        
        logger.info(f"Starting backtest from {start_date} to {end_date}")
        
        # Initialize
        strategy = strategy or TradingRules()
        self.scanner = DailyScanner(self.db, strategy)
        
        # Create result object
        result = BacktestResult(
            strategy_name="Default Strategy",
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            final_value=initial_capital,
            total_return=0,
            annual_return=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            max_drawdown=0,
            win_rate=0,
            total_trades=0,
            profitable_trades=0,
            avg_win=0,
            avg_loss=0,
            profit_factor=0,
            parameters=strategy.to_dict()
        )
        
        # Run simulation
        self._run_simulation(result, start_date, end_date, initial_capital, strategy)
        
        # Calculate metrics
        self._calculate_metrics(result)
        
        logger.info(f"Backtest completed. Return: {result.total_return:.2f}%")
        
        return result
    
    def _run_simulation(self, result: BacktestResult, start_date: str, 
                       end_date: str, initial_capital: float, 
                       strategy: TradingRules):
        """Run the actual simulation"""
        
        # Initialize portfolio
        cash = initial_capital
        positions = {}  # symbol -> {'shares': x, 'avg_price': y}
        
        # Get trading days
        trading_days = self._get_trading_days(start_date, end_date)
        
        for day in trading_days:
            # Update positions with current prices
            portfolio_value = cash
            for symbol, position in positions.items():
                current_price = self._get_price_on_date(symbol, day)
                if current_price:
                    portfolio_value += position['shares'] * current_price
            
            result.equity_curve.append(portfolio_value)
            
            # Get signals for the day
            signals = self._get_signals_for_date(day, strategy)
            
            # Process signals
            for signal in signals:
                if signal['type'] == 'BUY' and cash > 0:
                    # Calculate position size
                    position_size = self._calculate_position_size(
                        cash, portfolio_value, signal, strategy
                    )
                    
                    if position_size > 0:
                        # Execute buy
                        price = signal['price'] * (1 + self.slippage)
                        cost = position_size * price * (1 + self.commission)
                        
                        if cost <= cash:
                            shares = position_size
                            cash -= cost
                            
                            # Update position
                            if signal['symbol'] in positions:
                                old_shares = positions[signal['symbol']]['shares']
                                old_avg = positions[signal['symbol']]['avg_price']
                                new_shares = old_shares + shares
                                new_avg = (old_shares * old_avg + shares * price) / new_shares
                                positions[signal['symbol']] = {
                                    'shares': new_shares,
                                    'avg_price': new_avg,
                                    'entry_date': day
                                }
                            else:
                                positions[signal['symbol']] = {
                                    'shares': shares,
                                    'avg_price': price,
                                    'entry_date': day
                                }
                            
                            # Record trade
                            result.trades.append({
                                'date': day,
                                'symbol': signal['symbol'],
                                'type': 'BUY',
                                'shares': shares,
                                'price': price,
                                'commission': position_size * price * self.commission
                            })
                            result.total_trades += 1
                
                elif signal['type'] == 'SELL' and signal['symbol'] in positions:
                    # Execute sell
                    position = positions[signal['symbol']]
                    price = signal['price'] * (1 - self.slippage)
                    proceeds = position['shares'] * price * (1 - self.commission)
                    cash += proceeds
                    
                    # Calculate P&L
                    pnl = (price - position['avg_price']) * position['shares']
                    if pnl > 0:
                        result.profitable_trades += 1
                    
                    # Record trade
                    result.trades.append({
                        'date': day,
                        'symbol': signal['symbol'],
                        'type': 'SELL',
                        'shares': position['shares'],
                        'price': price,
                        'pnl': pnl,
                        'commission': position['shares'] * price * self.commission
                    })
                    result.total_trades += 1
                    
                    # Remove position
                    del positions[signal['symbol']]
            
            # Check stop losses and take profits
            positions = self._check_exits(positions, day, strategy, cash, result)
        
        # Close all positions at end
        for symbol, position in positions.items():
            price = self._get_price_on_date(symbol, end_date)
            if price:
                proceeds = position['shares'] * price * (1 - self.commission)
                cash += proceeds
        
        result.final_value = cash
    
    def _calculate_position_size(self, cash: float, portfolio_value: float,
                                signal: Dict, strategy: TradingRules) -> int:
        """Calculate position size based on strategy"""
        
        if strategy.position_sizing_method == 'equal':
            # Equal weight
            position_value = portfolio_value / strategy.max_positions
        elif strategy.position_sizing_method == 'kelly':
            # Kelly criterion
            kelly_fraction = self._calculate_kelly_fraction(signal)
            position_value = portfolio_value * kelly_fraction
        else:
            # Risk parity
            position_value = portfolio_value * strategy.max_position_size
        
        # Apply constraints
        position_value = min(position_value, cash)
        position_value = min(position_value, portfolio_value * strategy.max_position_size)
        
        # Calculate shares
        shares = int(position_value / signal['price'])
        
        return shares
    
    def _calculate_kelly_fraction(self, signal: Dict) -> float:
        """Calculate Kelly fraction for position sizing"""
        # Simplified Kelly: f = (p*b - q) / b
        # where p = win probability, b = win/loss ratio, q = 1-p
        
        win_prob = 0.5 + signal['strength'] * 0.2  # Convert strength to probability
        win_loss_ratio = 2.0  # Assume 2:1 win/loss
        
        kelly = (win_prob * win_loss_ratio - (1 - win_prob)) / win_loss_ratio
        
        # Apply Kelly fraction limit (typically 25% of full Kelly)
        kelly = max(0, min(0.25, kelly * 0.25))
        
        return kelly
    
    def _check_exits(self, positions: Dict, date: str, strategy: TradingRules,
                    cash: float, result: BacktestResult) -> Dict:
        """Check for stop loss and take profit exits"""
        
        positions_to_close = []
        
        for symbol, position in positions.items():
            current_price = self._get_price_on_date(symbol, date)
            if not current_price:
                continue
            
            # Calculate return
            returns = (current_price - position['avg_price']) / position['avg_price']
            
            # Check stop loss
            if returns <= -strategy.stop_loss_percent:
                positions_to_close.append((symbol, 'Stop Loss'))
            
            # Check take profit
            elif returns >= strategy.take_profit_percent:
                positions_to_close.append((symbol, 'Take Profit'))
            
            # Check max holding period
            elif self._days_between(position['entry_date'], date) >= strategy.max_holding_period:
                positions_to_close.append((symbol, 'Max Holding Period'))
        
        # Close positions
        for symbol, reason in positions_to_close:
            position = positions[symbol]
            price = self._get_price_on_date(symbol, date)
            
            if price:
                proceeds = position['shares'] * price * (1 - self.commission)
                cash += proceeds
                
                # Calculate P&L
                pnl = (price - position['avg_price']) * position['shares']
                if pnl > 0:
                    result.profitable_trades += 1
                
                # Record trade
                result.trades.append({
                    'date': date,
                    'symbol': symbol,
                    'type': 'SELL',
                    'shares': position['shares'],
                    'price': price,
                    'pnl': pnl,
                    'reason': reason,
                    'commission': position['shares'] * price * self.commission
                })
                result.total_trades += 1
                
                del positions[symbol]
        
        return positions
    
    def _calculate_metrics(self, result: BacktestResult):
        """Calculate performance metrics"""
        
        # Basic returns
        result.total_return = ((result.final_value - result.initial_capital) / 
                              result.initial_capital) * 100
        
        # Annual return
        days = self._days_between(result.start_date, result.end_date)
        years = days / 365.25
        if years > 0:
            result.annual_return = (((result.final_value / result.initial_capital) ** 
                                    (1 / years)) - 1) * 100
        
        # Calculate daily returns
        if len(result.equity_curve) > 1:
            equity_array = np.array(result.equity_curve)
            daily_returns = np.diff(equity_array) / equity_array[:-1]
            result.daily_returns = daily_returns.tolist()
            
            # Sharpe ratio (assuming 0% risk-free rate)
            if len(daily_returns) > 0:
                mean_return = np.mean(daily_returns)
                std_return = np.std(daily_returns)
                if std_return > 0:
                    result.sharpe_ratio = np.sqrt(252) * mean_return / std_return
            
            # Sortino ratio
            negative_returns = daily_returns[daily_returns < 0]
            if len(negative_returns) > 0:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    result.sortino_ratio = np.sqrt(252) * mean_return / downside_std
            
            # Max drawdown
            peak = equity_array[0]
            max_dd = 0
            for value in equity_array:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown = max_dd * 100
        
        # Win rate
        if result.total_trades > 0:
            result.win_rate = (result.profitable_trades / result.total_trades) * 100
        
        # Average win/loss
        wins = [t['pnl'] for t in result.trades if 'pnl' in t and t['pnl'] > 0]
        losses = [t['pnl'] for t in result.trades if 'pnl' in t and t['pnl'] < 0]
        
        if wins:
            result.avg_win = np.mean(wins)
        if losses:
            result.avg_loss = abs(np.mean(losses))
        
        # Profit factor
        if losses and result.avg_loss > 0:
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses))
            if total_losses > 0:
                result.profit_factor = total_wins / total_losses
    
    def _get_trading_days(self, start_date: str, end_date: str) -> List[str]:
        """Get list of trading days between dates"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT DISTINCT date FROM price_data 
                WHERE date BETWEEN ? AND ?
                ORDER BY date
            ''', (start_date, end_date))
            
            return [row[0] for row in cursor.fetchall()]
    
    def _get_price_on_date(self, symbol: str, date: str) -> Optional[float]:
        """Get closing price for symbol on specific date"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT close FROM price_data 
                WHERE symbol = ? AND date = ?
            ''', (symbol, date))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def _get_signals_for_date(self, date: str, strategy: TradingRules) -> List[Dict]:
        """Get trading signals for specific date"""
        # Simplified for backtest - would implement full signal generation
        signals = []
        
        # This would normally call the scanner with historical data
        # For now, return empty list
        return signals
    
    def _days_between(self, start: str, end: str) -> int:
        """Calculate days between two dates"""
        start_dt = datetime.strptime(start, '%Y-%m-%d')
        end_dt = datetime.strptime(end, '%Y-%m-%d')
        return (end_dt - start_dt).days

    def save_results(self, result: BacktestResult):
        """Save backtest results to database"""
        self.db.save_backtest_results({
            'strategy_name': result.strategy_name,
            'start_date': result.start_date,
            'end_date': result.end_date,
            'initial_capital': result.initial_capital,
            'final_value': result.final_value,
            'total_return': result.total_return,
            'annual_return': result.annual_return,
            'sharpe_ratio': result.sharpe_ratio,
            'max_drawdown': result.max_drawdown,
            'win_rate': result.win_rate,
            'total_trades': result.total_trades,
            'parameters': result.parameters,
            'equity_curve': result.equity_curve
        })