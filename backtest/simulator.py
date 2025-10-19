"""
Backtest Simulator Module
Simulate trading strategies on historical data
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class TradingSimulator:
    """Simulate trading with realistic conditions"""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.commission_rate = 0.001  # 0.1%
        self.slippage_rate = 0.0005  # 0.05%
        self.min_commission = 1.0
        
    def reset(self):
        """Reset simulator state"""
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
    
    def execute_signal(self, date: datetime, signal: Dict, 
                       current_prices: Dict[str, float]):
        """Execute a trading signal"""
        
        symbol = signal['symbol']
        action = signal['action']
        
        if symbol not in current_prices:
            return False
        
        price = current_prices[symbol]
        
        if action == 'BUY':
            return self._execute_buy(date, symbol, price, signal)
        elif action == 'SELL':
            return self._execute_sell(date, symbol, price, signal)
        
        return False
    
    def _execute_buy(self, date: datetime, symbol: str, 
                    price: float, signal: Dict) -> bool:
        """Execute buy order"""
        
        # Calculate position size
        position_value = self._calculate_position_size(signal)
        
        if position_value > self.cash:
            position_value = self.cash * 0.95  # Use 95% of available cash
        
        if position_value < 100:  # Minimum position size
            return False
        
        # Apply slippage
        execution_price = price * (1 + self.slippage_rate)
        
        # Calculate shares
        shares = int(position_value / execution_price)
        if shares == 0:
            return False
        
        # Calculate commission
        commission = max(shares * execution_price * self.commission_rate, 
                        self.min_commission)
        
        total_cost = shares * execution_price + commission
        
        if total_cost > self.cash:
            return False
        
        # Update cash
        self.cash -= total_cost
        
        # Update or create position
        if symbol in self.positions:
            position = self.positions[symbol]
            new_shares = position['shares'] + shares
            new_avg_price = ((position['shares'] * position['avg_price']) + 
                           (shares * execution_price)) / new_shares
            
            position['shares'] = new_shares
            position['avg_price'] = new_avg_price
        else:
            self.positions[symbol] = {
                'shares': shares,
                'avg_price': execution_price,
                'entry_date': date
            }
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'BUY',
            'shares': shares,
            'price': execution_price,
            'commission': commission,
            'cash_after': self.cash,
            'reason': signal.get('reason', '')
        })
        
        return True
    
    def _execute_sell(self, date: datetime, symbol: str,
                     price: float, signal: Dict) -> bool:
        """Execute sell order"""
        
        if symbol not in self.positions:
            return False
        
        position = self.positions[symbol]
        shares = position['shares']
        
        if shares == 0:
            return False
        
        # Partial sell if specified
        if 'shares' in signal:
            shares = min(shares, signal['shares'])
        
        # Apply slippage
        execution_price = price * (1 - self.slippage_rate)
        
        # Calculate commission
        commission = max(shares * execution_price * self.commission_rate,
                        self.min_commission)
        
        proceeds = shares * execution_price - commission
        
        # Update cash
        self.cash += proceeds
        
        # Calculate P&L
        pnl = (execution_price - position['avg_price']) * shares - commission
        pnl_percent = (pnl / (position['avg_price'] * shares)) * 100
        
        # Update or remove position
        position['shares'] -= shares
        if position['shares'] == 0:
            del self.positions[symbol]
        
        # Record trade
        self.trades.append({
            'date': date,
            'symbol': symbol,
            'action': 'SELL',
            'shares': shares,
            'price': execution_price,
            'pnl': pnl,
            'pnl_percent': pnl_percent,
            'commission': commission,
            'cash_after': self.cash,
            'reason': signal.get('reason', '')
        })
        
        return True
    
    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate position size from signal"""
        
        portfolio_value = self.get_portfolio_value()
        
        # Use signal strength for sizing
        strength = signal.get('signal_strength', 0.5)
        base_size = portfolio_value * 0.1  # 10% base size
        
        # Adjust by signal strength
        position_size = base_size * (0.5 + strength * 0.5)
        
        # Apply maximum position size
        max_size = portfolio_value * 0.2  # 20% max
        position_size = min(position_size, max_size)
        
        return position_size
    
    def update_portfolio_value(self, date: datetime, prices: Dict[str, float]):
        """Update portfolio value with current prices"""
        
        portfolio_value = self.cash
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                position['current_price'] = prices[symbol]
                position['market_value'] = position['shares'] * prices[symbol]
                position['unrealized_pnl'] = (
                    (prices[symbol] - position['avg_price']) * position['shares']
                )
                portfolio_value += position['market_value']
        
        self.equity_curve.append({
            'date': date,
            'portfolio_value': portfolio_value,
            'cash': self.cash,
            'positions_value': portfolio_value - self.cash,
            'num_positions': len(self.positions)
        })
        
        return portfolio_value
    
    def get_portfolio_value(self, prices: Optional[Dict[str, float]] = None) -> float:
        """Get current portfolio value"""
        
        value = self.cash
        
        for symbol, position in self.positions.items():
            if prices and symbol in prices:
                value += position['shares'] * prices[symbol]
            elif 'current_price' in position:
                value += position['shares'] * position['current_price']
            else:
                value += position['shares'] * position['avg_price']
        
        return value
    
    def check_stop_losses(self, date: datetime, prices: Dict[str, float],
                         stop_loss_pct: float = 0.02) -> List[Dict]:
        """Check and execute stop losses"""
        
        executions = []
        
        for symbol, position in list(self.positions.items()):
            if symbol in prices:
                current_price = prices[symbol]
                avg_price = position['avg_price']
                
                # Check stop loss
                if current_price <= avg_price * (1 - stop_loss_pct):
                    signal = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'reason': f'Stop loss triggered at {stop_loss_pct*100}%'
                    }
                    
                    if self._execute_sell(date, symbol, current_price, signal):
                        executions.append(signal)
        
        return executions
    
    def check_take_profits(self, date: datetime, prices: Dict[str, float],
                          take_profit_pct: float = 0.05) -> List[Dict]:
        """Check and execute take profits"""
        
        executions = []
        
        for symbol, position in list(self.positions.items()):
            if symbol in prices:
                current_price = prices[symbol]
                avg_price = position['avg_price']
                
                # Check take profit
                if current_price >= avg_price * (1 + take_profit_pct):
                    signal = {
                        'symbol': symbol,
                        'action': 'SELL',
                        'reason': f'Take profit triggered at {take_profit_pct*100}%'
                    }
                    
                    if self._execute_sell(date, symbol, current_price, signal):
                        executions.append(signal)
        
        return executions
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        
        if not self.equity_curve:
            return {}
        
        equity_df = pd.DataFrame(self.equity_curve)
        trades_df = pd.DataFrame(self.trades)
        
        initial_value = self.initial_capital
        final_value = equity_df.iloc[-1]['portfolio_value']
        
        # Calculate returns
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate other metrics
        summary = {
            'initial_capital': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_trades': len(self.trades),
            'open_positions': len(self.positions),
            'cash_remaining': self.cash
        }
        
        # Trade statistics
        if not trades_df.empty:
            sells = trades_df[trades_df['action'] == 'SELL']
            
            if not sells.empty:
                summary['winning_trades'] = len(sells[sells['pnl'] > 0])
                summary['losing_trades'] = len(sells[sells['pnl'] <= 0])
                summary['win_rate'] = (summary['winning_trades'] / 
                                      len(sells)) * 100
                summary['avg_win'] = sells[sells['pnl'] > 0]['pnl'].mean()
                summary['avg_loss'] = sells[sells['pnl'] <= 0]['pnl'].mean()
                summary['total_commission'] = trades_df['commission'].sum()
        
        return summary