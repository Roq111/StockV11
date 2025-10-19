"""
Performance Analysis Module
Calculate portfolio performance metrics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class PerformanceAnalyzer:
    """Analyze portfolio performance"""
    
    def __init__(self, db):
        self.db = db
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
    
    def calculate_metrics(self, portfolio_name: str = "default") -> Dict:
        """Calculate comprehensive performance metrics"""
        
        portfolio = self.db.get_portfolio(portfolio_name)
        if not portfolio:
            return {}
        
        # Get equity curve
        equity_curve = self._get_equity_curve(portfolio['id'])
        if equity_curve.empty:
            return {}
        
        # Calculate returns
        daily_returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = (equity_curve.iloc[-1] - equity_curve.iloc[0]) / equity_curve.iloc[0] * 100
        
        # Annual return
        days = len(equity_curve)
        years = days / 252
        annual_return = ((equity_curve.iloc[-1] / equity_curve.iloc[0]) ** (1/years) - 1) * 100 if years > 0 else 0
        
        # Risk metrics
        sharpe_ratio = self._calculate_sharpe(daily_returns)
        sortino_ratio = self._calculate_sortino(daily_returns)
        max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        # Win/loss metrics
        win_loss_metrics = self._calculate_win_loss_metrics(portfolio['id'])
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': daily_returns.std() * np.sqrt(252) * 100,
            **win_loss_metrics
        }
    
    def _get_equity_curve(self, portfolio_id: int) -> pd.Series:
        """Get daily equity curve"""
        
        with self.db.get_connection() as conn:
            # Get transactions
            transactions = pd.read_sql_query('''
                SELECT transaction_date, symbol, transaction_type, shares, price
                FROM transactions
                WHERE portfolio_id = ?
                ORDER BY transaction_date
            ''', conn, params=[portfolio_id])
            
            if transactions.empty:
                return pd.Series()
            
            # Get portfolio info
            cursor = conn.cursor()
            cursor.execute('SELECT initial_capital FROM portfolios WHERE id = ?', (portfolio_id,))
            initial_capital = cursor.fetchone()[0]
            
            # Build equity curve
            start_date = transactions['transaction_date'].min()
            end_date = datetime.now()
            
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            equity_curve = pd.Series(index=dates, dtype=float)
            
            cash = initial_capital
            positions = {}
            
            for date in dates:
                # Process transactions for this date
                day_transactions = transactions[
                    pd.to_datetime(transactions['transaction_date']).dt.date == date.date()
                ]
                
                for _, tx in day_transactions.iterrows():
                    if tx['transaction_type'] == 'BUY':
                        if tx['symbol'] not in positions:
                            positions[tx['symbol']] = {'shares': 0, 'avg_price': 0}
                        
                        total_shares = positions[tx['symbol']]['shares'] + tx['shares']
                        positions[tx['symbol']]['avg_price'] = (
                            (positions[tx['symbol']]['shares'] * positions[tx['symbol']]['avg_price'] +
                             tx['shares'] * tx['price']) / total_shares
                        )
                        positions[tx['symbol']]['shares'] = total_shares
                        cash -= tx['shares'] * tx['price']
                        
                    elif tx['transaction_type'] == 'SELL':
                        if tx['symbol'] in positions:
                            positions[tx['symbol']]['shares'] -= tx['shares']
                            if positions[tx['symbol']]['shares'] <= 0:
                                del positions[tx['symbol']]
                            cash += tx['shares'] * tx['price']
                
                # Calculate portfolio value
                portfolio_value = cash
                for symbol, pos in positions.items():
                    current_price = self._get_price_on_date(symbol, date)
                    if current_price:
                        portfolio_value += pos['shares'] * current_price
                
                equity_curve[date] = portfolio_value
            
            return equity_curve.dropna()
    
    def _get_price_on_date(self, symbol: str, date: pd.Timestamp) -> Optional[float]:
        """Get price for symbol on specific date"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT close FROM price_data 
                WHERE symbol = ? AND date <= ?
                ORDER BY date DESC
                LIMIT 1
            ''', (symbol, date.strftime('%Y-%m-%d')))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def _calculate_sharpe(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.empty:
            return 0
        
        excess_returns = returns - self.risk_free_rate / 252
        
        if returns.std() == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if returns.empty:
            return 0
        
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if downside_returns.empty or downside_returns.std() == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_max_drawdown(self, equity_curve: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if equity_curve.empty:
            return 0
        
        cumulative = (1 + equity_curve.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        
        return abs(drawdown.min()) * 100
    
    def _calculate_win_loss_metrics(self, portfolio_id: int) -> Dict:
        """Calculate win/loss metrics"""
        
        with self.db.get_connection() as conn:
            # Get closed positions
            cursor = conn.cursor()
            cursor.execute('''
                SELECT realized_pnl 
                FROM positions 
                WHERE portfolio_id = ? AND status = 'closed'
            ''', (portfolio_id,))
            
            pnls = [row[0] for row in cursor.fetchall()]
            
            if not pnls:
                return {
                    'win_rate': 0,
                    'avg_win': 0,
                    'avg_loss': 0,
                    'profit_factor': 0,
                    'total_trades': 0,
                    'profitable_trades': 0
                }
            
            wins = [pnl for pnl in pnls if pnl > 0]
            losses = [pnl for pnl in pnls if pnl < 0]
            
            win_rate = len(wins) / len(pnls) * 100 if pnls else 0
            avg_win = np.mean(wins) if wins else 0
            avg_loss = abs(np.mean(losses)) if losses else 0
            
            total_wins = sum(wins) if wins else 0
            total_losses = abs(sum(losses)) if losses else 0
            profit_factor = total_wins / total_losses if total_losses > 0 else 0
            
            return {
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_trades': len(pnls),
                'profitable_trades': len(wins)
            }