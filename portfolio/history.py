"""
Portfolio History Module
Track and analyze portfolio history
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PortfolioHistory:
    """Manage portfolio history and analytics"""
    
    def __init__(self, db):
        self.db = db
    
    def record_daily_snapshot(self, portfolio_id: int):
        """Record daily portfolio snapshot"""
        try:
            # Get current portfolio value
            from portfolio.manager import PortfolioManager
            pm = PortfolioManager(self.db)
            
            portfolio = self.db.get_portfolio_by_id(portfolio_id)
            if not portfolio:
                return
            
            positions = pm.get_current_portfolio(portfolio['name'])
            
            # Calculate values
            cash_balance = portfolio['cash_balance']
            positions_value = sum(p.get('market_value', 0) for p in positions)
            total_value = cash_balance + positions_value
            
            # Get previous snapshot
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute('''
                    SELECT total_value, cumulative_pnl 
                    FROM performance_history 
                    WHERE portfolio_id = ? 
                    ORDER BY date DESC 
                    LIMIT 1
                ''', (portfolio_id,))
                
                prev = cursor.fetchone()
                
                if prev:
                    daily_return = (total_value - prev[0]) / prev[0]
                    daily_pnl = total_value - prev[0]
                    cumulative_pnl = prev[1] + daily_pnl
                else:
                    daily_return = 0
                    daily_pnl = 0
                    cumulative_pnl = total_value - portfolio['initial_capital']
                
                # Calculate cumulative return
                cumulative_return = ((total_value - portfolio['initial_capital']) / 
                                   portfolio['initial_capital'])
                
                # Insert snapshot
                cursor.execute('''
                    INSERT OR REPLACE INTO performance_history 
                    (portfolio_id, date, total_value, cash_balance, positions_value,
                     daily_return, cumulative_return, daily_pnl, cumulative_pnl)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (portfolio_id, datetime.now().date(), total_value, cash_balance,
                     positions_value, daily_return, cumulative_return, daily_pnl,
                     cumulative_pnl))
                
                conn.commit()
                logger.info(f"Recorded snapshot for portfolio {portfolio_id}")
                
        except Exception as e:
            logger.error(f"Error recording snapshot: {e}")
    
    def get_history(self, portfolio_id: int, days: int = 30) -> pd.DataFrame:
        """Get portfolio history"""
        with self.db.get_connection() as conn:
            query = '''
                SELECT * FROM performance_history 
                WHERE portfolio_id = ? 
                AND date >= date('now', '-{} days')
                ORDER BY date
            '''.format(days)
            
            df = pd.read_sql_query(query, conn, params=[portfolio_id])
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df
    
    def calculate_rolling_metrics(self, portfolio_id: int, window: int = 30) -> Dict:
        """Calculate rolling performance metrics"""
        history = self.get_history(portfolio_id, days=window * 3)
        
        if history.empty:
            return {}
        
        metrics = {}
        
        # Rolling returns
        metrics['rolling_return'] = history['daily_return'].rolling(window).mean() * 252
        
        # Rolling volatility
        metrics['rolling_volatility'] = history['daily_return'].rolling(window).std() * np.sqrt(252)
        
        # Rolling Sharpe
        risk_free = 0.02 / 252
        excess_returns = history['daily_return'] - risk_free
        metrics['rolling_sharpe'] = (excess_returns.rolling(window).mean() / 
                                    history['daily_return'].rolling(window).std() * np.sqrt(252))
        
        # Rolling max drawdown
        rolling_max = history['total_value'].rolling(window, min_periods=1).max()
        drawdown = (history['total_value'] - rolling_max) / rolling_max
        metrics['rolling_max_drawdown'] = drawdown.rolling(window).min()
        
        return metrics
    
    def get_transaction_history(self, portfolio_id: int, 
                              symbol: Optional[str] = None,
                              days: Optional[int] = None) -> List[Dict]:
        """Get transaction history"""
        with self.db.get_connection() as conn:
            query = 'SELECT * FROM transactions WHERE portfolio_id = ?'
            params = [portfolio_id]
            
            if symbol:
                query += ' AND symbol = ?'
                params.append(symbol)
            
            if days:
                query += ' AND transaction_date >= date("now", "-{} days")'.format(days)
            
            query += ' ORDER BY transaction_date DESC'
            
            cursor = conn.cursor()
            cursor.execute(query, params)
            
            return [dict(row) for row in cursor.fetchall()]
    
    def analyze_trade_history(self, portfolio_id: int) -> Dict:
        """Analyze trading history"""
        transactions = self.get_transaction_history(portfolio_id)
        
        if not transactions:
            return {}
        
        # Group by symbol
        trades_by_symbol = {}
        for tx in transactions:
            symbol = tx['symbol']
            if symbol not in trades_by_symbol:
                trades_by_symbol[symbol] = []
            trades_by_symbol[symbol].append(tx)
        
        # Analyze each symbol
        analysis = {
            'total_trades': len(transactions),
            'unique_symbols': len(trades_by_symbol),
            'avg_trades_per_symbol': len(transactions) / len(trades_by_symbol),
            'symbols': {}
        }
        
        for symbol, trades in trades_by_symbol.items():
            buys = [t for t in trades if t['transaction_type'] == 'BUY']
            sells = [t for t in trades if t['transaction_type'] == 'SELL']
            
            analysis['symbols'][symbol] = {
                'total_trades': len(trades),
                'buys': len(buys),
                'sells': len(sells),
                'total_bought': sum(t['shares'] for t in buys),
                'total_sold': sum(t['shares'] for t in sells),
                'avg_buy_price': np.mean([t['price'] for t in buys]) if buys else 0,
                'avg_sell_price': np.mean([t['price'] for t in sells]) if sells else 0
            }
        
        return analysis
    
    def get_monthly_performance(self, portfolio_id: int) -> pd.DataFrame:
        """Get monthly performance summary"""
        history = self.get_history(portfolio_id, days=365)
        
        if history.empty:
            return pd.DataFrame()
        
        # Resample to monthly
        monthly = history.resample('M').agg({
            'total_value': 'last',
            'daily_return': lambda x: (1 + x).prod() - 1,
            'daily_pnl': 'sum'
        })
        
        monthly.columns = ['ending_value', 'monthly_return', 'monthly_pnl']
        
        return monthly
    
    def export_history(self, portfolio_id: int, filename: str):
        """Export portfolio history to CSV"""
        try:
            history = self.get_history(portfolio_id, days=9999)
            transactions = pd.DataFrame(self.get_transaction_history(portfolio_id))
            
            with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
                history.to_excel(writer, sheet_name='Performance')
                transactions.to_excel(writer, sheet_name='Transactions', index=False)
            
            logger.info(f"Exported history to {filename}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting history: {e}")
            return False