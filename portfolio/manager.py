"""
Portfolio Manager Module
Complete portfolio management system
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class PortfolioManager:
    """Manage investment portfolios"""
    
    def __init__(self, db):
        self.db = db
        
    def get_current_portfolio(self, portfolio_name: str = "default") -> List[Dict]:
        """Get current portfolio positions"""
        
        portfolio = self.db.get_portfolio(portfolio_name)
        if not portfolio:
            return []
        
        positions = self.db.get_positions(portfolio['id'], status='open')
        
        # Update current prices
        for position in positions:
            current_price = self.db.get_latest_price(position['symbol'])
            if current_price:
                position['current_price'] = current_price
                position['market_value'] = position['shares'] * current_price
                position['unrealized_pnl'] = (current_price - position['average_price']) * position['shares']
                position['unrealized_pnl_percent'] = ((current_price - position['average_price']) / 
                                                      position['average_price'] * 100)
        
        return positions
    
    def add_position(self, portfolio_name: str, symbol: str, 
                    shares: int, price: float) -> bool:
        """Add position to portfolio"""
        try:
            portfolio = self.db.get_portfolio(portfolio_name)
            if not portfolio:
                portfolio_id = self.db.create_portfolio(portfolio_name, 100000)
                portfolio = self.db.get_portfolio(portfolio_name)
            
            self.db.execute_transaction(
                portfolio_id=portfolio['id'],
                symbol=symbol,
                transaction_type='BUY',
                shares=shares,
                price=price,
                notes=f"Manual buy order"
            )
            
            logger.info(f"Added {shares} shares of {symbol} at ${price}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False
    
    def remove_position(self, portfolio_name: str, symbol: str, 
                       shares: Optional[int] = None) -> bool:
        """Remove position from portfolio"""
        try:
            portfolio = self.db.get_portfolio(portfolio_name)
            if not portfolio:
                return False
            
            positions = self.db.get_positions(portfolio['id'], status='open')
            position = next((p for p in positions if p['symbol'] == symbol), None)
            
            if not position:
                logger.warning(f"No position found for {symbol}")
                return False
            
            # If shares not specified, sell all
            if shares is None:
                shares = position['shares']
            
            current_price = self.db.get_latest_price(symbol)
            if not current_price:
                logger.error(f"Cannot get current price for {symbol}")
                return False
            
            self.db.execute_transaction(
                portfolio_id=portfolio['id'],
                symbol=symbol,
                transaction_type='SELL',
                shares=shares,
                price=current_price,
                notes=f"Manual sell order"
            )
            
            logger.info(f"Sold {shares} shares of {symbol} at ${current_price}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing position: {e}")
            return False
    
    def get_portfolio_value(self, portfolio_name: str = "default") -> float:
        """Calculate total portfolio value"""
        
        portfolio = self.db.get_portfolio(portfolio_name)
        if not portfolio:
            return 0
        
        total_value = portfolio['cash_balance']
        
        positions = self.get_current_portfolio(portfolio_name)
        for position in positions:
            if 'market_value' in position:
                total_value += position['market_value']
        
        return total_value
    
    def get_portfolio_performance(self, portfolio_name: str = "default") -> Dict:
        """Calculate portfolio performance metrics"""
        
        portfolio = self.db.get_portfolio(portfolio_name)
        if not portfolio:
            return {}
        
        current_value = self.get_portfolio_value(portfolio_name)
        initial_capital = portfolio['initial_capital']
        
        # Calculate returns
        total_return = (current_value - initial_capital) / initial_capital * 100 if initial_capital > 0 else 0
        
        # Get transaction history for more metrics
        with self.db.get_connection() as conn:
            df = pd.read_sql_query('''
                SELECT * FROM transactions 
                WHERE portfolio_id = ?
                ORDER BY transaction_date
            ''', conn, params=[portfolio['id']])
        
        # Calculate additional metrics
        metrics = {
            'total_value': current_value,
            'initial_capital': initial_capital,
            'total_return': total_return,
            'cash_balance': portfolio['cash_balance'],
            'positions_count': len(self.get_current_portfolio(portfolio_name)),
            'total_trades': len(df) if not df.empty else 0,
            'first_trade': df['transaction_date'].min() if not df.empty else None,
            'last_trade': df['transaction_date'].max() if not df.empty else None
        }
        
        return metrics
    
    def get_transaction_history(self, portfolio_name: str = "default") -> List[Dict]:
        """Get transaction history"""
        portfolio = self.db.get_portfolio(portfolio_name)
        if not portfolio:
            return []
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM transactions 
                WHERE portfolio_id = ?
                ORDER BY transaction_date DESC
                LIMIT 100
            ''', (portfolio['id'],))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def rebalance_portfolio(self, portfolio_name: str, target_weights: Dict[str, float]) -> List[Dict]:
        """Rebalance portfolio to target weights"""
        recommendations = []
        
        portfolio = self.db.get_portfolio(portfolio_name)
        if not portfolio:
            return recommendations
        
        current_value = self.get_portfolio_value(portfolio_name)
        positions = self.get_current_portfolio(portfolio_name)
        
        # Calculate current weights
        current_weights = {}
        for pos in positions:
            if 'market_value' in pos:
                current_weights[pos['symbol']] = pos['market_value'] / current_value
        
        # Calculate rebalancing trades
        for symbol, target_weight in target_weights.items():
            current_weight = current_weights.get(symbol, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # 1% threshold
                target_value = current_value * target_weight
                current_value_symbol = current_value * current_weight
                value_diff = target_value - current_value_symbol
                
                current_price = self.db.get_latest_price(symbol)
                if current_price:
                    shares = int(value_diff / current_price)
                    
                    if shares != 0:
                        recommendations.append({
                            'symbol': symbol,
                            'action': 'BUY' if shares > 0 else 'SELL',
                            'shares': abs(shares),
                            'price': current_price,
                            'reason': f"Rebalance to {target_weight*100:.1f}% weight"
                        })
        
        return recommendations