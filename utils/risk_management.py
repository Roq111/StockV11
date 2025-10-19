"""
Risk Management Module
Portfolio risk analysis and management
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class RiskManager:
    """Manage portfolio risk"""
    
    def __init__(self, db):
        self.db = db
        self.market_index = "SPY"
    
    def analyze_portfolio(self, portfolio_name: str = "default") -> Dict:
        """Analyze portfolio risk metrics"""
        
        from portfolio.manager import PortfolioManager
        pm = PortfolioManager(self.db)
        
        positions = pm.get_current_portfolio(portfolio_name)
        if not positions:
            return {}
        
        # Get portfolio returns
        portfolio_returns = self._get_portfolio_returns(positions)
        if portfolio_returns.empty:
            return {}
        
        # Get market returns
        market_returns = self._get_market_returns()
        
        # Calculate risk metrics
        metrics = {
            'volatility': portfolio_returns.std() * np.sqrt(252) * 100,
            'beta': self._calculate_beta(portfolio_returns, market_returns),
            'var_95': self._calculate_var(portfolio_returns, 0.95),
            'cvar_95': self._calculate_cvar(portfolio_returns, 0.95),
            'correlation': portfolio_returns.corr(market_returns) if not market_returns.empty else 0,
            'max_position_weight': self._get_max_position_weight(positions),
            'concentration_risk': self._calculate_concentration_risk(positions)
        }
        
        return metrics
    
    def check_risk_limits(self, portfolio_name: str, trade: Dict) -> Dict[str, bool]:
        """Check if trade violates risk limits"""
        
        from config import Config
        config = Config()
        rules = config.trading_rules
        
        checks = {
            'position_size': True,
            'max_positions': True,
            'concentration': True,
            'daily_loss': True
        }
        
        # Check position size limit
        from portfolio.manager import PortfolioManager
        pm = PortfolioManager(self.db)
        
        portfolio_value = pm.get_portfolio_value(portfolio_name)
        trade_value = trade['shares'] * trade['price']
        
        if trade_value > portfolio_value * rules.max_position_size:
            checks['position_size'] = False
        
        # Check max positions
        positions = pm.get_current_portfolio(portfolio_name)
        if len(positions) >= rules.max_positions and trade['action'] == 'BUY':
            checks['max_positions'] = False
        
        return checks
    
    def calculate_position_size(self, portfolio_value: float, 
                              symbol: str, entry_price: float,
                              stop_loss: float) -> int:
        """Calculate position size based on risk"""
        
        from config import Config
        config = Config()
        rules = config.trading_rules
        
        # Risk per trade (2% default)
        risk_amount = portfolio_value * 0.02
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        # Position size
        shares = int(risk_amount / risk_per_share)
        
        # Apply maximum position size constraint
        max_shares = int((portfolio_value * rules.max_position_size) / entry_price)
        
        return min(shares, max_shares)
    
    def _get_portfolio_returns(self, positions: List[Dict]) -> pd.Series:
        """Calculate portfolio returns"""
        
        if not positions:
            return pd.Series()
        
        # Get returns for each position
        all_returns = []
        weights = []
        
        total_value = sum(p.get('market_value', 0) for p in positions)
        
        for position in positions:
            symbol = position['symbol']
            weight = position.get('market_value', 0) / total_value if total_value > 0 else 0
            
            # Get price history
            df = self.db.get_price_history(symbol)
            if not df.empty:
                returns = df['close'].pct_change().dropna()
                all_returns.append(returns)
                weights.append(weight)
        
        if not all_returns:
            return pd.Series()
        
        # Align dates
        aligned_returns = pd.concat(all_returns, axis=1).fillna(0)
        
        # Calculate weighted portfolio returns
        portfolio_returns = (aligned_returns * weights).sum(axis=1)
        
        return portfolio_returns
    
    def _get_market_returns(self) -> pd.Series:
        """Get market index returns"""
        
        df = self.db.get_price_history(self.market_index)
        if df.empty:
            return pd.Series()
        
        return df['close'].pct_change().dropna()
    
    def _calculate_beta(self, portfolio_returns: pd.Series, 
                       market_returns: pd.Series) -> float:
        """Calculate portfolio beta"""
        
        if portfolio_returns.empty or market_returns.empty:
            return 1.0
        
        # Align dates
        aligned = pd.concat([portfolio_returns, market_returns], axis=1).dropna()
        
        if len(aligned) < 20:
            return 1.0
        
        covariance = aligned.iloc[:, 0].cov(aligned.iloc[:, 1])
        market_variance = aligned.iloc[:, 1].var()
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    def _calculate_var(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Value at Risk"""
        
        if returns.empty:
            return 0
        
        return np.percentile(returns, (1 - confidence) * 100)
    
    def _calculate_cvar(self, returns: pd.Series, confidence: float) -> float:
        """Calculate Conditional Value at Risk"""
        
        if returns.empty:
            return 0
        
        var = self._calculate_var(returns, confidence)
        return returns[returns <= var].mean()
    
    def _get_max_position_weight(self, positions: List[Dict]) -> float:
        """Get maximum position weight"""
        
        if not positions:
            return 0
        
        total_value = sum(p.get('market_value', 0) for p in positions)
        
        if total_value == 0:
            return 0
        
        max_value = max(p.get('market_value', 0) for p in positions)
        
        return max_value / total_value
    
    def _calculate_concentration_risk(self, positions: List[Dict]) -> float:
        """Calculate concentration risk (Herfindahl index)"""
        
        if not positions:
            return 0
        
        total_value = sum(p.get('market_value', 0) for p in positions)
        
        if total_value == 0:
            return 0
        
        herfindahl = sum((p.get('market_value', 0) / total_value) ** 2 
                        for p in positions)
        
        return herfindahl