"""
Backtest Metrics Module
Calculate comprehensive performance metrics
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class PerformanceMetrics:
    """Calculate backtest performance metrics"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
    
    def calculate_all_metrics(self, equity_curve: pd.DataFrame,
                            trades: pd.DataFrame) -> Dict:
        """Calculate all performance metrics"""
        
        metrics = {}
        
        # Return metrics
        metrics.update(self.calculate_return_metrics(equity_curve))
        
        # Risk metrics  
        metrics.update(self.calculate_risk_metrics(equity_curve))
        
        # Trade metrics
        metrics.update(self.calculate_trade_metrics(trades))
        
        # Risk-adjusted metrics
        metrics.update(self.calculate_risk_adjusted_metrics(equity_curve))
        
        # Drawdown metrics
        metrics.update(self.calculate_drawdown_metrics(equity_curve))
        
        return metrics
    
    def calculate_return_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate return-based metrics"""
        
        if equity_curve.empty:
            return {}
        
        initial_value = equity_curve.iloc[0]['portfolio_value']
        final_value = equity_curve.iloc[-1]['portfolio_value']
        
        # Total return
        total_return = ((final_value - initial_value) / initial_value) * 100
        
        # Calculate daily returns
        equity_curve['returns'] = equity_curve['portfolio_value'].pct_change()
        
        # Annual return (CAGR)
        days = len(equity_curve)
        years = days / 252
        annual_return = (((final_value / initial_value) ** (1 / years)) - 1) * 100 if years > 0 else 0
        
        # Monthly returns
        monthly_returns = equity_curve.set_index('date')['returns'].resample('M').apply(
            lambda x: (1 + x).prod() - 1
        )
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'monthly_return_mean': monthly_returns.mean() * 100,
            'monthly_return_std': monthly_returns.std() * 100,
            'best_month': monthly_returns.max() * 100,
            'worst_month': monthly_returns.min() * 100,
            'positive_months': (monthly_returns > 0).sum(),
            'negative_months': (monthly_returns < 0).sum()
        }
    
    def calculate_risk_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate risk metrics"""
        
        if 'returns' not in equity_curve.columns:
            equity_curve['returns'] = equity_curve['portfolio_value'].pct_change()
        
        returns = equity_curve['returns'].dropna()
        
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Downside deviation
        downside_returns = returns[returns < 0]
        downside_dev = downside_returns.std() * np.sqrt(252)
        
        # VaR and CVaR
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean()
        
        # Skewness and Kurtosis
        skewness = stats.skew(returns)
        kurtosis = stats.kurtosis(returns)
        
        return {
            'annual_volatility': annual_vol * 100,
            'downside_deviation': downside_dev * 100,
            'var_95': var_95 * 100,
            'cvar_95': cvar_95 * 100,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'positive_days': (returns > 0).sum(),
            'negative_days': (returns < 0).sum(),
            'hit_rate': (returns > 0).sum() / len(returns) * 100
        }
    
    def calculate_trade_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate trade-based metrics"""
        
        if trades.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'profit_factor': 0
            }
        
        # Separate buys and sells
        sells = trades[trades['action'] == 'SELL'].copy()
        
        if sells.empty:
            return {
                'total_trades': len(trades),
                'win_rate': 0,
                'profit_factor': 0
            }
        
        # Win/Loss analysis
        winning_trades = sells[sells['pnl'] > 0]
        losing_trades = sells[sells['pnl'] <= 0]
        
        win_rate = len(winning_trades) / len(sells) * 100
        
        # Profit factor
        total_wins = winning_trades['pnl'].sum() if not winning_trades.empty else 0
        total_losses = abs(losing_trades['pnl'].sum()) if not losing_trades.empty else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else 0
        
        # Average trade metrics
        avg_win = winning_trades['pnl'].mean() if not winning_trades.empty else 0
        avg_loss = losing_trades['pnl'].mean() if not losing_trades.empty else 0
        
        # Largest wins/losses
        largest_win = winning_trades['pnl'].max() if not winning_trades.empty else 0
        largest_loss = losing_trades['pnl'].min() if not losing_trades.empty else 0
        
        # Trade duration
        if 'date' in trades.columns:
            trades['date'] = pd.to_datetime(trades['date'])
            # Calculate holding periods (simplified)
            
        return {
            'total_trades': len(trades),
            'total_sells': len(sells),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'largest_win': largest_win,
            'largest_loss': largest_loss,
            'avg_win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0,
            'expectancy': (win_rate/100 * avg_win) + ((1-win_rate/100) * avg_loss)
        }
    
    def calculate_risk_adjusted_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate risk-adjusted performance metrics"""
        
        if 'returns' not in equity_curve.columns:
            equity_curve['returns'] = equity_curve['portfolio_value'].pct_change()
        
        returns = equity_curve['returns'].dropna()
        
        # Sharpe Ratio
        excess_returns = returns - self.risk_free_rate / 252
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino Ratio
        downside_returns = returns[returns < 0]
        sortino_ratio = (np.sqrt(252) * excess_returns.mean() / 
                        downside_returns.std() if len(downside_returns) > 0 and downside_returns.std() > 0 else 0)
        
        # Calmar Ratio
        annual_return = returns.mean() * 252
        max_dd = self.calculate_max_drawdown(equity_curve['portfolio_value'])
        calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio
        }
    
    def calculate_drawdown_metrics(self, equity_curve: pd.DataFrame) -> Dict:
        """Calculate drawdown metrics"""
        
        portfolio_value = equity_curve['portfolio_value']
        
        # Calculate drawdown series
        cumulative = (1 + portfolio_value.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        # Max drawdown
        max_drawdown = drawdown.min()
        
        # Max drawdown duration
        drawdown_start = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration = i - drawdown_start
            else:
                if current_duration > max_duration:
                    max_duration = current_duration
                drawdown_start = None
                current_duration = 0
        
        # Recovery time
        max_dd_idx = drawdown.idxmin()
        if max_dd_idx < len(drawdown) - 1:
            recovery_idx = None
            for i in range(max_dd_idx + 1, len(drawdown)):
                if drawdown.iloc[i] >= 0:
                    recovery_idx = i
                    break
            recovery_time = recovery_idx - max_dd_idx if recovery_idx else None
        else:
            recovery_time = None
        
        # Average drawdown
        drawdowns_only = drawdown[drawdown < 0]
        avg_drawdown = drawdowns_only.mean() if not drawdowns_only.empty else 0
        
        return {
            'max_drawdown': abs(max_drawdown),
            'max_drawdown_duration': max_duration,
            'recovery_time': recovery_time,
            'avg_drawdown': abs(avg_drawdown),
            'drawdown_periods': (drawdown < 0).sum(),
            'current_drawdown': abs(drawdown.iloc[-1]) if drawdown.iloc[-1] < 0 else 0
        }
    
    def calculate_max_drawdown(self, values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative = (1 + values.pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def calculate_rolling_metrics(self, equity_curve: pd.DataFrame,
                                 window: int = 252) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        if 'returns' not in equity_curve.columns:
            equity_curve['returns'] = equity_curve['portfolio_value'].pct_change()
        
        rolling_metrics = pd.DataFrame(index=equity_curve.index)
        
        # Rolling return
        rolling_metrics['rolling_return'] = (
            equity_curve['returns'].rolling(window).apply(
                lambda x: (1 + x).prod() - 1
            ) * 100
        )
        
        # Rolling volatility
        rolling_metrics['rolling_volatility'] = (
            equity_curve['returns'].rolling(window).std() * np.sqrt(252) * 100
        )
        
        # Rolling Sharpe
        rolling_metrics['rolling_sharpe'] = (
            equity_curve['returns'].rolling(window).apply(
                lambda x: np.sqrt(252) * x.mean() / x.std() if x.std() > 0 else 0
            )
        )
        
        # Rolling max drawdown
        rolling_metrics['rolling_max_dd'] = (
            equity_curve['portfolio_value'].rolling(window).apply(
                lambda x: self.calculate_max_drawdown(x)
            )
        )
        
        return rolling_metrics