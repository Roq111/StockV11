"""
Recommendations Module
Generate actionable trading recommendations
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional
import logging

from models import Signal, Position, Portfolio

logger = logging.getLogger(__name__)

class RecommendationEngine:
    """Generate trading recommendations from signals"""
    
    def __init__(self, db, config=None):
        self.db = db
        self.config = config
    
    def generate_recommendations(self, signals: List[Signal], 
                                portfolio: Portfolio,
                                risk_params: Optional[Dict] = None) -> List[Dict]:
        """Generate trading recommendations from signals"""
        
        recommendations = []
        
        if not signals:
            return recommendations
        
        # Default risk parameters
        if not risk_params:
            risk_params = {
                'max_position_size': 0.1,
                'max_positions': 10,
                'risk_per_trade': 0.02,
                'min_signal_strength': 0.6
            }
        
        # Filter signals by strength
        strong_signals = [s for s in signals 
                         if s.signal_strength >= risk_params['min_signal_strength']]
        
        # Current positions
        current_positions = {p.symbol: p for p in portfolio.positions 
                           if p.status == 'open'}
        
        # Available capital
        available_capital = portfolio.cash_balance
        max_position_value = portfolio.total_value * risk_params['max_position_size']
        
        for signal in strong_signals:
            # Check if we already have a position
            if signal.symbol in current_positions:
                # Check for exit signals
                if signal.signal_type.value == "SELL":
                    rec = self._generate_exit_recommendation(
                        signal, current_positions[signal.symbol], portfolio
                    )
                    if rec:
                        recommendations.append(rec)
            else:
                # Check for entry signals
                if signal.signal_type.value == "BUY":
                    # Check position limits
                    if len(current_positions) >= risk_params['max_positions']:
                        continue
                    
                    rec = self._generate_entry_recommendation(
                        signal, portfolio, risk_params, 
                        available_capital, max_position_value
                    )
                    if rec:
                        recommendations.append(rec)
                        # Update available capital
                        available_capital -= rec['position_value']
        
        # Sort by priority
        recommendations = self._prioritize_recommendations(recommendations)
        
        return recommendations
    
    def _generate_entry_recommendation(self, signal: Signal, portfolio: Portfolio,
                                      risk_params: Dict, available_capital: float,
                                      max_position_value: float) -> Optional[Dict]:
        """Generate entry recommendation"""
        
        # Get current price
        current_price = self.db.get_latest_price(signal.symbol)
        if not current_price:
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(
            portfolio.total_value,
            current_price,
            signal.stop_loss if signal.stop_loss else current_price * 0.98,
            risk_params['risk_per_trade']
        )
        
        # Apply constraints
        position_value = position_size * current_price
        position_value = min(position_value, max_position_value)
        position_value = min(position_value, available_capital * 0.95)
        
        if position_value < 1000:  # Minimum position size
            return None
        
        shares = int(position_value / current_price)
        
        return {
            'action': 'BUY',
            'symbol': signal.symbol,
            'shares': shares,
            'price': current_price,
            'position_value': shares * current_price,
            'reason': signal.reason,
            'signal_strength': signal.signal_strength,
            'confidence': signal.confidence,
            'target_price': signal.target_price,
            'stop_loss': signal.stop_loss,
            'expected_return': ((signal.target_price - current_price) / current_price * 100) 
                              if signal.target_price else None,
            'risk_reward_ratio': self._calculate_risk_reward(
                current_price, signal.target_price, signal.stop_loss
            ),
            'priority': signal.signal_strength * signal.confidence,
            'generated_at': datetime.now()
        }
    
    def _generate_exit_recommendation(self, signal: Signal, 
                                     position: Position,
                                     portfolio: Portfolio) -> Optional[Dict]:
        """Generate exit recommendation"""
        
        current_price = self.db.get_latest_price(signal.symbol)
        if not current_price:
            return None
        
        # Calculate P&L
        pnl = (current_price - position.average_price) * position.shares
        pnl_percent = (pnl / position.cost_basis) * 100
        
        return {
            'action': 'SELL',
            'symbol': signal.symbol,
            'shares': position.shares,
            'price': current_price,
            'position_value': position.shares * current_price,
            'reason': signal.reason,
            'signal_strength': signal.signal_strength,
            'confidence': signal.confidence,
            'current_pnl': pnl,
            'current_pnl_percent': pnl_percent,
            'holding_period': (datetime.now() - position.opened_date).days,
            'priority': signal.signal_strength * signal.confidence,
            'generated_at': datetime.now()
        }
    
    def _calculate_position_size(self, portfolio_value: float,
                                entry_price: float,
                                stop_loss: float,
                                risk_per_trade: float) -> int:
        """Calculate position size based on risk"""
        
        # Risk amount
        risk_amount = portfolio_value * risk_per_trade
        
        # Risk per share
        risk_per_share = abs(entry_price - stop_loss)
        
        if risk_per_share == 0:
            return 0
        
        # Position size
        shares = int(risk_amount / risk_per_share)
        
        return shares
    
    def _calculate_risk_reward(self, entry: float, 
                              target: Optional[float],
                              stop: Optional[float]) -> Optional[float]:
        """Calculate risk/reward ratio"""
        
        if not target or not stop:
            return None
        
        potential_profit = target - entry
        potential_loss = entry - stop
        
        if potential_loss == 0:
            return None
        
        return potential_profit / potential_loss
    
    def _prioritize_recommendations(self, recommendations: List[Dict]) -> List[Dict]:
        """Prioritize recommendations"""
        
        # Add scoring
        for rec in recommendations:
            score = 0
            
            # Signal strength (40%)
            score += rec.get('signal_strength', 0) * 0.4
            
            # Confidence (30%)
            score += rec.get('confidence', 0) * 0.3
            
            # Risk/Reward (20%)
            rr = rec.get('risk_reward_ratio', 1)
            if rr and rr > 0:
                score += min(rr / 3, 1) * 0.2  # Normalize to 0-1
            
            # Expected return (10%)
            exp_return = rec.get('expected_return', 0)
            if exp_return:
                score += min(abs(exp_return) / 10, 1) * 0.1
            
            rec['score'] = score
        
        # Sort by score
        recommendations.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        return recommendations
    
    def filter_by_risk(self, recommendations: List[Dict],
                      max_risk: float = 0.05) -> List[Dict]:
        """Filter recommendations by risk level"""
        
        filtered = []
        total_risk = 0
        
        for rec in recommendations:
            if rec['action'] == 'BUY':
                position_risk = rec.get('position_value', 0) * 0.02  # 2% risk per trade
                
                if total_risk + position_risk <= max_risk:
                    filtered.append(rec)
                    total_risk += position_risk
        
        return filtered
    
    def generate_rebalancing_recommendations(self, portfolio: Portfolio,
                                           target_allocations: Dict[str, float]) -> List[Dict]:
        """Generate rebalancing recommendations"""
        
        recommendations = []
        current_value = portfolio.total_value
        
        # Current allocations
        current_allocations = {}
        for position in portfolio.positions:
            if position.status == 'open':
                weight = position.market_value / current_value
                current_allocations[position.symbol] = weight
        
        # Calculate adjustments needed
        for symbol, target_weight in target_allocations.items():
            current_weight = current_allocations.get(symbol, 0)
            weight_diff = target_weight - current_weight
            
            if abs(weight_diff) > 0.01:  # 1% threshold
                value_diff = weight_diff * current_value
                current_price = self.db.get_latest_price(symbol)
                
                if current_price:
                    shares = int(abs(value_diff) / current_price)
                    
                    if shares > 0:
                        recommendations.append({
                            'action': 'BUY' if weight_diff > 0 else 'SELL',
                            'symbol': symbol,
                            'shares': shares,
                            'price': current_price,
                            'reason': f"Rebalance to {target_weight*100:.1f}% allocation",
                            'current_weight': current_weight,
                            'target_weight': target_weight,
                            'weight_diff': weight_diff
                        })
        
        return recommendations