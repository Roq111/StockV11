"""
Signal Generation Module
Generate trading signals from technical analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging
from enum import Enum

from models import Signal, SignalType

logger = logging.getLogger(__name__)

class SignalGenerator:
    """Generate trading signals from technical indicators"""
    
    def __init__(self, db, config=None):
        self.db = db
        self.config = config
        
        # Signal weights
        self.weights = {
            'trend': 0.25,
            'momentum': 0.20,
            'volume': 0.15,
            'volatility': 0.15,
            'support_resistance': 0.15,
            'pattern': 0.10
        }
    
    def generate_signals(self, symbol: str, df: pd.DataFrame) -> List[Signal]:
        """Generate all signals for a symbol"""
        signals = []
        
        # Check data availability
        if len(df) < 50:
            return signals
        
        # Generate different signal types
        trend_signal = self._trend_signals(symbol, df)
        if trend_signal:
            signals.append(trend_signal)
        
        momentum_signal = self._momentum_signals(symbol, df)
        if momentum_signal:
            signals.append(momentum_signal)
        
        volume_signal = self._volume_signals(symbol, df)
        if volume_signal:
            signals.append(volume_signal)
        
        volatility_signal = self._volatility_signals(symbol, df)
        if volatility_signal:
            signals.append(volatility_signal)
        
        pattern_signal = self._pattern_signals(symbol, df)
        if pattern_signal:
            signals.append(pattern_signal)
        
        # Combine signals
        if signals:
            combined = self._combine_signals(signals)
            return [combined]
        
        return []
    
    def _trend_signals(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Generate trend-based signals"""
        latest = df.iloc[-1]
        
        # Moving average crossover
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            sma_20 = latest['sma_20']
            sma_50 = latest['sma_50']
            prev_sma_20 = df.iloc[-2]['sma_20']
            prev_sma_50 = df.iloc[-2]['sma_50']
            
            # Golden cross
            if sma_20 > sma_50 and prev_sma_20 <= prev_sma_50:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    signal_strength=0.8,
                    action="BUY",
                    reason="Golden Cross - SMA20 crossed above SMA50",
                    target_price=latest['close'] * 1.05,
                    stop_loss=latest['close'] * 0.98,
                    confidence=0.7
                )
            
            # Death cross
            elif sma_20 < sma_50 and prev_sma_20 >= prev_sma_50:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    signal_strength=0.8,
                    action="SELL",
                    reason="Death Cross - SMA20 crossed below SMA50",
                    target_price=latest['close'] * 0.95,
                    stop_loss=latest['close'] * 1.02,
                    confidence=0.7
                )
        
        # Trend strength
        if 'sma_200' in df.columns:
            if latest['close'] > latest['sma_200'] * 1.02:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    signal_strength=0.6,
                    action="BUY",
                    reason="Strong uptrend - Price above SMA200",
                    confidence=0.6
                )
        
        return None
    
    def _momentum_signals(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Generate momentum-based signals"""
        latest = df.iloc[-1]
        
        # RSI signals
        if 'rsi' in df.columns:
            rsi = latest['rsi']
            prev_rsi = df.iloc[-2]['rsi']
            
            # Oversold bounce
            if rsi > 30 and prev_rsi <= 30:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    signal_strength=0.7,
                    action="BUY",
                    reason=f"RSI oversold bounce ({rsi:.1f})",
                    target_price=latest['close'] * 1.03,
                    stop_loss=latest['close'] * 0.97,
                    confidence=0.65
                )
            
            # Overbought reversal
            elif rsi < 70 and prev_rsi >= 70:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    signal_strength=0.7,
                    action="SELL",
                    reason=f"RSI overbought reversal ({rsi:.1f})",
                    confidence=0.65
                )
        
        # MACD signals
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = latest['macd']
            signal_line = latest['macd_signal']
            prev_macd = df.iloc[-2]['macd']
            prev_signal = df.iloc[-2]['macd_signal']
            
            # MACD crossover
            if macd > signal_line and prev_macd <= prev_signal:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    signal_strength=0.75,
                    action="BUY",
                    reason="MACD bullish crossover",
                    confidence=0.7
                )
            elif macd < signal_line and prev_macd >= prev_signal:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    signal_strength=0.75,
                    action="SELL",
                    reason="MACD bearish crossover",
                    confidence=0.7
                )
        
        return None
    
    def _volume_signals(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Generate volume-based signals"""
        latest = df.iloc[-1]
        
        if 'volume' in df.columns and 'volume_sma' in df.columns:
            volume_ratio = latest['volume'] / latest['volume_sma']
            
            # Volume spike with price increase
            if volume_ratio > 2.0 and latest['close'] > latest['open']:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    signal_strength=0.6,
                    action="BUY",
                    reason=f"Volume spike ({volume_ratio:.1f}x average) with price increase",
                    confidence=0.6
                )
            
            # Volume spike with price decrease
            elif volume_ratio > 2.0 and latest['close'] < latest['open']:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    signal_strength=0.6,
                    action="SELL",
                    reason=f"Volume spike ({volume_ratio:.1f}x average) with price decrease",
                    confidence=0.6
                )
        
        return None
    
    def _volatility_signals(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Generate volatility-based signals"""
        latest = df.iloc[-1]
        
        # Bollinger Bands signals
        if all(col in df.columns for col in ['bb_upper', 'bb_lower', 'bb_middle']):
            price = latest['close']
            upper = latest['bb_upper']
            lower = latest['bb_lower']
            middle = latest['bb_middle']
            
            # Price at lower band - potential bounce
            if price <= lower * 1.01:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.BUY,
                    signal_strength=0.65,
                    action="BUY",
                    reason="Price at Bollinger Band lower band",
                    target_price=middle,
                    stop_loss=lower * 0.98,
                    confidence=0.6
                )
            
            # Price at upper band - potential reversal
            elif price >= upper * 0.99:
                return Signal(
                    symbol=symbol,
                    signal_type=SignalType.SELL,
                    signal_strength=0.65,
                    action="SELL",
                    reason="Price at Bollinger Band upper band",
                    target_price=middle,
                    stop_loss=upper * 1.02,
                    confidence=0.6
                )
        
        return None
    
    def _pattern_signals(self, symbol: str, df: pd.DataFrame) -> Optional[Signal]:
        """Generate pattern-based signals"""
        
        # Detect support/resistance breakouts
        recent_high = df['high'].rolling(20).max().iloc[-2]
        recent_low = df['low'].rolling(20).min().iloc[-2]
        latest = df.iloc[-1]
        
        # Breakout above resistance
        if latest['close'] > recent_high:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.BUY,
                signal_strength=0.8,
                action="BUY",
                reason=f"Breakout above resistance at ${recent_high:.2f}",
                target_price=latest['close'] * 1.05,
                stop_loss=recent_high * 0.98,
                confidence=0.75
            )
        
        # Breakdown below support
        elif latest['close'] < recent_low:
            return Signal(
                symbol=symbol,
                signal_type=SignalType.SELL,
                signal_strength=0.8,
                action="SELL",
                reason=f"Breakdown below support at ${recent_low:.2f}",
                confidence=0.75
            )
        
        return None
    
    def _combine_signals(self, signals: List[Signal]) -> Signal:
        """Combine multiple signals into one"""
        
        # Count buy and sell signals
        buy_signals = [s for s in signals if s.signal_type == SignalType.BUY]
        sell_signals = [s for s in signals if s.signal_type == SignalType.SELL]
        
        # Determine overall signal
        if len(buy_signals) > len(sell_signals):
            signal_type = SignalType.BUY
            action = "BUY"
            relevant_signals = buy_signals
        elif len(sell_signals) > len(buy_signals):
            signal_type = SignalType.SELL
            action = "SELL"
            relevant_signals = sell_signals
        else:
            signal_type = SignalType.HOLD
            action = "HOLD"
            relevant_signals = signals
        
        # Calculate combined strength
        total_strength = sum(s.signal_strength for s in relevant_signals)
        avg_strength = total_strength / len(relevant_signals) if relevant_signals else 0
        
        # Combine reasons
        reasons = [s.reason for s in relevant_signals[:3]]  # Top 3 reasons
        combined_reason = f"{len(relevant_signals)} signals: " + "; ".join(reasons)
        
        # Average confidence
        avg_confidence = np.mean([s.confidence for s in relevant_signals])
        
        return Signal(
            symbol=signals[0].symbol,
            signal_type=signal_type,
            signal_strength=min(1.0, avg_strength),
            action=action,
            reason=combined_reason,
            confidence=avg_confidence
        )
    
    def rank_signals(self, signals: List[Signal]) -> List[Signal]:
        """Rank signals by strength and confidence"""
        
        # Calculate composite score
        for signal in signals:
            signal.metadata['score'] = (
                signal.signal_strength * 0.6 +
                signal.confidence * 0.4
            )
        
        # Sort by score
        signals.sort(key=lambda x: x.metadata.get('score', 0), reverse=True)
        
        return signals