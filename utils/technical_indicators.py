"""
Technical Indicators Module
Comprehensive technical analysis indicators
"""

import pandas as pd
import numpy as np
import talib
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class TechnicalIndicators:
    """Calculate technical indicators for analysis"""
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators"""
        
        # Make copy to avoid modifying original
        df = df.copy()
        
        # Trend indicators
        df = self.add_moving_averages(df)
        df = self.add_macd(df)
        df = self.add_bollinger_bands(df)
        
        # Momentum indicators
        df = self.add_rsi(df)
        df = self.add_stochastic(df)
        df = self.add_momentum(df)
        
        # Volume indicators
        df = self.add_volume_indicators(df)
        
        # Volatility indicators
        df = self.add_atr(df)
        df = self.add_volatility(df)
        
        # Pattern recognition
        df = self.add_candlestick_patterns(df)
        
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add moving averages"""
        
        # Simple moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        
        # Exponential moving averages
        for period in [12, 26, 50]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        # Weighted moving average
        df['wma_20'] = talib.WMA(df['close'].values, timeperiod=20)
        
        # Hull moving average
        df['hma_20'] = self._calculate_hull_ma(df['close'], 20)
        
        return df
    
    def add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD indicator"""
        
        macd, signal, hist = talib.MACD(
            df['close'].values,
            fastperiod=12,
            slowperiod=26,
            signalperiod=9
        )
        
        df['macd'] = macd
        df['macd_signal'] = signal
        df['macd_hist'] = hist
        
        # MACD crossover signals
        df['macd_cross_up'] = ((df['macd'] > df['macd_signal']) & 
                               (df['macd'].shift(1) <= df['macd_signal'].shift(1)))
        df['macd_cross_down'] = ((df['macd'] < df['macd_signal']) & 
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1)))
        
        return df
    
    def add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands"""
        
        upper, middle, lower = talib.BBANDS(
            df['close'].values,
            timeperiod=20,
            nbdevup=2,
            nbdevdn=2,
            matype=0
        )
        
        df['bb_upper'] = upper
        df['bb_middle'] = middle
        df['bb_lower'] = lower
        df['bb_width'] = upper - lower
        df['bb_percent'] = (df['close'] - lower) / (upper - lower)
        
        # Squeeze indicator
        keltner_upper, keltner_lower = self._calculate_keltner_channels(df)
        df['bb_squeeze'] = (df['bb_upper'] < keltner_upper) & (df['bb_lower'] > keltner_lower)
        
        return df
    
    def add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add RSI indicator"""
        
        df['rsi'] = talib.RSI(df['close'].values, timeperiod=14)
        df['rsi_5'] = talib.RSI(df['close'].values, timeperiod=5)
        df['rsi_21'] = talib.RSI(df['close'].values, timeperiod=21)
        
        # RSI divergence
        df['rsi_divergence'] = self._calculate_divergence(
            df['close'], df['rsi'], lookback=20
        )
        
        # Stochastic RSI
        stoch_rsi = talib.STOCHRSI(df['close'].values, timeperiod=14)
        df['stoch_rsi'] = stoch_rsi[0]
        
        return df
    
    def add_stochastic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Stochastic oscillator"""
        
        slowk, slowd = talib.STOCH(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            fastk_period=14,
            slowk_period=3,
            slowk_matype=0,
            slowd_period=3,
            slowd_matype=0
        )
        
        df['stoch_k'] = slowk
        df['stoch_d'] = slowd
        
        # Stochastic signals
        df['stoch_oversold'] = (df['stoch_k'] < 20) & (df['stoch_d'] < 20)
        df['stoch_overbought'] = (df['stoch_k'] > 80) & (df['stoch_d'] > 80)
        
        return df
    
    def add_momentum(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add momentum indicators"""
        
        # Rate of change
        df['roc'] = talib.ROC(df['close'].values, timeperiod=10)
        
        # Commodity Channel Index
        df['cci'] = talib.CCI(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=20
        )
        
        # Williams %R
        df['williams_r'] = talib.WILLR(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=14
        )
        
        # Ultimate Oscillator
        df['ultimate_osc'] = talib.ULTOSC(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod1=7,
            timeperiod2=14,
            timeperiod3=28
        )
        
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume-based indicators"""
        
        # On Balance Volume
        df['obv'] = talib.OBV(df['close'].values, df['volume'].values)
        
        # Volume SMA
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        
        # Accumulation/Distribution
        df['ad'] = talib.AD(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values
        )
        
        # Chaikin Money Flow
        df['cmf'] = self._calculate_cmf(df)
        
        # Volume Price Trend
        df['vpt'] = self._calculate_vpt(df)
        
        # Money Flow Index
        df['mfi'] = talib.MFI(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            df['volume'].values,
            timeperiod=14
        )
        
        return df
    
    def add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Average True Range"""
        
        df['atr'] = talib.ATR(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            timeperiod=14
        )
        
        # ATR percentage
        df['atr_percent'] = (df['atr'] / df['close']) * 100
        
        # Chandelier Exit
        df['chandelier_long'] = df['high'].rolling(22).max() - df['atr'] * 3
        df['chandelier_short'] = df['low'].rolling(22).min() + df['atr'] * 3
        
        return df
    
    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volatility indicators"""
        
        # Historical volatility
        df['returns'] = df['close'].pct_change()
        df['volatility_20'] = df['returns'].rolling(20).std() * np.sqrt(252)
        df['volatility_60'] = df['returns'].rolling(60).std() * np.sqrt(252)
        
        # Parkinson volatility
        df['parkinson_vol'] = self._calculate_parkinson_volatility(df)
        
        # Garman-Klass volatility
        df['gk_vol'] = self._calculate_garman_klass_volatility(df)
        
        return df
    
    def add_candlestick_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add candlestick pattern recognition"""
        
        # Bullish patterns
        df['cdl_hammer'] = talib.CDLHAMMER(
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values
        )
        
        df['cdl_bullish_engulfing'] = talib.CDLENGULFING(
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values
        )
        
        df['cdl_morning_star'] = talib.CDLMORNINGSTAR(
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values
        )
        
        # Bearish patterns
        df['cdl_shooting_star'] = talib.CDLSHOOTINGSTAR(
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values
        )
        
        df['cdl_bearish_engulfing'] = talib.CDLENGULFING(
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values
        ) * -1  # Invert for bearish
        
        df['cdl_evening_star'] = talib.CDLEVENINGSTAR(
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values
        )
        
        # Neutral patterns
        df['cdl_doji'] = talib.CDLDOJI(
            df['open'].values, df['high'].values,
            df['low'].values, df['close'].values
        )
        
        return df
    
    def _calculate_hull_ma(self, series: pd.Series, period: int) -> pd.Series:
        """Calculate Hull Moving Average"""
        
        half_period = period // 2
        sqrt_period = int(np.sqrt(period))
        
        wma_half = talib.WMA(series.values, timeperiod=half_period)
        wma_full = talib.WMA(series.values, timeperiod=period)
        
        raw_hma = 2 * wma_half - wma_full
        hma = talib.WMA(raw_hma, timeperiod=sqrt_period)
        
        return pd.Series(hma, index=series.index)
    
    def _calculate_keltner_channels(self, df: pd.DataFrame,
                                   period: int = 20,
                                   multiplier: float = 2) -> Tuple[pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        
        middle = df['close'].ewm(span=period, adjust=False).mean()
        atr = talib.ATR(df['high'].values, df['low'].values,
                       df['close'].values, timeperiod=period)
        
        upper = middle + multiplier * atr
        lower = middle - multiplier * atr
        
        return upper, lower
    
    def _calculate_divergence(self, price: pd.Series, indicator: pd.Series,
                            lookback: int = 20) -> pd.Series:
        """Calculate divergence between price and indicator"""
        
        divergence = pd.Series(0, index=price.index)
        
        for i in range(lookback, len(price)):
            price_slice = price.iloc[i-lookback:i]
            indicator_slice = indicator.iloc[i-lookback:i]
            
            # Bearish divergence: price makes higher high, indicator doesn't
            if (price.iloc[i] > price_slice.max() and
                indicator.iloc[i] <= indicator_slice.max()):
                divergence.iloc[i] = -1
            
            # Bullish divergence: price makes lower low, indicator doesn't
            elif (price.iloc[i] < price_slice.min() and
                  indicator.iloc[i] >= indicator_slice.min()):
                divergence.iloc[i] = 1
        
        return divergence
    
    def _calculate_cmf(self, df: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Chaikin Money Flow"""
        
        mf_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / \
                       (df['high'] - df['low'])
        mf_volume = mf_multiplier * df['volume']
        
        cmf = mf_volume.rolling(period).sum() / df['volume'].rolling(period).sum()
        
        return cmf
    
    def _calculate_vpt(self, df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Price Trend"""
        
        price_change = df['close'].pct_change()
        vpt = (price_change * df['volume']).cumsum()
        
        return vpt
    
    def _calculate_parkinson_volatility(self, df: pd.DataFrame,
                                       period: int = 20) -> pd.Series:
        """Calculate Parkinson volatility estimator"""
        
        log_hl = np.log(df['high'] / df['low'])
        vol = np.sqrt((log_hl ** 2).rolling(period).mean() / (4 * np.log(2))) * np.sqrt(252)
        
        return vol
    
    def _calculate_garman_klass_volatility(self, df: pd.DataFrame,
                                          period: int = 20) -> pd.Series:
        """Calculate Garman-Klass volatility estimator"""
        
        log_hl = np.log(df['high'] / df['low'])
        log_co = np.log(df['close'] / df['open'])
        
        gk = np.sqrt(
            0.5 * (log_hl ** 2).rolling(period).mean() -
            (2 * np.log(2) - 1) * (log_co ** 2).rolling(period).mean()
        ) * np.sqrt(252)
        
        return gk