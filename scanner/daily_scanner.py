"""
Daily Scanner Module - Fixed for Your Database Schema
Works with your actual tables: stocks, price_data, technical_indicators, signals
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any
import logging
import sqlite3

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    print("Warning: TA-Lib not installed. Using fallback calculations.")

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DailyScanner:
    """Daily scanner compatible with your database schema"""
    
    def __init__(self, db_manager):
        self.db = db_manager
        self.setup_scan_tables()
        
    def setup_scan_tables(self):
        """Ensure all required tables exist"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check which tables exist
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table'
                """)
                existing_tables = [row[0] for row in cursor.fetchall()]
                logger.info(f"Existing tables: {existing_tables}")
                
                # Create scan_results table if it doesn't exist
                if 'scan_results' not in existing_tables:
                    cursor.execute('''
                        CREATE TABLE scan_results (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT NOT NULL,
                            signal_type TEXT NOT NULL,
                            signal_strength REAL,
                            indicator TEXT,
                            reason TEXT,
                            price REAL,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_scan_results_timestamp 
                        ON scan_results(timestamp DESC)
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_scan_results_symbol 
                        ON scan_results(symbol)
                    ''')
                    
                    conn.commit()
                    logger.info("Created scan_results table")
                
                # Create missing tables if needed
                if 'stocks' not in existing_tables:
                    cursor.execute('''
                        CREATE TABLE stocks (
                            symbol TEXT PRIMARY KEY,
                            name TEXT,
                            sector TEXT,
                            industry TEXT,
                            market_cap REAL,
                            active BOOLEAN DEFAULT 1
                        )
                    ''')
                    conn.commit()
                    logger.info("Created stocks table")
                
                if 'price_data' not in existing_tables:
                    cursor.execute('''
                        CREATE TABLE price_data (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT,
                            date DATE,
                            open REAL,
                            high REAL,
                            low REAL,
                            close REAL,
                            adjusted_close REAL,
                            volume INTEGER,
                            FOREIGN KEY (symbol) REFERENCES stocks(symbol),
                            UNIQUE(symbol, date)
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_price_data_symbol_date 
                        ON price_data(symbol, date)
                    ''')
                    
                    conn.commit()
                    logger.info("Created price_data table")
                
                if 'technical_indicators' not in existing_tables:
                    cursor.execute('''
                        CREATE TABLE technical_indicators (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT,
                            date DATE,
                            sma_20 REAL,
                            sma_50 REAL,
                            sma_200 REAL,
                            ema_12 REAL,
                            ema_26 REAL,
                            rsi REAL,
                            macd REAL,
                            macd_signal REAL,
                            macd_hist REAL,
                            bb_upper REAL,
                            bb_middle REAL,
                            bb_lower REAL,
                            atr REAL,
                            volume_sma REAL,
                            FOREIGN KEY (symbol) REFERENCES stocks(symbol),
                            UNIQUE(symbol, date)
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_technical_indicators_symbol_date 
                        ON technical_indicators(symbol, date)
                    ''')
                    
                    conn.commit()
                    logger.info("Created technical_indicators table")
                
                if 'signals' not in existing_tables:
                    cursor.execute('''
                        CREATE TABLE signals (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            symbol TEXT,
                            signal_type TEXT,
                            signal_strength REAL,
                            reason TEXT,
                            generated_date TIMESTAMP,
                            expiry_date TIMESTAMP,
                            status TEXT,
                            metadata TEXT,
                            FOREIGN KEY (symbol) REFERENCES stocks(symbol)
                        )
                    ''')
                    
                    cursor.execute('''
                        CREATE INDEX IF NOT EXISTS idx_signals_symbol 
                        ON signals(symbol)
                    ''')
                    
                    conn.commit()
                    logger.info("Created signals table")
                    
        except Exception as e:
            logger.error(f"Error in setup_scan_tables: {e}")
    
    def scan(self, portfolio_name: str = "default") -> List[Dict]:
        return self.scan_with_optimized_config(portfolio_name)
        """Run daily scan"""
        logger.info(f"Starting daily scan for portfolio: {portfolio_name}")
        
        # Get list of stocks to scan
        stocks = self._get_scannable_stocks()
        
        if not stocks:
            logger.warning("No stocks found to scan. Please import stock data first.")
            print("\n⚠️  No stocks found in database.")
            print("Please run 'Import Data' from main menu first to download stock data.")
            return []
        
        logger.info(f"Found {len(stocks)} stocks to scan")
        
        all_signals = []
        errors = []
        successful_scans = 0
        
        # Progress tracking
        print(f"\nScanning {len(stocks)} stocks...")
        print("Progress: ", end="")
        
        # Scan each stock
        for i, symbol in enumerate(stocks, 1):
            if i % 10 == 0:
                print(f"{i}...", end="", flush=True)
            
            try:
                signals = self._scan_stock(symbol)
                if signals:
                    all_signals.extend(signals)
                    self._save_signals_to_db(signals)
                    successful_scans += 1
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {str(e)}")
                errors.append({'symbol': symbol, 'error': str(e)})
        
        print(" Done!")
        
        # Log summary
        print(f"""
        ╔════════════════════════════════════╗
        ║         SCAN COMPLETE              ║
        ╠════════════════════════════════════╣
        ║ Total stocks scanned: {len(stocks):13}║
        ║ Successful scans:     {successful_scans:13}║
        ║ Total signals found:  {len(all_signals):13}║
        ║ Errors encountered:   {len(errors):13}║
        ╚════════════════════════════════════╝
        """)
        
        return all_signals
    
    def _get_scannable_stocks(self) -> List[str]:
        """Get list of stocks from database"""
        stocks = []
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get stocks with recent price data
                cursor.execute("""
                    SELECT DISTINCT s.symbol 
                    FROM stocks s
                    WHERE EXISTS (
                        SELECT 1 FROM price_data pd 
                        WHERE pd.symbol = s.symbol 
                        AND pd.date >= date('now', '-100 days')
                        AND pd.close > 0
                        AND pd.volume > 0
                    )
                    AND s.symbol NOT LIKE '%-%'
                    AND s.symbol NOT LIKE '%.%'
                    AND LENGTH(s.symbol) <= 5
                    ORDER BY s.symbol
                    LIMIT 500
                """)
                
                results = cursor.fetchall()
                stocks = [row[0] for row in results] if results else []
                
                if not stocks:
                    # Try simpler query
                    cursor.execute("""
                        SELECT DISTINCT symbol 
                        FROM price_data 
                        WHERE date >= date('now', '-100 days')
                        AND close > 0
                        LIMIT 100
                    """)
                    results = cursor.fetchall()
                    stocks = [row[0] for row in results] if results else []
                
        except Exception as e:
            logger.error(f"Error getting stocks: {e}")
            
        return stocks
    
    def _scan_stock(self, symbol: str) -> List[Dict]:
        """Scan individual stock for signals"""
        try:
            # Get price data from price_data table
            df = self._get_price_data(symbol)
            
            if df is None or len(df) < 30:
                return []
            
            # Check for existing technical indicators
            indicators = self._get_technical_indicators(symbol, df)
            
            if not indicators:
                # Calculate if not found
                if TALIB_AVAILABLE:
                    indicators = self._calculate_indicators_talib(df)
                else:
                    indicators = self._calculate_indicators_fallback(df)
                
                # Save calculated indicators
                self._save_technical_indicators(symbol, df, indicators)
            
            # Generate signals
            signals = self._generate_signals(symbol, df, indicators)
            
            return signals
            
        except Exception as e:
            logger.debug(f"Error scanning {symbol}: {e}")
            return []
    
    def _get_price_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Get price data from price_data table"""
        try:
            with self.db.get_connection() as conn:
                query = """
                    SELECT date, open, high, low, close, adjusted_close, volume
                    FROM price_data
                    WHERE symbol = ?
                    AND date >= date('now', '-100 days')
                    AND close > 0
                    ORDER BY date ASC
                """
                
                df = pd.read_sql_query(query, conn, params=(symbol,))
                
                if df.empty or len(df) < 30:
                    return None
                
                # Convert date to datetime and set as index
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
                
                # Use adjusted_close if available, otherwise use close
                if 'adjusted_close' in df.columns and df['adjusted_close'].notna().any():
                    df['close'] = df['adjusted_close'].fillna(df['close'])
                
                # Ensure numeric types
                numeric_columns = ['open', 'high', 'low', 'close', 'volume']
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Remove NaN rows
                df = df.dropna()
                
                return df if len(df) >= 30 else None
                
        except Exception as e:
            logger.debug(f"Error getting price data for {symbol}: {e}")
            return None
    
    def _get_technical_indicators(self, symbol: str, df: pd.DataFrame) -> Optional[Dict]:
        """Get technical indicators from database if available"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                latest_date = df.index[-1].strftime('%Y-%m-%d')
                
                # Check if we have recent indicators
                cursor.execute("""
                    SELECT sma_20, sma_50, rsi, macd, macd_signal, 
                           bb_upper, bb_middle, bb_lower
                    FROM technical_indicators
                    WHERE symbol = ? AND date = ?
                """, (symbol, latest_date))
                
                row = cursor.fetchone()
                if row and all(v is not None for v in row):
                    # Build indicators dict from database
                    indicators = {
                        'sma_20': np.array([row[0]]),
                        'sma_50': np.array([row[1]]),
                        'rsi': np.array([row[2]]),
                        'macd': np.array([row[3]]),
                        'macd_signal': np.array([row[4]]),
                        'bb_upper': np.array([row[5]]),
                        'bb_middle': np.array([row[6]]),
                        'bb_lower': np.array([row[7]])
                    }
                    return indicators
                    
        except Exception as e:
            logger.debug(f"Error getting indicators for {symbol}: {e}")
        
        return None
    
    def _calculate_indicators_talib(self, df: pd.DataFrame) -> Dict:
        """Calculate indicators using TA-Lib"""
        indicators = {}
        
        try:
            close = np.asarray(df['close'].values, dtype=np.float64)
            high = np.asarray(df['high'].values, dtype=np.float64)
            low = np.asarray(df['low'].values, dtype=np.float64)
            volume = np.asarray(df['volume'].values, dtype=np.float64)
            
            # Calculate all indicators
            indicators['rsi'] = talib.RSI(close, timeperiod=14)
            indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = talib.MACD(close)
            indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = talib.BBANDS(close)
            indicators['sma_20'] = talib.SMA(close, timeperiod=20)
            indicators['sma_50'] = talib.SMA(close, timeperiod=50) if len(close) >= 50 else np.full_like(close, np.nan)
            indicators['sma_200'] = talib.SMA(close, timeperiod=200) if len(close) >= 200 else np.full_like(close, np.nan)
            indicators['ema_12'] = talib.EMA(close, timeperiod=12)
            indicators['ema_26'] = talib.EMA(close, timeperiod=26)
            indicators['atr'] = talib.ATR(high, low, close, timeperiod=14)
            indicators['obv'] = talib.OBV(close, volume)
            
        except Exception as e:
            logger.debug(f"TA-Lib error, using fallback: {e}")
            return self._calculate_indicators_fallback(df)
        
        return indicators
    
    def _calculate_indicators_fallback(self, df: pd.DataFrame) -> Dict:
        """Calculate indicators without TA-Lib"""
        close = df['close'].values
        
        indicators = {
            'rsi': self._rsi_fallback(close, 14),
            'sma_20': self._sma_fallback(close, 20),
            'sma_50': self._sma_fallback(close, 50) if len(close) >= 50 else np.full_like(close, np.nan),
            'sma_200': self._sma_fallback(close, 200) if len(close) >= 200 else np.full_like(close, np.nan),
            'ema_12': self._ema_fallback(close, 12),
            'ema_26': self._ema_fallback(close, 26)
        }
        
        # MACD
        indicators['macd'], indicators['macd_signal'], indicators['macd_hist'] = self._macd_fallback(close)
        
        # Bollinger Bands
        indicators['bb_upper'], indicators['bb_middle'], indicators['bb_lower'] = self._bbands_fallback(close)
        
        # ATR (simplified)
        indicators['atr'] = self._atr_fallback(df)
        
        # OBV (simplified)
        indicators['obv'] = self._obv_fallback(df)
        
        return indicators
    
    def _rsi_fallback(self, prices, period=14):
        """RSI calculation"""
        if len(prices) < period + 1:
            return np.full_like(prices, np.nan)
        
        deltas = np.diff(prices)
        seed = deltas[:period+1]
        up = seed[seed >= 0].sum() / period if len(seed[seed >= 0]) > 0 else 0
        down = -seed[seed < 0].sum() / period if len(seed[seed < 0]) > 0 else 0
        
        rsi = np.zeros_like(prices)
        rsi[:period] = np.nan
        
        if down == 0:
            rsi[period] = 100
        else:
            rs = up / down
            rsi[period] = 100 - (100 / (1 + rs))
        
        for i in range(period + 1, len(prices)):
            delta = deltas[i - 1]
            if delta > 0:
                upval = delta
                downval = 0
            else:
                upval = 0
                downval = -delta
            
            up = (up * (period - 1) + upval) / period
            down = (down * (period - 1) + downval) / period
            
            if down == 0:
                rsi[i] = 100
            else:
                rs = up / down
                rsi[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _sma_fallback(self, prices, period):
        """SMA calculation"""
        if len(prices) < period:
            return np.full_like(prices, np.nan)
        
        sma = np.zeros_like(prices)
        sma[:period-1] = np.nan
        
        for i in range(period-1, len(prices)):
            sma[i] = np.mean(prices[i-period+1:i+1])
        
        return sma
    
    def _ema_fallback(self, prices, period):
        """EMA calculation"""
        if len(prices) < period:
            return np.full_like(prices, np.nan)
        
        ema = np.zeros_like(prices)
        ema[:period-1] = np.nan
        ema[period-1] = np.mean(prices[:period])
        
        multiplier = 2 / (period + 1)
        
        for i in range(period, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema
    
    def _macd_fallback(self, prices):
        """MACD calculation"""
        if len(prices) < 26:
            nan_array = np.full_like(prices, np.nan)
            return nan_array, nan_array, nan_array
        
        ema12 = self._ema_fallback(prices, 12)
        ema26 = self._ema_fallback(prices, 26)
        macd = ema12 - ema26
        
        signal = np.full_like(prices, np.nan)
        if len(prices) >= 35:
            signal[33:] = self._ema_fallback(macd[25:], 9)[8:]
        
        hist = macd - signal
        
        return macd, signal, hist
    
    def _bbands_fallback(self, prices, period=20, std_dev=2):
        """Bollinger Bands calculation"""
        if len(prices) < period:
            nan_array = np.full_like(prices, np.nan)
            return nan_array, nan_array, nan_array
        
        middle = self._sma_fallback(prices, period)
        std = np.zeros_like(prices)
        std[:period-1] = np.nan
        
        for i in range(period-1, len(prices)):
            std[i] = np.std(prices[i-period+1:i+1])
        
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        
        return upper, middle, lower
    
    def _atr_fallback(self, df: pd.DataFrame, period=14):
        """ATR calculation"""
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.zeros(len(df))
        tr[0] = high[0] - low[0]
        
        for i in range(1, len(df)):
            hl = high[i] - low[i]
            hc = abs(high[i] - close[i-1])
            lc = abs(low[i] - close[i-1])
            tr[i] = max(hl, hc, lc)
        
        atr = np.zeros_like(tr)
        atr[:period] = np.nan
        atr[period] = np.mean(tr[:period+1])
        
        for i in range(period+1, len(tr)):
            atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
        
        return atr
    
    def _obv_fallback(self, df: pd.DataFrame):
        """OBV calculation"""
        close = df['close'].values
        volume = df['volume'].values
        
        obv = np.zeros_like(close)
        obv[0] = volume[0]
        
        for i in range(1, len(close)):
            if close[i] > close[i-1]:
                obv[i] = obv[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv[i] = obv[i-1] - volume[i]
            else:
                obv[i] = obv[i-1]
        
        return obv
    
    def _save_technical_indicators(self, symbol: str, df: pd.DataFrame, indicators: Dict):
        """Save calculated indicators to database"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                latest_date = df.index[-1].strftime('%Y-%m-%d')
                
                # Get latest values
                values = {
                    'sma_20': indicators.get('sma_20', [np.nan])[-1],
                    'sma_50': indicators.get('sma_50', [np.nan])[-1],
                    'sma_200': indicators.get('sma_200', [np.nan])[-1],
                    'ema_12': indicators.get('ema_12', [np.nan])[-1],
                    'ema_26': indicators.get('ema_26', [np.nan])[-1],
                    'rsi': indicators.get('rsi', [np.nan])[-1],
                    'macd': indicators.get('macd', [np.nan])[-1],
                    'macd_signal': indicators.get('macd_signal', [np.nan])[-1],
                    'macd_hist': indicators.get('macd_hist', [np.nan])[-1],
                    'bb_upper': indicators.get('bb_upper', [np.nan])[-1],
                    'bb_middle': indicators.get('bb_middle', [np.nan])[-1],
                    'bb_lower': indicators.get('bb_lower', [np.nan])[-1],
                    'atr': indicators.get('atr', [np.nan])[-1]
                }
                
                # Insert or update
                cursor.execute("""
                    INSERT OR REPLACE INTO technical_indicators
                    (symbol, date, sma_20, sma_50, sma_200, ema_12, ema_26,
                     rsi, macd, macd_signal, macd_hist, bb_upper, bb_middle, bb_lower, atr)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (symbol, latest_date, 
                      values['sma_20'], values['sma_50'], values['sma_200'],
                      values['ema_12'], values['ema_26'], values['rsi'],
                      values['macd'], values['macd_signal'], values['macd_hist'],
                      values['bb_upper'], values['bb_middle'], values['bb_lower'],
                      values['atr']))
                
                conn.commit()
                
        except Exception as e:
            logger.debug(f"Error saving indicators for {symbol}: {e}")
    
    def _generate_signals(self, symbol: str, df: pd.DataFrame, indicators: Dict) -> List[Dict]:
        """Generate trading signals"""
        signals = []
        
        try:
            if len(df) < 2:
                return signals
            
            latest_close = df['close'].iloc[-1]
            prev_close = df['close'].iloc[-2]
            latest_volume = df['volume'].iloc[-1]
            avg_volume = df['volume'].rolling(window=min(20, len(df)), min_periods=5).mean().iloc[-1]
            
            # RSI signals
            if 'rsi' in indicators:
                rsi_value = indicators['rsi'][-1] if isinstance(indicators['rsi'], np.ndarray) else indicators['rsi']
                if not np.isnan(rsi_value):
                    if rsi_value < 30:
                        signals.append(self._create_signal(
                            symbol, 'BUY', 'RSI', rsi_value,
                            f'RSI oversold at {rsi_value:.1f}',
                            (30 - rsi_value) / 30, latest_close
                        ))
                    elif rsi_value > 70:
                        signals.append(self._create_signal(
                            symbol, 'SELL', 'RSI', rsi_value,
                            f'RSI overbought at {rsi_value:.1f}',
                            (rsi_value - 70) / 30, latest_close
                        ))
            
            # MACD signals
            if all(k in indicators for k in ['macd', 'macd_signal']):
                macd_curr = indicators['macd'][-1] if isinstance(indicators['macd'], np.ndarray) else indicators['macd']
                signal_curr = indicators['macd_signal'][-1] if isinstance(indicators['macd_signal'], np.ndarray) else indicators['macd_signal']
                
                if not any(np.isnan([macd_curr, signal_curr])) and len(df) > 1:
                    # Check for crossovers (need previous values)
                    if len(indicators['macd']) > 1 and len(indicators['macd_signal']) > 1:
                        macd_prev = indicators['macd'][-2]
                        signal_prev = indicators['macd_signal'][-2]
                        
                        if macd_prev <= signal_prev and macd_curr > signal_curr:
                            signals.append(self._create_signal(
                                symbol, 'BUY', 'MACD', macd_curr - signal_curr,
                                'MACD bullish crossover',
                                min(1.0, abs(macd_curr - signal_curr) * 100), latest_close
                            ))
                        elif macd_prev >= signal_prev and macd_curr < signal_curr:
                            signals.append(self._create_signal(
                                symbol, 'SELL', 'MACD', signal_curr - macd_curr,
                                'MACD bearish crossover',
                                min(1.0, abs(signal_curr - macd_curr) * 100), latest_close
                            ))
            
            # Bollinger Band signals
            if all(k in indicators for k in ['bb_upper', 'bb_lower']):
                bb_upper = indicators['bb_upper'][-1] if isinstance(indicators['bb_upper'], np.ndarray) else indicators['bb_upper']
                bb_lower = indicators['bb_lower'][-1] if isinstance(indicators['bb_lower'], np.ndarray) else indicators['bb_lower']
                
                if not any(np.isnan([bb_upper, bb_lower])):
                    if latest_close <= bb_lower * 1.01:
                        signals.append(self._create_signal(
                            symbol, 'BUY', 'BB', (bb_lower - latest_close) / latest_close * 100,
                            'Price at lower Bollinger Band',
                            min(1.0, abs(bb_lower - latest_close) / latest_close * 10), latest_close
                        ))
                    elif latest_close >= bb_upper * 0.99:
                        signals.append(self._create_signal(
                            symbol, 'SELL', 'BB', (latest_close - bb_upper) / latest_close * 100,
                            'Price at upper Bollinger Band',
                            min(1.0, abs(latest_close - bb_upper) / latest_close * 10), latest_close
                        ))
            
            # Volume spike signal
            if avg_volume and avg_volume > 0:
                volume_ratio = latest_volume / avg_volume
                if volume_ratio > 2.0:
                    price_change_pct = (latest_close - prev_close) / prev_close * 100
                    if abs(price_change_pct) > 1:
                        signal_type = 'BUY' if price_change_pct > 0 else 'SELL'
                        signals.append(self._create_signal(
                            symbol, signal_type, 'VOLUME', volume_ratio,
                            f'Volume spike {volume_ratio:.1f}x average',
                            min(1.0, volume_ratio / 5), latest_close
                        ))
            
        except Exception as e:
            logger.debug(f"Error generating signals for {symbol}: {e}")
        
        return signals
    
    def _create_signal(self, symbol: str, signal_type: str, indicator: str, 
                      value: float, reason: str, strength: float, price: float) -> Dict:
        """Create a signal dictionary"""
        return {
            'symbol': symbol,
            'type': signal_type,
            'indicator': indicator,
            'value': round(value, 2) if not np.isnan(value) else 0,
            'reason': reason,
            'strength': round(min(1.0, max(0.0, strength)), 2),
            'price': round(price, 2)
        }
    
    def _save_signals_to_db(self, signals: List[Dict]):
        """Save signals to both scan_results and signals tables"""
        if not signals:
            return
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                for signal in signals:
                    # Save to scan_results
                    cursor.execute('''
                        INSERT INTO scan_results 
                        (symbol, signal_type, signal_strength, indicator, reason, price, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        signal['symbol'],
                        signal['type'],
                        signal['strength'],
                        signal['indicator'],
                        signal['reason'],
                        signal['price'],
                        datetime.now()
                    ))
                    
                    # Save to signals table
                    cursor.execute('''
                        INSERT INTO signals
                        (symbol, signal_type, signal_strength, reason, 
                         generated_date, expiry_date, status, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    ''', (
                        signal['symbol'],
                        signal['type'],
                        signal['strength'],
                        f"{signal['indicator']}: {signal['reason']}",
                        datetime.now(),
                        datetime.now() + timedelta(days=1),  # 1-day expiry
                        'ACTIVE',
                        f'{{"indicator": "{signal["indicator"]}", "value": {signal["value"]}, "price": {signal["price"]}}}'
                    ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving signals: {e}")
    
    def get_scan_results(self, limit: int = 50) -> List[Dict]:
        """Get recent scan results"""
        results = []
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get today's results
                cursor.execute('''
                    SELECT symbol, signal_type, signal_strength, indicator, 
                           reason, price, timestamp
                    FROM scan_results
                    WHERE DATE(timestamp) = DATE('now')
                    ORDER BY signal_strength DESC, timestamp DESC
                    LIMIT ?
                ''', (limit,))
                
                rows = cursor.fetchall()
                
                if not rows:
                    # Get most recent results
                    cursor.execute('''
                        SELECT symbol, signal_type, signal_strength, indicator, 
                               reason, price, timestamp
                        FROM scan_results
                        ORDER BY timestamp DESC
                        LIMIT ?
                    ''', (limit,))
                    rows = cursor.fetchall()
                
                for row in rows:
                    results.append({
                        'symbol': row[0],
                        'type': row[1],
                        'strength': row[2],
                        'indicator': row[3],
                        'reason': row[4],
                        'price': row[5],
                        'timestamp': row[6]
                    })
                
        except Exception as e:
            logger.error(f"Error getting scan results: {e}")
        
        return results
    
    def display_results(self, signals: List[Dict]):
        """Display scan results"""
        if not signals:
            print("\n📊 No trading signals found.")
            print("This could mean:")
            print("  • The market is in a neutral state")
            print("  • No stocks meet the signal criteria")
            print("  • Try importing more stock data")
            return
        
        # Sort by strength
        signals = sorted(signals, key=lambda x: x.get('strength', 0), reverse=True)
        
        # Separate signals
        buy_signals = [s for s in signals if s['type'] == 'BUY']
        sell_signals = [s for s in signals if s['type'] == 'SELL']
        
        # Display BUY signals
        if buy_signals:
            print("\n" + "="*100)
            print(" 📈 BUY SIGNALS")
            print("="*100)
            print(f"{'Symbol':<10} {'Indicator':<10} {'Strength':<10} {'Price':<12} {'Reason':<45}")
            print("-"*100)
            
            for signal in buy_signals[:15]:
                strength_bar = '█' * int(signal.get('strength', 0) * 10)
                print(f"{signal['symbol']:<10} {signal.get('indicator', 'N/A'):<10} "
                      f"{signal.get('strength', 0):<10.2f} ${signal.get('price', 0):<11.2f} "
                      f"{signal.get('reason', 'N/A')[:45]:<45}")
        
        # Display SELL signals
        if sell_signals:
            print("\n" + "="*100)
            print(" 📉 SELL SIGNALS")
            print("="*100)
            print(f"{'Symbol':<10} {'Indicator':<10} {'Strength':<10} {'Price':<12} {'Reason':<45}")
            print("-"*100)
            
            for signal in sell_signals[:15]:
                print(f"{signal['symbol']:<10} {signal.get('indicator', 'N/A'):<10} "
                      f"{signal.get('strength', 0):<10.2f} ${signal.get('price', 0):<11.2f} "
                      f"{signal.get('reason', 'N/A')[:45]:<45}")
        
        print("\n" + "="*100)
        print(f" 📊 Summary: {len(buy_signals)} BUY signals | {len(sell_signals)} SELL signals | "
              f"Total: {len(signals)} signals")
        print("="*100)
        
        # Show top opportunities
        if signals:
            print("\n🎯 TOP 5 OPPORTUNITIES (by strength):")
            for i, signal in enumerate(signals[:5], 1):
                emoji = "📈" if signal['type'] == 'BUY' else "📉"
                print(f"  {i}. {emoji} {signal['symbol']}: {signal['type']} "
                      f"(Strength: {signal['strength']:.2f}) - {signal['reason']}")

    def scan_with_optimized_config(self, portfolio_name='default'):
        """Enhanced scan using optimized configuration from database"""
        logger.info(f"Starting optimized scan for portfolio: {portfolio_name}")
        
        # Load optimized configuration
        config = self.load_optimized_configuration()
        
        if config['is_optimized']:
            logger.info(f"Using OPTIMIZED config - Score: {config.get('score', 0):.2f}")
        else:
            logger.info("Using DEFAULT configuration")
        
        # Get stocks to scan
        stocks = self._get_scannable_stocks()
        
        if not stocks:
            logger.warning("No stocks found to scan")
            return []
        
        all_signals = []
        
        for i, symbol in enumerate(stocks, 1):
            if i % 10 == 0:
                logger.info(f"Progress: {i}/{len(stocks)}")
            
            try:
                # Analyze with optimized parameters
                signals = self._analyze_stock_with_config(symbol, config)
                
                for signal in signals:
                    # Calculate weighted score
                    score = self._calculate_weighted_score(signal, config)
                    signal['weighted_score'] = score
                    
                    # Filter by minimum score
                    if score >= config.get('min_entry_score', 70):
                        # Apply optimized stop loss and take profit
                        if signal['type'] == 'BUY':
                            signal['stop_loss'] = round(signal['price'] * (1 - config.get('stop_loss_pct', 0.05)), 2)
                            signal['take_profit'] = round(signal['price'] * (1 + config.get('profit_target_pct', 0.10)), 2)
                        
                        all_signals.append(signal)
                        
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
        
        # Sort by weighted score
        all_signals.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
        
        logger.info(f"Scan complete: {len(all_signals)} signals found")
        return all_signals

    def load_optimized_configuration(self):
        """Load optimized configuration from trading_configs table"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='trading_configs'
                """)
                
                if not cursor.fetchone():
                    logger.warning("trading_configs table not found, using defaults")
                    return self._get_default_config()
                
                # Get active configuration
                cursor.execute("""
                    SELECT parameters, performance_score, total_return, sharpe_ratio, created_at
                    FROM trading_configs
                    WHERE is_active = 1
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                
                if result:
                    import json
                    config = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                    config['is_optimized'] = True
                    config['score'] = result[1] if result[1] else 0
                    config['return'] = result[2] if result[2] else 0
                    config['sharpe'] = result[3] if result[3] else 0
                    
                    logger.info(f"Loaded optimized config - Score: {config['score']:.2f}, Return: {config['return']:.2%}")
                    return config
                else:
                    logger.info("No active configuration, using defaults")
                    return self._get_default_config()
                    
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()

    def _get_default_config(self):
        """Get default configuration"""
        return {
            'is_optimized': False,
            'min_entry_score': 70.0,
            'min_confidence': 0.65,
            'max_positions': 10,
            'max_position_pct': 0.10,
            'stop_loss_pct': 0.05,
            'profit_target_pct': 0.10,
            'trailing_stop_pct': 0.03,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_threshold': 0.0,
            'volume_spike_threshold': 2.0,
            'bb_period': 20,
            'bb_std': 2,
            'technical_weight': 0.40,
            'momentum_weight': 0.25,
            'volume_weight': 0.15,
            'sentiment_weight': 0.10,
            'fundamental_weight': 0.10
        }

    def _analyze_stock_with_config(self, symbol, config):
        """Analyze stock using optimized configuration parameters"""
        signals = []
        
        try:
            # Get price data
            df = self._get_price_data(symbol)
            
            if df is None or len(df) < 50:
                return signals
            
            # Calculate indicators (using your existing function)
            indicators = self.indicators.calculate_all(df)
            
            latest = df.iloc[-1]
            
            # RSI with OPTIMIZED thresholds
            rsi_oversold = config.get('rsi_oversold', 30)
            rsi_overbought = config.get('rsi_overbought', 70)
            
            if 'rsi' in indicators and not pd.isna(indicators['rsi'].iloc[-1]):
                rsi_value = indicators['rsi'].iloc[-1]
                
                if rsi_value < rsi_oversold:
                    signals.append({
                        'symbol': symbol,
                        'type': 'BUY',
                        'indicator': 'RSI',
                        'value': round(rsi_value, 2),
                        'reason': f'RSI oversold at {rsi_value:.1f} (threshold: {rsi_oversold})',
                        'strength': round((rsi_oversold - rsi_value) / rsi_oversold, 2),
                        'price': round(latest['close'], 2),
                        'indicator_type': 'technical'
                    })
                elif rsi_value > rsi_overbought:
                    signals.append({
                        'symbol': symbol,
                        'type': 'SELL',
                        'indicator': 'RSI',
                        'value': round(rsi_value, 2),
                        'reason': f'RSI overbought at {rsi_value:.1f} (threshold: {rsi_overbought})',
                        'strength': round((rsi_value - rsi_overbought) / (100 - rsi_overbought), 2),
                        'price': round(latest['close'], 2),
                        'indicator_type': 'technical'
                    })
            
            # MACD with optimized threshold
            macd_threshold = config.get('macd_threshold', 0.0)
            
            if all(col in indicators.columns for col in ['macd', 'macd_signal']):
                macd = indicators['macd'].iloc[-1]
                macd_signal = indicators['macd_signal'].iloc[-1]
                
                if not pd.isna(macd) and not pd.isna(macd_signal):
                    # Check for crossover
                    if len(indicators) > 1:
                        prev_macd = indicators['macd'].iloc[-2]
                        prev_signal = indicators['macd_signal'].iloc[-2]
                        
                        if prev_macd <= prev_signal and macd > macd_signal and macd > macd_threshold:
                            signals.append({
                                'symbol': symbol,
                                'type': 'BUY',
                                'indicator': 'MACD',
                                'value': round(macd - macd_signal, 4),
                                'reason': f'MACD bullish crossover above {macd_threshold}',
                                'strength': round(min(1.0, abs(macd - macd_signal) * 50), 2),
                                'price': round(latest['close'], 2),
                                'indicator_type': 'momentum'
                            })
            
            # Volume spike with optimized threshold
            volume_threshold = config.get('volume_spike_threshold', 2.0)
            
            if 'volume' in df.columns:
                avg_volume = df['volume'].rolling(20).mean().iloc[-1]
                current_volume = df['volume'].iloc[-1]
                
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    
                    if volume_ratio > volume_threshold:
                        price_change = (latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
                        
                        if abs(price_change) > 2:
                            signals.append({
                                'symbol': symbol,
                                'type': 'BUY' if price_change > 0 else 'SELL',
                                'indicator': 'VOLUME',
                                'value': round(volume_ratio, 2),
                                'reason': f'Volume spike {volume_ratio:.1f}x (threshold: {volume_threshold}x)',
                                'strength': round(min(1.0, volume_ratio / (volume_threshold * 2)), 2),
                                'price': round(latest['close'], 2),
                                'indicator_type': 'volume'
                            })
            
            # Bollinger Bands with optimized parameters
            bb_period = int(config.get('bb_period', 20))
            bb_std = config.get('bb_std', 2)
            
            if all(col in indicators.columns for col in ['bb_upper', 'bb_lower']):
                bb_upper = indicators['bb_upper'].iloc[-1]
                bb_lower = indicators['bb_lower'].iloc[-1]
                
                if not pd.isna(bb_upper) and not pd.isna(bb_lower):
                    if latest['close'] <= bb_lower:
                        signals.append({
                            'symbol': symbol,
                            'type': 'BUY',
                            'indicator': 'BB',
                            'value': round((bb_lower - latest['close']) / latest['close'] * 100, 2),
                            'reason': f'Price at lower BB (period: {bb_period}, std: {bb_std})',
                            'strength': round(min(1.0, (bb_lower - latest['close']) / latest['close'] * 10), 2),
                            'price': round(latest['close'], 2),
                            'indicator_type': 'technical'
                        })
                    elif latest['close'] >= bb_upper:
                        signals.append({
                            'symbol': symbol,
                            'type': 'SELL',
                            'indicator': 'BB',
                            'value': round((latest['close'] - bb_upper) / latest['close'] * 100, 2),
                            'reason': f'Price at upper BB (period: {bb_period}, std: {bb_std})',
                            'strength': round(min(1.0, (latest['close'] - bb_upper) / latest['close'] * 10), 2),
                            'price': round(latest['close'], 2),
                            'indicator_type': 'technical'
                        })
                        
        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
        
        return signals

    def _calculate_weighted_score(self, signal, config):
        """Calculate weighted score using optimized weights"""
        
        # Map indicator types to weights
        weight_map = {
            'technical': config.get('technical_weight', 0.4),
            'momentum': config.get('momentum_weight', 0.25),
            'volume': config.get('volume_weight', 0.15),
            'sentiment': config.get('sentiment_weight', 0.1),
            'fundamental': config.get('fundamental_weight', 0.1)
        }
        
        indicator_type = signal.get('indicator_type', 'technical')
        weight = weight_map.get(indicator_type, 0.2)
        
        # Get signal strength
        strength = signal.get('strength', 0.5)
        
        # Calculate weighted score (0-100 scale)
        weighted_score = strength * weight * 100
        
        # Add confidence boost if available
        if 'confidence' in signal:
            confidence = signal['confidence']
            if confidence >= config.get('min_confidence', 0.65):
                weighted_score *= (1 + (confidence - 0.5))
        
        return round(weighted_score, 2)

    def load_optimized_configuration(self):
        """Load optimized configuration from database"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get active optimized configuration
                cursor.execute("""
                    SELECT parameters, performance_score, total_return, sharpe_ratio
                    FROM trading_configs
                    WHERE is_active = 1
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                
                if result:
                    import json
                    params = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                    
                    logger.info(f"Loaded optimized config - Score: {result[1]:.2f}, Return: {result[2]:.2%}")
                    return params, True
                else:
                    logger.info("No optimized config found, using defaults")
                    return self.get_default_config(), False
                    
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.get_default_config(), False

    def get_default_config(self):
        """Get default configuration"""
        return {
            'min_entry_score': 70.0,
            'min_confidence': 0.65,
            'max_positions': 10,
            'max_position_pct': 0.10,
            'stop_loss_pct': 0.05,
            'profit_target_pct': 0.10,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'volume_spike_threshold': 2.0,
            'bb_period': 20,
            'bb_std': 2,
            'technical_weight': 0.40,
            'momentum_weight': 0.25,
            'volume_weight': 0.15,
            'sentiment_weight': 0.10,
            'fundamental_weight': 0.10
        }

    def scan_with_config(self, portfolio_name='default', scan_date=None):
        """Scan using optimized configuration"""
        
        # Load configuration
        config, is_optimized = self.load_optimized_configuration()
        
        if is_optimized:
            logger.info("Using OPTIMIZED configuration")
            logger.info(f"Min Score: {config['min_entry_score']}, Stop Loss: {config['stop_loss_pct']:.1%}")
        else:
            logger.info("Using DEFAULT configuration")
        
        # Get stocks to scan
        stocks = self._get_scannable_stocks()
        
        all_signals = []
        
        for symbol in stocks:
            try:
                # Analyze with config
                signals = self.analyze_stock_optimized(symbol, scan_date, config)
                
                for signal in signals:
                    # Calculate weighted score
                    score = self.calculate_weighted_score(signal, config)
                    signal['weighted_score'] = score
                    
                    # Filter by minimum score
                    if score >= config['min_entry_score']:
                        # Apply optimized stop/profit levels
                        if signal['type'] == 'BUY':
                            signal['stop_loss'] = signal['price'] * (1 - config['stop_loss_pct'])
                            signal['take_profit'] = signal['price'] * (1 + config['profit_target_pct'])
                        
                        all_signals.append(signal)
                        
            except Exception as e:
                logger.debug(f"Error scanning {symbol}: {e}")
        
        # Sort by score
        all_signals.sort(key=lambda x: x.get('weighted_score', 0), reverse=True)
        
        return all_signals

    def analyze_stock_optimized(self, symbol, scan_date, config):
        """Analyze stock using optimized parameters"""
        signals = []
        
        try:
            # Get price data
            df = self._get_price_data(symbol)
            
            if df is None or len(df) < 50:
                return signals
            
            # Calculate indicators
            indicators = self._calculate_indicators(df)
            
            # Use OPTIMIZED thresholds
            rsi_oversold = config.get('rsi_oversold', 30)
            rsi_overbought = config.get('rsi_overbought', 70)
            volume_threshold = config.get('volume_spike_threshold', 2.0)
            
            # RSI signals with optimized thresholds
            if 'rsi' in indicators:
                rsi_value = indicators['rsi']
                if rsi_value < rsi_oversold:
                    signals.append({
                        'symbol': symbol,
                        'type': 'BUY',
                        'indicator': 'RSI',
                        'value': rsi_value,
                        'reason': f'RSI oversold at {rsi_value:.1f} (optimized: {rsi_oversold})',
                        'strength': (rsi_oversold - rsi_value) / rsi_oversold,
                        'price': df['close'].iloc[-1],
                        'indicator_type': 'technical'
                    })
                elif rsi_value > rsi_overbought:
                    signals.append({
                        'symbol': symbol,
                        'type': 'SELL',
                        'indicator': 'RSI',
                        'value': rsi_value,
                        'reason': f'RSI overbought at {rsi_value:.1f} (optimized: {rsi_overbought})',
                        'strength': (rsi_value - rsi_overbought) / (100 - rsi_overbought),
                        'price': df['close'].iloc[-1],
                        'indicator_type': 'technical'
                    })
            
            # Volume signals with optimized threshold
            if 'volume_ratio' in indicators:
                if indicators['volume_ratio'] > volume_threshold:
                    signals.append({
                        'symbol': symbol,
                        'type': 'BUY',
                        'indicator': 'VOLUME',
                        'value': indicators['volume_ratio'],
                        'reason': f'Volume spike {indicators["volume_ratio"]:.1f}x (optimized: {volume_threshold}x)',
                        'strength': min(1.0, indicators['volume_ratio'] / (volume_threshold * 2)),
                        'price': df['close'].iloc[-1],
                        'indicator_type': 'volume'
                    })
            
            # Add more indicators as needed...
            
        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
        
        return signals

    def calculate_weighted_score(self, signal, config):
        """Calculate weighted score using optimized weights"""
        
        # Get weight for indicator type
        weight_map = {
            'technical': config.get('technical_weight', 0.4),
            'momentum': config.get('momentum_weight', 0.25),
            'volume': config.get('volume_weight', 0.15),
            'sentiment': config.get('sentiment_weight', 0.1),
            'fundamental': config.get('fundamental_weight', 0.1)
        }
        
        indicator_type = signal.get('indicator_type', 'technical')
        weight = weight_map.get(indicator_type, 0.2)
        
        # Calculate score
        strength = signal.get('strength', 0.5)
        weighted_score = strength * weight * 100
        
        return round(weighted_score, 2)

    def _calculate_indicators(self, df):
        """Calculate technical indicators"""
        indicators = {}
        
        try:
            close = df['close'].values
            volume = df['volume'].values
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(close)
            
            # Volume ratio
            avg_volume = np.mean(volume[-20:]) if len(volume) >= 20 else np.mean(volume)
            current_volume = volume[-1] if len(volume) > 0 else 0
            indicators['volume_ratio'] = current_volume / avg_volume if avg_volume > 0 else 0
            
            # Add more indicators...
            
        except Exception as e:
            logger.debug(f"Error calculating indicators: {e}")
        
        return indicators

    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50  # Neutral
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi