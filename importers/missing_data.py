"""
Missing Data Importer
Find and import missing data for existing stocks
"""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

class MissingDataImporter:
    """Import missing data for stocks"""
    
    def __init__(self, db):
        self.db = db
    
    def find_missing_data(self) -> Dict[str, List[str]]:
        """Find missing data for all stocks"""
        missing = {}
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get all stocks
            cursor.execute("SELECT symbol FROM stocks")
            symbols = [row[0] for row in cursor.fetchall()]
            
            for symbol in symbols:
                # Get date range
                cursor.execute('''
                    SELECT MIN(date), MAX(date) 
                    FROM price_data 
                    WHERE symbol = ?
                ''', (symbol,))
                
                result = cursor.fetchone()
                if result and result[0]:
                    min_date = datetime.strptime(result[0], '%Y-%m-%d')
                    max_date = datetime.strptime(result[1], '%Y-%m-%d')
                    
                    # Find missing dates
                    missing_dates = self._find_missing_dates(
                        cursor, symbol, min_date, max_date
                    )
                    
                    if missing_dates:
                        missing[symbol] = missing_dates
        
        return missing
    
    def _find_missing_dates(self, cursor, symbol: str, 
                           start_date: datetime, end_date: datetime) -> List[str]:
        """Find missing dates for a symbol"""
        # Get existing dates
        cursor.execute('''
            SELECT date FROM price_data 
            WHERE symbol = ? AND date BETWEEN ? AND ?
            ORDER BY date
        ''', (symbol, start_date.strftime('%Y-%m-%d'), 
              end_date.strftime('%Y-%m-%d')))
        
        existing_dates = set(row[0] for row in cursor.fetchall())
        
        # Generate all trading days
        current = start_date
        missing = []
        
        while current <= end_date:
            if current.weekday() < 5:  # Monday to Friday
                date_str = current.strftime('%Y-%m-%d')
                if date_str not in existing_dates:
                    missing.append(date_str)
            current += timedelta(days=1)
        
        return missing
    
    def import_all_missing(self) -> Dict[str, int]:
        """Import all missing data"""
        missing = self.find_missing_data()
        results = {}
        
        for symbol, dates in missing.items():
            try:
                count = self.import_missing_for_symbol(symbol, dates)
                results[symbol] = count
                logger.info(f"Imported {count} missing days for {symbol}")
            except Exception as e:
                logger.error(f"Error importing missing data for {symbol}: {e}")
                results[symbol] = 0
        
        return results
    
    def import_missing_for_symbol(self, symbol: str, dates: List[str]) -> int:
        """Import missing data for specific symbol"""
        if not dates:
            return 0
        
        # Get date range
        min_date = min(dates)
        max_date = max(dates)
        
        # Import data
        ticker = yf.Ticker(symbol)
        hist = ticker.history(start=min_date, end=max_date)
        
        if hist.empty:
            return 0
        
        # Save data
        count = 0
        with self.db.get_connection() as conn:
            for date_str in dates:
                try:
                    date = pd.to_datetime(date_str)
                    if date in hist.index:
                        row = hist.loc[date]
                        cursor = conn.cursor()
                        cursor.execute('''
                            INSERT OR IGNORE INTO price_data 
                            (symbol, date, open, high, low, close, adjusted_close, volume)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        ''', (
                            symbol, date_str,
                            row['Open'], row['High'], row['Low'], row['Close'],
                            row.get('Close', row['Close']), int(row['Volume'])
                        ))
                        count += 1
                except Exception as e:
                    logger.error(f"Error importing {symbol} for {date_str}: {e}")
            
            conn.commit()
        
        return count