"""
New Stock Importer
Import data for new stocks not in database
"""

import yfinance as yf
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class NewStockImporter:
    """Import new stocks to the system"""
    
    def __init__(self, db):
        self.db = db
        self.stock_importer = None
    
    def import_stock(self, symbol: str, period: str = "2y") -> bool:
        """Import a new stock"""
        try:
            # Check if already exists
            existing = self.db.get_stock_info(symbol)
            if existing:
                logger.info(f"{symbol} already exists in database")
                return self._update_existing(symbol)
            
            # Import new stock
            from importers.stock_importer import StockImporter
            self.stock_importer = StockImporter(self.db)
            
            success = self.stock_importer.import_stock(symbol, period)
            
            if success:
                logger.info(f"Successfully imported new stock: {symbol}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error importing new stock {symbol}: {e}")
            return False
    
    def _update_existing(self, symbol: str) -> bool:
        """Update existing stock with latest data"""
        try:
            from importers.stock_importer import StockImporter
            importer = StockImporter(self.db)
            return importer.update_latest_data(symbol)
        except Exception as e:
            logger.error(f"Error updating {symbol}: {e}")
            return False
    
    def validate_symbol(self, symbol: str) -> bool:
        """Validate if symbol exists and is tradeable"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Check if valid
            if info and 'symbol' in info:
                return True
            
            return False
            
        except Exception:
            return False