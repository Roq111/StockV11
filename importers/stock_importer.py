"""
Dynamic Stock Data Importer
Fetches current lists from web and maintains updated database
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Set
import logging
import requests
from bs4 import BeautifulSoup
import time
import json

logger = logging.getLogger(__name__)

class StockImporter:
    """Import and maintain current stock lists dynamically"""
    
    def __init__(self, db):
        self.db = db
        self.start_date = "2022-01-01"
        
    def update_stock_universe(self) -> Dict[str, List[str]]:
        """Update the master list of all stocks in database"""
        print("Updating stock universe...")
        
        results = {
            'sp500': [],
            'nasdaq100': [],
            'russell2000': [],
            'etfs': [],
            'total': 0,
            'new': 0,
            'removed': 0
        }
        
        # Get current stocks from database
        existing_stocks = self._get_existing_stocks()
        
        # Fetch current lists from web
        current_sp500 = self._fetch_sp500_list()
        current_nasdaq = self._fetch_nasdaq100_list()
        current_russell = self._fetch_russell2000_list()
        current_etfs = self._fetch_top_etfs_list()
        
        # Update database with current lists
        all_current = set()
        
        # Process each list
        for symbol in current_sp500:
            self._update_stock_record(symbol, 'S&P 500', 'Large Cap')
            results['sp500'].append(symbol)
            all_current.add(symbol)
        
        for symbol in current_nasdaq:
            self._update_stock_record(symbol, 'NASDAQ 100', 'Tech Heavy')
            results['nasdaq100'].append(symbol)
            all_current.add(symbol)
        
        for symbol in current_russell:
            self._update_stock_record(symbol, 'Russell 2000', 'Small Cap')
            results['russell2000'].append(symbol)
            all_current.add(symbol)
        
        for symbol in current_etfs:
            self._update_stock_record(symbol, 'ETF', 'Fund')
            results['etfs'].append(symbol)
            all_current.add(symbol)
        
        # Find new and removed stocks
        new_stocks = all_current - existing_stocks
        removed_stocks = existing_stocks - all_current
        
        # Mark removed stocks as inactive
        for symbol in removed_stocks:
            self._mark_stock_inactive(symbol)
        
        results['total'] = len(all_current)
        results['new'] = len(new_stocks)
        results['removed'] = len(removed_stocks)
        
        print(f"✓ Updated stock universe: {results['total']} active stocks")
        print(f"  New: {results['new']}, Removed: {results['removed']}")
        
        return results
    
    def _fetch_sp500_list(self) -> List[str]:
        """Fetch current S&P 500 list from multiple sources"""
        symbols = []
        
        # Method 1: Wikipedia via pandas
        try:
            print("  Fetching S&P 500 from Wikipedia...")
            tables = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
            if tables:
                df = tables[0]
                if 'Symbol' in df.columns:
                    symbols = df['Symbol'].tolist()
                    # Convert dots to dashes for Yahoo Finance compatibility
                    symbols = [s.replace('.', '-') for s in symbols]
                    print(f"    ✓ Found {len(symbols)} S&P 500 symbols")
                    return symbols
        except Exception as e:
            print(f"    ✗ Wikipedia failed: {e}")
        
        # Method 2: Alternative source - SLICKCHARTS
        try:
            print("  Fetching S&P 500 from SlickCharts...")
            url = "https://www.slickcharts.com/sp500"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                table = soup.find('table', class_='table')
                
                if table:
                    symbols = []
                    for row in table.find_all('tr')[1:]:  # Skip header
                        cells = row.find_all('td')
                        if len(cells) >= 3:
                            symbol = cells[2].text.strip()  # Symbol is in 3rd column
                            if symbol:
                                symbols.append(symbol.replace('.', '-'))
                    
                    if symbols:
                        print(f"    ✓ Found {len(symbols)} S&P 500 symbols")
                        return symbols
        except Exception as e:
            print(f"    ✗ SlickCharts failed: {e}")
        
        # Method 3: DataHub.io
        try:
            print("  Fetching S&P 500 from DataHub...")
            url = "https://datahub.io/core/s-and-p-500-companies/r/constituents.json"
            response = requests.get(url)
            
            if response.status_code == 200:
                data = response.json()
                symbols = [item['Symbol'].replace('.', '-') for item in data]
                print(f"    ✓ Found {len(symbols)} S&P 500 symbols")
                return symbols
        except Exception as e:
            print(f"    ✗ DataHub failed: {e}")
        
        print("    ⚠ Using cached S&P 500 list from database")
        return self._get_stocks_by_index('S&P 500')
    
    def _fetch_nasdaq100_list(self) -> List[str]:
        """Fetch current NASDAQ 100 list"""
        symbols = []
        
        # Method 1: Wikipedia
        try:
            print("  Fetching NASDAQ 100...")
            tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            
            for table in tables:
                if 'Ticker' in table.columns:
                    symbols = table['Ticker'].tolist()
                    print(f"    ✓ Found {len(symbols)} NASDAQ 100 symbols")
                    return symbols
                elif 'Symbol' in table.columns:
                    symbols = table['Symbol'].tolist()
                    print(f"    ✓ Found {len(symbols)} NASDAQ 100 symbols")
                    return symbols
        except Exception as e:
            print(f"    ✗ NASDAQ 100 fetch failed: {e}")
        
        # Method 2: Use QQQ holdings (NASDAQ 100 ETF)
        try:
            print("  Fetching from QQQ holdings...")
            ticker = yf.Ticker('QQQ')
            info = ticker.info
            
            # Try to get holdings
            if hasattr(ticker, 'holdings'):
                holdings = ticker.holdings
                if holdings is not None:
                    symbols = holdings.index.tolist()[:100]
                    print(f"    ✓ Found {len(symbols)} NASDAQ 100 symbols from QQQ")
                    return symbols
        except:
            pass
        
        print("    ⚠ Using cached NASDAQ 100 list from database")
        return self._get_stocks_by_index('NASDAQ 100')
    
    def _fetch_russell2000_list(self) -> List[str]:
        """Fetch Russell 2000 components (sample)"""
        print("  Fetching Russell 2000 sample...")
        
        # For Russell 2000, we'll use IWM ETF top holdings
        # Full list requires subscription, so we get top holdings
        try:
            # Get popular small caps from various sources
            symbols = []
            
            # Method 1: Get from IWM ETF info
            ticker = yf.Ticker('IWM')
            
            # Method 2: Get most active small caps
            # This would need a proper API or data source
            
            # For now, return empty and rely on database
            print("    ⚠ Russell 2000 requires data subscription")
            return self._get_stocks_by_index('Russell 2000')
            
        except Exception as e:
            print(f"    ✗ Russell 2000 fetch failed: {e}")
            return []
    
    def _fetch_top_etfs_list(self) -> List[str]:
        """Fetch top ETFs by volume and AUM"""
        print("  Fetching top ETFs...")
        
        etfs = []
        
        # Method 1: ETF Database most popular
        try:
            url = "https://etfdb.com/compare/volume/"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Parse table for ETF symbols
                # This would need proper parsing
        except:
            pass
        
        # Method 2: Use known top ETFs
        # These are consistently the most traded ETFs
        top_etfs = [
            # Core Index ETFs
            'SPY', 'IVV', 'VOO', 'QQQ', 'DIA', 'IWM', 'VTI',
            # Sector ETFs  
            'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLB', 'XLRE', 'XLU',
            # International
            'EFA', 'VEA', 'EEM', 'VWO', 'IEMG', 'FXI', 'EWJ',
            # Fixed Income
            'AGG', 'BND', 'TLT', 'IEF', 'HYG', 'JNK', 'LQD',
            # Commodities
            'GLD', 'IAU', 'SLV', 'USO', 'GDX', 'GDXJ',
            # Thematic
            'ARKK', 'ARKQ', 'ARKW', 'ARKG', 'ARKF', 'ICLN',
            # Volatility
            'VXX', 'UVXY', 'TQQQ', 'SQQQ', 'SPXU', 'UPRO'
        ]
        
        print(f"    ✓ Using {len(top_etfs)} top ETFs")
        return top_etfs
    
    def _get_existing_stocks(self) -> Set[str]:
        """Get all existing active stocks from database"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol FROM stocks 
                WHERE active = 1 OR active IS NULL
            """)
            return set(row[0] for row in cursor.fetchall())
    
    def _get_stocks_by_index(self, index_name: str) -> List[str]:
        """Get stocks by index from database"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol FROM stocks 
                WHERE sector = ? AND (active = 1 OR active IS NULL)
            """, (index_name,))
            return [row[0] for row in cursor.fetchall()]
    
    def _update_stock_record(self, symbol: str, index: str, category: str):
        """Update or insert stock record with metadata"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Check if exists
            cursor.execute("SELECT symbol FROM stocks WHERE symbol = ?", (symbol,))
            exists = cursor.fetchone() is not None
            
            if exists:
                # Update existing record
                cursor.execute("""
                    UPDATE stocks 
                    SET sector = ?, industry = ?, active = 1, last_updated = ?
                    WHERE symbol = ?
                """, (index, category, datetime.now(), symbol))
            else:
                # Insert new record
                cursor.execute("""
                    INSERT INTO stocks (symbol, name, sector, industry, active, last_updated)
                    VALUES (?, ?, ?, ?, 1, ?)
                """, (symbol, symbol, index, category, datetime.now()))
            
            conn.commit()
    
    def _mark_stock_inactive(self, symbol: str):
        """Mark stock as inactive (delisted or removed from index)"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE stocks 
                SET active = 0, last_updated = ?
                WHERE symbol = ?
            """, (datetime.now(), symbol))
            conn.commit()
    
    def import_all_active_stocks(self) -> Dict:
        """Import price data for all active stocks"""
        print("\n" + "="*60)
        print("Importing price data for all active stocks")
        print(f"Date range: {self.start_date} to {datetime.now().strftime('%Y-%m-%d')}")
        print("="*60)
        
        # First, update the stock universe
        universe_stats = self.update_stock_universe()
        
        # Get all active stocks from database
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol, sector, name 
                FROM stocks 
                WHERE active = 1 OR active IS NULL
                ORDER BY sector, symbol
            """)
            stocks = cursor.fetchall()
        
        print(f"\nImporting data for {len(stocks)} active stocks")
        print("="*60)
        
        results = {
            'total': len(stocks),
            'success': 0,
            'failed': [],
            'by_index': {}
        }
        
        # Import data for each stock
        for i, (symbol, sector, name) in enumerate(stocks, 1):
            try:
                print(f"[{i}/{len(stocks)}] {symbol} ({sector})...", end='')
                
                success = self._import_stock_data(symbol)
                
                if success:
                    results['success'] += 1
                    
                    # Track by index
                    if sector not in results['by_index']:
                        results['by_index'][sector] = {'success': 0, 'failed': 0}
                    results['by_index'][sector]['success'] += 1
                    
                    print(f" ✓")
                else:
                    results['failed'].append(symbol)
                    
                    if sector not in results['by_index']:
                        results['by_index'][sector] = {'success': 0, 'failed': 0}
                    results['by_index'][sector]['failed'] += 1
                    
                    print(f" ✗")
                
                # Progress update
                if i % 25 == 0:
                    print(f"\nProgress: {i}/{len(stocks)} ({i/len(stocks)*100:.1f}%)")
                    print(f"Success rate: {results['success']}/{i} ({results['success']/i*100:.1f}%)\n")
                
                # Rate limiting
                if i % 5 == 0:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print(f"\n\nImport interrupted by user")
                break
            except Exception as e:
                print(f" ✗ Error: {e}")
                results['failed'].append(symbol)
        
        # Print summary
        self._print_import_summary(results, universe_stats)
        
        return results
    
    def _import_stock_data(self, symbol: str) -> bool:
        """Import historical data for a single stock"""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get historical data from start_date
            hist = ticker.history(start=self.start_date, end=datetime.now().strftime('%Y-%m-%d'))
            
            if hist.empty:
                return False
            
            # Get and update stock info
            try:
                info = ticker.info
                if info:
                    self._update_stock_info(symbol, info)
            except:
                pass  # Info might fail but we still have price data
            
            # Save price data
            self._save_price_data(symbol, hist)
            
            return True
            
        except Exception as e:
            logger.error(f"Error importing {symbol}: {e}")
            return False
    
    def _update_stock_info(self, symbol: str, info: Dict):
        """Update stock information"""
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                UPDATE stocks
                SET name = ?, market_cap = ?, shares_outstanding = ?, 
                    country = ?, last_updated = ?
                WHERE symbol = ?
            """, (
                info.get('longName', symbol),
                info.get('marketCap', 0),
                info.get('sharesOutstanding', 0),
                info.get('country', 'US'),
                datetime.now(),
                symbol
            ))
            
            conn.commit()
    
    def _save_price_data(self, symbol: str, data: pd.DataFrame):
        """Save price data to database"""
        with self.db.get_connection() as conn:
            # Prepare data
            data = data.reset_index()
            data['symbol'] = symbol
            
            # Clean column names
            data.columns = [col.lower().replace(' ', '_') for col in data.columns]
            
            # Ensure date format
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
            elif 'index' in data.columns:
                data.rename(columns={'index': 'date'}, inplace=True)
                data['date'] = pd.to_datetime(data['date']).dt.strftime('%Y-%m-%d')
            
            # Add adjusted_close if missing
            if 'adjusted_close' not in data.columns:
                data['adjusted_close'] = data.get('close', 0)
            
            # Save each row
            for _, row in data.iterrows():
                try:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT OR REPLACE INTO price_data 
                        (symbol, date, open, high, low, close, adjusted_close, volume)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        row['date'],
                        float(row.get('open', 0)),
                        float(row.get('high', 0)),
                        float(row.get('low', 0)),
                        float(row.get('close', 0)),
                        float(row.get('adjusted_close', row.get('close', 0))),
                        int(row.get('volume', 0))
                    ))
                except Exception as e:
                    logger.error(f"Error saving row for {symbol}: {e}")
                    continue
            
            conn.commit()
    
    def _print_import_summary(self, results: Dict, universe_stats: Dict):
        """Print comprehensive import summary"""
        print("\n" + "="*60)
        print("IMPORT SUMMARY")
        print("="*60)
        
        print(f"\nStock Universe Update:")
        print(f"  Total active stocks: {universe_stats['total']}")
        print(f"  New stocks added: {universe_stats['new']}")
        print(f"  Stocks marked inactive: {universe_stats['removed']}")
        
        print(f"\nPrice Data Import:")
        print(f"  Total processed: {results['total']}")
        print(f"  Successfully imported: {results['success']}")
        print(f"  Failed: {len(results['failed'])}")
        print(f"  Success rate: {results['success']/results['total']*100:.1f}%")
        
        print(f"\nBreakdown by Index:")
        for index, stats in results['by_index'].items():
            total = stats['success'] + stats['failed']
            print(f"  {index}: {stats['success']}/{total} ({stats['success']/total*100:.1f}%)")
        
        if results['failed']:
            print(f"\nFailed symbols ({len(results['failed'])}):")
            if len(results['failed']) <= 20:
                print(f"  {', '.join(results['failed'])}")
            else:
                print(f"  {', '.join(results['failed'][:20])} ... and {len(results['failed'])-20} more")
        
        print("="*60)
    
    def update_latest_prices(self) -> Dict:
        """Update latest prices for all active stocks"""
        print("Updating latest prices for all active stocks...")
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT symbol FROM stocks 
                WHERE active = 1 OR active IS NULL
            """)
            symbols = [row[0] for row in cursor.fetchall()]
        
        success = 0
        failed = []
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                hist = ticker.history(period="5d")
                
                if not hist.empty:
                    self._save_price_data(symbol, hist)
                    success += 1
                else:
                    failed.append(symbol)
                    
            except Exception as e:
                failed.append(symbol)
                logger.error(f"Failed to update {symbol}: {e}")
        
        print(f"✓ Updated: {success}/{len(symbols)}")
        if failed:
            print(f"✗ Failed: {len(failed)}")
        
        return {'success': success, 'failed': failed, 'total': len(symbols)}