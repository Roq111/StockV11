#!/usr/bin/env python3
"""
Trading System Main Entry Point
מערכת מסחר אלגוריתמית מתקדמת
"""

import json
import sys
import os
from datetime import datetime, timedelta
import time
from colorama import init, Fore, Style
import logging
import numpy as np
import pandas as pd

from importers.stock_importer import StockImporter

# Initialize colorama for Windows
init()

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import Config, TradingRules
from database import DatabaseManager
from importers.missing_data import MissingDataImporter
from importers.new_stock import NewStockImporter
from portfolio.manager import PortfolioManager
from portfolio.performance import PerformanceAnalyzer
from scanner.daily_scanner import DailyScanner
from backtest.engine import BacktestEngine
from optimizer.genetic_algorithm import GeneticOptimizer
from brokers.interactive_brokers import IBConnector
from utils.logger import setup_logger
from utils.risk_management import RiskManager
from optimizer.grid_search import GridSearchOptimizer  
from optimizer.bayesian import BayesianOptimizer

# Setup logging
logger = setup_logger('TradingSystem')

class TradingSystem:
    """Main Trading System Class"""
    
    def __init__(self):
        """Initialize the trading system"""
        self.config = Config()
        self.db = DatabaseManager()
        self.portfolio_manager = PortfolioManager(self.db)
        self.scanner = DailyScanner(self.db)
        self.backtest_engine = BacktestEngine(self.db)
        self.optimizer = GeneticOptimizer(self.db)
        self.ib_connector = None
        
        # Initialize database
        self.db.initialize()
        logger.info("Trading System Initialized")
    
    def display_menu(self):
        """Display main menu"""
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}📈 Trading System - Main Menu 📊{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"\n{Fore.YELLOW}1. 📥 Import Data{Style.RESET_ALL}")
        print(f"   1.1 Import All Stocks")
        print(f"   1.2 Import Missing Data")
        print(f"   1.3 Import New Stock")
        print(f"\n{Fore.YELLOW}2. 💼 Portfolio Management{Style.RESET_ALL}")
        print(f"   2.1 View Current Status")
        print(f"   2.2 Add/Remove Position")
        print(f"   2.3 View Activity History")
        print(f"   2.4 Performance Analysis")
        print(f"\n{Fore.YELLOW}3. 🔍 Daily Scanner{Style.RESET_ALL}")
        print(f"   3.1 Run Scan (Info Only)")
        print(f"   3.2 Run Scan & Execute")
        print(f"   3.3 Configure Rules")
        print(f"\n{Fore.YELLOW}4. 📊 Backtest{Style.RESET_ALL}")
        print(f"   4.1 Run Single Backtest")
        print(f"   4.2 Compare Strategies")
        print(f"   4.3 View Results")
        print(f"\n{Fore.YELLOW}5. 🎯 Optimizer{Style.RESET_ALL}")
        print(f"   5.1 Genetic Algorithm")
        print(f"   5.2 Grid Search")
        print(f"   5.3 Bayesian Optimization")
        print(f"   5.4 View Best Parameters")
        print(f"\n{Fore.YELLOW}6. 🔌 Broker Connection{Style.RESET_ALL}")
        print(f"   6.1 Connect to IB")
        print(f"   6.2 Execute Orders")
        print(f"   6.3 Sync Positions")
        print(f"\n{Fore.YELLOW}7. ⚙️  Settings{Style.RESET_ALL}")
        print(f"   7.1 Configure Rules")
        print(f"   7.2 Risk Management")
        print(f"   7.3 Database Maintenance")
        print(f"\n{Fore.RED}0. Exit{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    def import_menu(self):
        """Handle import operations"""
        while True:
            print(f"\n{Fore.CYAN}Import Menu:{Style.RESET_ALL}")
            print("1. Import All Stocks (Full Import)")
            print("2. Update Stock Universe (Refresh Lists)")
            print("3. Import Missing Data")
            print("4. Import New Stock")
            print("5. Update Today's Data")
            print("0. Back to Main Menu")
            
            choice = input(f"\n{Fore.GREEN}Enter choice: {Style.RESET_ALL}")
            
            if choice == '1':
                self.import_all_stocks()
            elif choice == '2':
                self.update_stock_universe()
            elif choice == '3':
                self.import_missing_data()
            elif choice == '4':
                self.import_new_stock()
            elif choice == '5':
                self.update_today_data()
            elif choice == '0':
                break
        
    def import_all_stocks(self):
        """Import all stocks - dynamically fetched from web"""
        try:
            # Try to use dynamic importer
            try:
                from importers.stock_importer import StockImporter
                
                importer = StockImporter(self.db)
                print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
                print(f"{Fore.GREEN}Dynamic Stock Import System{Style.RESET_ALL}")
                print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
                print(f"\n{Fore.YELLOW}This will:{Style.RESET_ALL}")
                print("1. Fetch current S&P 500, NASDAQ 100, Russell 2000, and ETF lists")
                print("2. Update the stock universe in database")
                print("3. Import historical data from 2022-01-01")
                print("4. Mark delisted stocks as inactive")
                
                confirm = input(f"\n{Fore.GREEN}Continue? (y/n): {Style.RESET_ALL}")
                if confirm.lower() != 'y':
                    print("Import cancelled")
                    return
                
                # Run the import
                results = importer.import_all_active_stocks()
                
                print(f"\n{Fore.GREEN}Import completed successfully!{Style.RESET_ALL}")
                
                # Save import log
                self._log_import_results(results)
                
                return
                
            except ImportError:
                print(f"{Fore.YELLOW}Dynamic importer not found, using standard importer...{Style.RESET_ALL}")
            
            # Fallback to standard importer with database stocks
            importer = StockImporter(self.db)
            print(f"{Fore.YELLOW}Importing stocks from database...{Style.RESET_ALL}")
            
            # Get all active stocks from database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # First check if we have stocks in database
                cursor.execute("SELECT COUNT(*) FROM stocks")
                count = cursor.fetchone()[0]
                
                if count == 0:
                    print(f"{Fore.RED}No stocks found in database!{Style.RESET_ALL}")
                    print(f"\n{Fore.YELLOW}Initializing with default stock list...{Style.RESET_ALL}")
                    
                    # Initialize with a default list if database is empty
                    self._initialize_default_stocks()
                
                # Now get all stocks
                cursor.execute("""
                    SELECT symbol, sector 
                    FROM stocks 
                    WHERE active = 1 OR active IS NULL
                    ORDER BY sector, symbol
                """)
                stocks = cursor.fetchall()
            
            if not stocks:
                print(f"{Fore.RED}No active stocks found!{Style.RESET_ALL}")
                return
            
            print(f"Found {len(stocks)} stocks in database")
            print(f"Data range: 2022-01-01 to {datetime.now().strftime('%Y-%m-%d')}")
            print("="*60)
            
            # Group by sector for better progress tracking
            stocks_by_sector = {}
            for symbol, sector in stocks:
                if sector not in stocks_by_sector:
                    stocks_by_sector[sector] = []
                stocks_by_sector[sector].append(symbol)
            
            # Import with progress tracking
            total_stocks = len(stocks)
            success_count = 0
            failed_symbols = []
            current = 0
            
            for sector, symbols in stocks_by_sector.items():
                print(f"\n{Fore.CYAN}Importing {sector} ({len(symbols)} stocks){Style.RESET_ALL}")
                
                for symbol in symbols:
                    current += 1
                    try:
                        print(f"[{current}/{total_stocks}] {symbol}...", end='')
                        
                        # Set start date
                        if hasattr(importer, 'start_date'):
                            importer.start_date = "2022-01-01"
                        
                        success = importer.import_stock(symbol)
                        
                        if success:
                            success_count += 1
                            print(f" {Fore.GREEN}✓{Style.RESET_ALL}")
                        else:
                            failed_symbols.append(symbol)
                            print(f" {Fore.RED}✗{Style.RESET_ALL}")
                        
                        # Progress update
                        if current % 20 == 0:
                            progress = (current / total_stocks) * 100
                            print(f"\nOverall progress: {progress:.1f}% ({current}/{total_stocks})")
                        
                        # Rate limiting
                        if current % 5 == 0:
                            time.sleep(1)
                            
                    except KeyboardInterrupt:
                        print(f"\n{Fore.YELLOW}Import interrupted by user{Style.RESET_ALL}")
                        break
                    except Exception as e:
                        failed_symbols.append(symbol)
                        print(f" {Fore.RED}✗ Error: {str(e)}{Style.RESET_ALL}")
            
            # Print summary
            self._print_import_summary(total_stocks, success_count, failed_symbols)
            
        except Exception as e:
            print(f"{Fore.RED}Error during import: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()

    def _initialize_default_stocks(self):
        """Initialize database with default stock lists"""
        print("Initializing stock database...")
        
        # Default lists - will be updated when dynamic importer runs
        default_stocks = {
            'S&P 500': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
                'JPM', 'V', 'JNJ', 'WMT', 'PG', 'MA', 'UNH', 'DIS', 'HD', 'PYPL',
                'BAC', 'VZ', 'ADBE', 'NFLX', 'KO', 'PFE', 'CMCSA', 'PEP', 'TMO',
                'CSCO', 'ABT', 'CVX', 'NKE', 'WFC', 'XOM', 'ABBV', 'CRM', 'ACN'
            ],
            'NASDAQ 100': [
                'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'AVGO',
                'PEP', 'COST', 'ADBE', 'CSCO', 'TMUS', 'TXN', 'NFLX', 'QCOM',
                'INTC', 'INTU', 'AMD', 'AMGN', 'HON', 'AMAT', 'PYPL', 'SBUX'
            ],
            'ETF': [
                'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'EFA', 'EEM', 'VEA',
                'AGG', 'GLD', 'TLT', 'XLF', 'XLK', 'XLE', 'XLV', 'ARKK', 'VNQ',
                'HYG', 'SLV', 'USO', 'TQQQ', 'SQQQ', 'VXX', 'GDX'
            ]
        }
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            for index, symbols in default_stocks.items():
                for symbol in symbols:
                    try:
                        cursor.execute("""
                            INSERT OR IGNORE INTO stocks 
                            (symbol, name, sector, active, last_updated)
                            VALUES (?, ?, ?, 1, ?)
                        """, (symbol, symbol, index, datetime.now()))
                    except Exception as e:
                        logger.error(f"Error inserting {symbol}: {e}")
            
            conn.commit()
            
        print(f"✓ Initialized with {sum(len(s) for s in default_stocks.values())} default stocks")

    def update_today_data(self):
        """Update latest data for all active stocks"""
        try:
            print(f"{Fore.YELLOW}Updating latest data for all active stocks...{Style.RESET_ALL}")
            
            # Get active stocks from database
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT symbol FROM stocks 
                    WHERE active = 1 OR active IS NULL
                    ORDER BY symbol
                """)
                symbols = [row[0] for row in cursor.fetchall()]
            
            if not symbols:
                print(f"{Fore.YELLOW}No active stocks found. Please import stocks first.{Style.RESET_ALL}")
                return
            
            print(f"Updating {len(symbols)} stocks...")
            
            # Try dynamic importer first
            try:
                from importers.stock_importer import DynamicStockImporter
                importer = DynamicStockImporter(self.db)
                results = importer.update_latest_prices()
                
                print(f"\n{Fore.GREEN}Update Complete:{Style.RESET_ALL}")
                print(f"  Successfully updated: {results['success']}/{results['total']}")
                if results['failed']:
                    print(f"  Failed: {len(results['failed'])}")
                
                return
                
            except ImportError:
                pass
            
            # Fallback to standard importer
            importer = StockImporter(self.db)
            success_count = 0
            failed_symbols = []
            
            for i, symbol in enumerate(symbols, 1):
                try:
                    success = importer.update_latest_data(symbol)
                    
                    if success:
                        success_count += 1
                    else:
                        failed_symbols.append(symbol)
                    
                    # Progress
                    if i % 20 == 0:
                        print(f"Progress: {i}/{len(symbols)} ({(i/len(symbols))*100:.1f}%)")
                    
                    # Rate limiting
                    if i % 5 == 0:
                        time.sleep(0.5)
                        
                except Exception as e:
                    failed_symbols.append(symbol)
                    logger.error(f"Failed to update {symbol}: {e}")
            
            print(f"\n{Fore.GREEN}Update Summary:{Style.RESET_ALL}")
            print(f"  Updated: {success_count}/{len(symbols)}")
            if failed_symbols:
                print(f"  Failed: {', '.join(failed_symbols[:10])}")
                if len(failed_symbols) > 10:
                    print(f"    ... and {len(failed_symbols)-10} more")
                
        except Exception as e:
            print(f"{Fore.RED}Error updating data: {str(e)}{Style.RESET_ALL}")

    def _print_import_summary(self, total, success, failed):
        """Print import summary"""
        print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Import Summary:{Style.RESET_ALL}")
        print(f"{'='*60}")
        print(f"Total symbols processed: {total}")
        print(f"Successfully imported: {success}")
        print(f"Failed: {len(failed)}")
        print(f"Success rate: {(success/total)*100:.1f}%" if total > 0 else "N/A")
        
        if failed:
            print(f"\nFailed symbols ({len(failed)}):")
            if len(failed) <= 20:
                print(f"  {', '.join(failed)}")
            else:
                print(f"  {', '.join(failed[:20])} ... and {len(failed)-20} more")
        
        print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")

    def _log_import_results(self, results):
        """Log import results to database"""
        try:
            # Create import_logs table if not exists
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS import_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TIMESTAMP,
                        total_symbols INTEGER,
                        success_count INTEGER,
                        failed_count INTEGER,
                        failed_symbols TEXT,
                        metadata TEXT
                    )
                """)
                
                # Insert log
                cursor.execute("""
                    INSERT INTO import_logs 
                    (date, total_symbols, success_count, failed_count, failed_symbols, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    datetime.now(),
                    results.get('total', 0),
                    results.get('success', 0),
                    len(results.get('failed', [])),
                    ','.join(results.get('failed', [])[:100]),
                    json.dumps(results.get('by_index', {}))
                ))
                
                conn.commit()
        except Exception as e:
            logger.error(f"Error logging import results: {e}")

    def update_stock_universe(self):
        """Update the list of stocks from current market indices"""
        try:
            from importers.stock_importer import DynamicStockImporter
            
            print(f"{Fore.CYAN}Updating Stock Universe{Style.RESET_ALL}")
            print("This will fetch current lists from:")
            print("  • S&P 500 (Wikipedia)")
            print("  • NASDAQ 100 (Wikipedia)")  
            print("  • Russell 2000 (Sample)")
            print("  • Top ETFs")
            
            importer = DynamicStockImporter(self.db)
            results = importer.update_stock_universe()
            
            print(f"\n{Fore.GREEN}Universe Update Complete:{Style.RESET_ALL}")
            print(f"  Total active stocks: {results['total']}")
            print(f"  New additions: {results['new']}")
            print(f"  Marked inactive: {results['removed']}")
            print(f"\nBreakdown:")
            print(f"  S&P 500: {len(results['sp500'])}")
            print(f"  NASDAQ 100: {len(results['nasdaq100'])}")
            print(f"  Russell 2000: {len(results['russell2000'])}")
            print(f"  ETFs: {len(results['etfs'])}")
            
        except ImportError:
            print(f"{Fore.RED}Dynamic importer not available{Style.RESET_ALL}")
        except Exception as e:
            print(f"{Fore.RED}Error updating universe: {str(e)}{Style.RESET_ALL}")
            
    
    def import_missing_data(self):
        """Import missing data for existing stocks"""
        try:
            importer = MissingDataImporter(self.db)
            print(f"{Fore.YELLOW}Checking for missing data...{Style.RESET_ALL}")
            
            missing = importer.find_missing_data()
            if not missing:
                print(f"{Fore.GREEN}No missing data found!{Style.RESET_ALL}")
                return
            
            print(f"Found missing data for {len(missing)} stocks")
            for symbol, dates in missing.items():
                print(f"  {symbol}: {len(dates)} missing days")
            
            if input("Import missing data? (y/n): ").lower() == 'y':
                importer.import_all_missing()
                print(f"{Fore.GREEN}Missing data imported successfully{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def import_new_stock(self):
        """Import a new stock"""
        try:
            symbol = input("Enter stock symbol: ").upper()
            
            importer = NewStockImporter(self.db)
            print(f"{Fore.YELLOW}Importing {symbol}...{Style.RESET_ALL}")
            
            success = importer.import_stock(symbol)
            if success:
                print(f"{Fore.GREEN}Successfully imported {symbol}{Style.RESET_ALL}")
                
                # Show basic info
                info = self.db.get_stock_info(symbol)
                if info:
                    print(f"\nStock Info:")
                    print(f"  Name: {info.get('name', 'N/A')}")
                    print(f"  Sector: {info.get('sector', 'N/A')}")
                    print(f"  Market Cap: ${info.get('market_cap', 0):,.0f}")
            else:
                print(f"{Fore.RED}Failed to import {symbol}{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def update_today_data(self):
        """Update today's data for all stocks"""
        try:
            importer = StockImporter(self.db)
            print(f"{Fore.YELLOW}Updating today's data...{Style.RESET_ALL}")
            
            # Get all symbols
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT symbol FROM stocks")
                symbols = [row[0] for row in cursor.fetchall()]
            
            print(f"Updating {len(symbols)} stocks...")
            success_count = 0
            
            for i, symbol in enumerate(symbols, 1):
                try:
                    importer.update_latest_data(symbol)
                    success_count += 1
                    if i % 10 == 0:
                        print(f"Progress: {i}/{len(symbols)}")
                except Exception as e:
                    logger.error(f"Failed to update {symbol}: {e}")
            
            print(f"{Fore.GREEN}Updated {success_count}/{len(symbols)} stocks{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def portfolio_menu(self):
        """Handle portfolio operations"""
        while True:
            print(f"\n{Fore.CYAN}Portfolio Management:{Style.RESET_ALL}")
            print("1. View Current Portfolio")
            print("2. Add Position")
            print("3. Remove Position")
            print("4. View History")
            print("5. Performance Analysis")
            print("6. Risk Analysis")
            print("0. Back to Main Menu")
            
            choice = input(f"\n{Fore.GREEN}Enter choice: {Style.RESET_ALL}")
            
            if choice == '1':
                self.view_portfolio()
            elif choice == '2':
                self.add_position()
            elif choice == '3':
                self.remove_position()
            elif choice == '4':
                self.view_history()
            elif choice == '5':
                self.performance_analysis()
            elif choice == '6':
                self.risk_analysis()
            elif choice == '0':
                break
    
    def view_portfolio(self):
        """View current portfolio status"""
        try:
            portfolio = self.portfolio_manager.get_current_portfolio()
            
            if not portfolio:
                print(f"{Fore.YELLOW}Portfolio is empty{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.GREEN}Current Portfolio:{Style.RESET_ALL}")
            print(f"{'='*80}")
            print(f"{'Symbol':<10} {'Shares':<10} {'Avg Price':<12} {'Current':<12} {'P&L':<12} {'P&L %':<10}")
            print(f"{'='*80}")
            
            total_value = 0
            total_cost = 0
            
            for position in portfolio:
                symbol = position['symbol']
                shares = position['shares']
                avg_price = position['average_price']
                current_price = position.get('current_price', 0)
                
                if current_price:
                    value = shares * current_price
                    cost = shares * avg_price
                    pnl = value - cost
                    pnl_pct = (pnl / cost) * 100 if cost > 0 else 0
                    
                    color = Fore.GREEN if pnl >= 0 else Fore.RED
                    
                    print(f"{symbol:<10} {shares:<10} ${avg_price:<11.2f} ${current_price:<11.2f} "
                          f"{color}${pnl:<11.2f} {pnl_pct:<9.2f}%{Style.RESET_ALL}")
                    
                    total_value += value
                    total_cost += cost
            
            print(f"{'='*80}")
            
            total_pnl = total_value - total_cost
            total_pnl_pct = (total_pnl / total_cost) * 100 if total_cost > 0 else 0
            
            color = Fore.GREEN if total_pnl >= 0 else Fore.RED
            
            print(f"\n{Fore.CYAN}Portfolio Summary:{Style.RESET_ALL}")
            print(f"Total Cost: ${total_cost:,.2f}")
            print(f"Total Value: ${total_value:,.2f}")
            print(f"Total P&L: {color}${total_pnl:,.2f} ({total_pnl_pct:.2f}%){Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def add_position(self):
        """Add position to portfolio"""
        try:
            symbol = input("Enter stock symbol: ").upper()
            shares = int(input("Enter number of shares: "))
            price = float(input("Enter price per share: "))
            
            success = self.portfolio_manager.add_position("default", symbol, shares, price)
            
            if success:
                print(f"{Fore.GREEN}Successfully added {shares} shares of {symbol} at ${price:.2f}{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to add position{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def remove_position(self):
        """Remove position from portfolio"""
        try:
            symbol = input("Enter stock symbol to sell: ").upper()
            shares = input("Enter number of shares (or press Enter for all): ")
            
            shares = int(shares) if shares else None
            
            success = self.portfolio_manager.remove_position("default", symbol, shares)
            
            if success:
                print(f"{Fore.GREEN}Successfully sold position{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to sell position{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def view_history(self):
        """View transaction history"""
        try:
            history = self.portfolio_manager.get_transaction_history("default")
            
            if not history:
                print(f"{Fore.YELLOW}No transaction history{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.GREEN}Transaction History:{Style.RESET_ALL}")
            print(f"{'='*100}")
            print(f"{'Date':<20} {'Type':<8} {'Symbol':<10} {'Shares':<10} {'Price':<10} {'Total':<12}")
            print(f"{'='*100}")
            
            for tx in history:
                date = tx['transaction_date'][:19]
                tx_type = tx['transaction_type']
                symbol = tx['symbol']
                shares = tx['shares']
                price = tx['price']
                total = shares * price
                
                color = Fore.GREEN if tx_type == 'SELL' else Fore.RED
                
                print(f"{date:<20} {color}{tx_type:<8}{Style.RESET_ALL} {symbol:<10} "
                      f"{shares:<10} ${price:<9.2f} ${total:<11.2f}")
            
            print(f"{'='*100}")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def performance_analysis(self):
        """Analyze portfolio performance"""
        try:
            analyzer = PerformanceAnalyzer(self.db)
            metrics = analyzer.calculate_metrics("default")
            
            print(f"\n{Fore.GREEN}Performance Analysis:{Style.RESET_ALL}")
            print(f"{'='*60}")
            print(f"Total Return: {metrics['total_return']:.2f}%")
            print(f"Annual Return: {metrics['annual_return']:.2f}%")
            print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
            print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
            print(f"Max Drawdown: {metrics['max_drawdown']:.2f}%")
            print(f"Win Rate: {metrics['win_rate']:.2f}%")
            print(f"Profit Factor: {metrics['profit_factor']:.2f}")
            print(f"Average Win: ${metrics['avg_win']:.2f}")
            print(f"Average Loss: ${metrics['avg_loss']:.2f}")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def risk_analysis(self):
        """Analyze portfolio risk"""
        try:
            from utils.risk_management import RiskManager
            risk_manager = RiskManager(self.db)
            
            risk_metrics = risk_manager.analyze_portfolio("default")
            
            print(f"\n{Fore.GREEN}Risk Analysis:{Style.RESET_ALL}")
            print(f"{'='*60}")
            print(f"Portfolio Beta: {risk_metrics['beta']:.2f}")
            print(f"Value at Risk (95%): ${risk_metrics['var_95']:,.2f}")
            print(f"Conditional VaR (95%): ${risk_metrics['cvar_95']:,.2f}")
            print(f"Correlation with SPY: {risk_metrics['correlation']:.2f}")
            print(f"Portfolio Volatility: {risk_metrics['volatility']:.2f}%")
            print(f"{'='*60}")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def scanner_menu(self):
        """Handle scanner operations with optimizer integration"""
        while True:
            print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN} 📊 DAILY SCANNER MENU - WITH OPTIMIZER{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            
            # Check optimizer status first
            self.check_optimizer_status()
            
            print("\n1. Run Daily Scan (View Only)")
            print("2. Run Daily Scan (Execute Trades)")
            print("3. View Historical Scan Results")
            print("4. Load/Update Optimized Configuration")
            print("5. View Current Configuration")
            print("0. Back to Main Menu")
            
            choice = input(f"\n{Fore.GREEN}Enter choice: {Style.RESET_ALL}")
            
            if choice == '1':
                self.run_daily_scan_optimized(execute=False)
            elif choice == '2':
                self.run_daily_scan_optimized(execute=True)
            elif choice == '3':
                self.view_historical_scans()
            elif choice == '4':
                self.update_scanner_configuration()
            elif choice == '5':
                self.view_current_configuration()
            elif choice == '0':
                break

    def view_historical_scans(self):
        """View historical scan results"""
        try:
            date = input("Enter date (YYYY-MM-DD) or press Enter for last 7 days: ").strip()
            
            with self.db.get_connection() as conn:
                if date:
                    query = """
                        SELECT symbol, signal_type, signal_strength, reason, price, timestamp
                        FROM scan_results
                        WHERE DATE(timestamp) = ?
                        ORDER BY signal_strength DESC
                        LIMIT 50
                    """
                    df = pd.read_sql_query(query, conn, params=(date,))
                else:
                    query = """
                        SELECT DATE(timestamp) as date, 
                            COUNT(*) as total_signals,
                            SUM(CASE WHEN signal_type = 'BUY' THEN 1 ELSE 0 END) as buy_signals,
                            SUM(CASE WHEN signal_type = 'SELL' THEN 1 ELSE 0 END) as sell_signals
                        FROM scan_results
                        WHERE timestamp >= datetime('now', '-7 days')
                        GROUP BY DATE(timestamp)
                        ORDER BY date DESC
                    """
                    df = pd.read_sql_query(query, conn)
                
                if not df.empty:
                    print(f"\n{Fore.CYAN}Historical Scan Results:{Style.RESET_ALL}")
                    print(df.to_string(index=False))
                else:
                    print(f"{Fore.YELLOW}No results found{Style.RESET_ALL}")
                    
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")

    def update_scanner_configuration(self):
        """Force reload of optimized configuration"""
        try:
            print(f"\n{Fore.YELLOW}Checking for latest optimized configuration...{Style.RESET_ALL}")
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get latest optimization result
                cursor.execute("""
                    SELECT id, parameters, performance_score, created_at
                    FROM trading_configs
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                
                if result:
                    # Deactivate all configs
                    cursor.execute("UPDATE trading_configs SET is_active = 0")
                    
                    # Activate the latest
                    cursor.execute("UPDATE trading_configs SET is_active = 1 WHERE id = ?", (result[0],))
                    conn.commit()
                    
                    print(f"{Fore.GREEN}✅ Configuration updated!{Style.RESET_ALL}")
                    print(f"Using optimization from: {result[3]}")
                    print(f"Performance score: {result[2]:.2f}")
                else:
                    print(f"{Fore.YELLOW}No optimization results found. Run optimizer first.{Style.RESET_ALL}")
                    
        except Exception as e:
            print(f"{Fore.RED}Error updating configuration: {e}{Style.RESET_ALL}")

    def view_current_configuration(self):
        """View the current configuration being used"""
        config, is_optimized = self.load_optimized_configuration()
        
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN} CURRENT CONFIGURATION - {'OPTIMIZED' if is_optimized else 'DEFAULT'}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        print(f"\n📊 Entry Criteria:")
        print(f"  Min Entry Score: {config.get('min_entry_score', 70)}")
        print(f"  Min Confidence: {config.get('min_confidence', 0.65):.2f}")
        
        print(f"\n💰 Position Management:")
        print(f"  Max Positions: {config.get('max_positions', 10)}")
        print(f"  Max Position %: {config.get('max_position_pct', 0.10):.1%}")
        
        print(f"\n🛡️ Risk Management:")
        print(f"  Stop Loss: {config.get('stop_loss_pct', 0.05):.1%}")
        print(f"  Take Profit: {config.get('profit_target_pct', 0.10):.1%}")
        
        print(f"\n📈 Indicator Thresholds:")
        print(f"  RSI Oversold: {config.get('rsi_oversold', 30)}")
        print(f"  RSI Overbought: {config.get('rsi_overbought', 70)}")
        print(f"  Volume Spike: {config.get('volume_spike_threshold', 2.0)}x")
        print(f"  Bollinger Period: {config.get('bb_period', 20)}")
        
        print(f"\n⚖️ Signal Weights:")
        print(f"  Technical: {config.get('technical_weight', 0.4):.1%}")
        print(f"  Momentum: {config.get('momentum_weight', 0.25):.1%}")
        print(f"  Volume: {config.get('volume_weight', 0.15):.1%}")

    def run_daily_scan_optimized(self, execute=False):
        """Run daily scan using OPTIMIZED configuration"""
        try:
            # Get portfolio name
            portfolio_name = input("Enter portfolio name (or press Enter for 'default'): ").strip() or "default"
            
            # Get scan date
            scan_date = input("Enter scan date (YYYY-MM-DD) or press Enter for today: ").strip()
            if not scan_date:
                scan_date = datetime.now().strftime('%Y-%m-%d')
            
            print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.CYAN} 📅 SCANNING DATE: {scan_date}{Style.RESET_ALL}")
            print(f"{Fore.CYAN} 💼 PORTFOLIO: {portfolio_name}{Style.RESET_ALL}")
            
            # LOAD OPTIMIZED CONFIGURATION
            config, is_optimized = self.load_optimized_configuration()
            
            if is_optimized:
                print(f"\n{Fore.GREEN}Parameters being used:{Style.RESET_ALL}")
                print(f"  Min Entry Score: {config.get('min_entry_score', 70)}")
                print(f"  Stop Loss: {config.get('stop_loss_pct', 0.05):.1%}")
                print(f"  Take Profit: {config.get('profit_target_pct', 0.10):.1%}")
                print(f"  RSI Oversold: {config.get('rsi_oversold', 30)}")
                print(f"  RSI Overbought: {config.get('rsi_overbought', 70)}")
                print(f"  Max Positions: {config.get('max_positions', 10)}")
            
            print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
            
            # Get portfolio info
            holdings = self.get_portfolio_holdings(portfolio_name)
            cash_balance = self.get_portfolio_cash(portfolio_name)
            
            print(f"\nCurrent cash: ${cash_balance:,.2f}")
            print(f"Current holdings: {len(holdings)} positions")
            
            # RUN SCAN WITH OPTIMIZED CONFIG
            print(f"\n{Fore.YELLOW}Running scan with {'OPTIMIZED' if is_optimized else 'DEFAULT'} parameters...{Style.RESET_ALL}")
            signals = self.scan_with_config(portfolio_name, scan_date, holdings, config)
            
            if not signals:
                print(f"\n{Fore.YELLOW}No signals found meeting the criteria{Style.RESET_ALL}")
                if is_optimized:
                    print("The optimized criteria might be too strict. Try:")
                    print("  • Running optimizer with different date range")
                    print("  • Adjusting minimum score threshold")
                return
            
            # Display results
            print(f"\n{Fore.GREEN}✅ Found {len(signals)} signals using {'OPTIMIZED' if is_optimized else 'DEFAULT'} configuration{Style.RESET_ALL}")
            
            self.display_signals_with_scores(signals, holdings, scan_date, config)
            self.display_recommendations_optimized(signals, holdings, cash_balance, portfolio_name, scan_date, config, execute)
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()

    def get_portfolio_cash(self, portfolio_name):
        """Get portfolio cash balance"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT cash_balance FROM portfolios WHERE name = ?", (portfolio_name,))
                result = cursor.fetchone()
                if result:
                    return result[0] or 100000
        except:
            pass
        return 100000

    def get_portfolio_holdings(self, portfolio_name):
        """Get portfolio holdings"""
        holdings = {}
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("SELECT id FROM portfolios WHERE name = ?", (portfolio_name,))
                result = cursor.fetchone()
                
                if result:
                    portfolio_id = result[0]
                    
                    # Try different table names
                    for table in ['portfolio_holdings', 'holdings', 'positions']:
                        try:
                            cursor.execute(f"""
                                SELECT symbol, quantity, avg_price
                                FROM {table}
                                WHERE portfolio_id = ? AND quantity > 0
                            """, (portfolio_id,))
                            
                            for row in cursor.fetchall():
                                holdings[row[0]] = {
                                    'quantity': row[1],
                                    'avg_price': row[2] or 0
                                }
                            break
                        except:
                            continue
        except:
            pass
        
        return holdings


    def display_recommendations_optimized(self, signals, holdings, cash_balance, portfolio_name, scan_date, config, execute):
        """Display trading recommendations based on optimized signals"""
        
        # Filter by type and holdings
        buy_candidates = [s for s in signals if s['type'] == 'BUY' and s['symbol'] not in holdings]
        sell_candidates = [s for s in signals if s['type'] == 'SELL' and s['symbol'] in holdings]
        
        # Sort by score
        buy_candidates.sort(key=lambda x: x['weighted_score'], reverse=True)
        sell_candidates.sort(key=lambda x: x['weighted_score'], reverse=True)
        
        print(f"\n{Fore.CYAN}{'='*120}{Style.RESET_ALL}")
        print(f"{Fore.CYAN} 💡 OPTIMIZED TRADING RECOMMENDATIONS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*120}{Style.RESET_ALL}")
        
        recommendations = []
        
        # Calculate position sizing using optimized parameters
        max_positions = config.get('max_positions', 10)
        max_position_pct = config.get('max_position_pct', 0.10)
        position_size = min(cash_balance * max_position_pct, cash_balance / max_positions)
        
        print(f"Portfolio: {portfolio_name}")
        print(f"Available Cash: ${cash_balance:,.2f}")
        print(f"Max Positions: {max_positions}")
        print(f"Position Size: ${position_size:,.2f} ({max_position_pct:.1%} of capital)")
        print("-" * 120)
        
        # BUY recommendations
        if buy_candidates:
            current_positions = len(holdings)
            slots_available = max(0, max_positions - current_positions)
            
            if slots_available > 0:
                print(f"\n{Fore.GREEN}🎯 BUY RECOMMENDATIONS (Top {min(slots_available, len(buy_candidates))}):{Style.RESET_ALL}")
                print(f"{'Symbol':<8} {'Score':<8} {'Shares':<10} {'Price':<10} {'Total':<12} {'Stop Loss':<10} {'Take Profit':<12}")
                print("-" * 80)
                
                for signal in buy_candidates[:slots_available]:
                    if signal['price'] > 0:
                        shares = int(position_size / signal['price'])
                        total = shares * signal['price']
                        
                        if total <= cash_balance:
                            rec = {
                                'symbol': signal['symbol'],
                                'type': 'BUY',
                                'shares': shares,
                                'price': signal['price'],
                                'total': total,
                                'stop_loss': signal['stop_loss'],
                                'take_profit': signal['take_profit'],
                                'score': signal['weighted_score']
                            }
                            recommendations.append(rec)
                            
                            print(f"{signal['symbol']:<8} {signal['weighted_score']:<8.1f} {shares:<10,} "
                                f"${signal['price']:<9.2f} ${total:<11,.2f} "
                                f"${signal['stop_loss']:<9.2f} ${signal['take_profit']:<11.2f}")
                            
                            cash_balance -= total
            else:
                print(f"\n{Fore.YELLOW}⚠️  Maximum positions ({max_positions}) reached. Sell some positions first.{Style.RESET_ALL}")
        
        # SELL recommendations
        if sell_candidates:
            print(f"\n{Fore.RED}⚠️  SELL RECOMMENDATIONS:{Style.RESET_ALL}")
            print(f"{'Symbol':<8} {'Score':<8} {'Current Price':<15} {'Reason':<40}")
            print("-" * 80)
            
            for signal in sell_candidates:
                rec = {
                    'symbol': signal['symbol'],
                    'type': 'SELL',
                    'price': signal['price'],
                    'score': signal['weighted_score']
                }
                recommendations.append(rec)
                
                print(f"{signal['symbol']:<8} {signal['weighted_score']:<8.1f} "
                    f"${signal['price']:<14.2f} {signal['reason'][:40]}")
        
        if recommendations:
            print(f"\n{Fore.CYAN}Total Recommendations: {len(recommendations)}{Style.RESET_ALL}")
            
            if execute:
                confirm = input(f"\n{Fore.YELLOW}Execute these trades? (yes/no): {Style.RESET_ALL}").lower()
                if confirm == 'yes':
                    self.execute_optimized_trades(recommendations, portfolio_name)

    def execute_optimized_trades(self, recommendations, portfolio_name):
        """Execute trades based on optimized recommendations"""
        executed = 0
        failed = 0
        
        print(f"\n{Fore.YELLOW}Executing {len(recommendations)} optimized trades...{Style.RESET_ALL}")
        
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get portfolio
                cursor.execute("SELECT id, cash_balance FROM portfolios WHERE name = ?", (portfolio_name,))
                result = cursor.fetchone()
                
                if not result:
                    print(f"{Fore.RED}Portfolio not found!{Style.RESET_ALL}")
                    return
                
                portfolio_id, cash_balance = result
                
                for rec in recommendations:
                    try:
                        if rec['type'] == 'BUY':
                            if rec['total'] <= cash_balance:
                                cursor.execute("""
                                    INSERT INTO transactions 
                                    (portfolio_id, symbol, transaction_type, quantity, price, total_amount, transaction_date)
                                    VALUES (?, ?, 'BUY', ?, ?, ?, datetime('now'))
                                """, (portfolio_id, rec['symbol'], rec['shares'], rec['price'], rec['total']))
                                
                                cash_balance -= rec['total']
                                
                                print(f"{Fore.GREEN}✅ Bought {rec['shares']:,} {rec['symbol']} @ ${rec['price']:.2f} "
                                    f"(Score: {rec['score']:.1f}){Style.RESET_ALL}")
                                executed += 1
                            else:
                                print(f"{Fore.RED}❌ Insufficient funds for {rec['symbol']}{Style.RESET_ALL}")
                                failed += 1
                                
                        elif rec['type'] == 'SELL':
                            # Execute sell
                            print(f"{Fore.GREEN}✅ Sell signal for {rec['symbol']} recorded (Score: {rec['score']:.1f}){Style.RESET_ALL}")
                            executed += 1
                            
                    except Exception as e:
                        print(f"{Fore.RED}❌ Failed {rec['type']} {rec['symbol']}: {e}{Style.RESET_ALL}")
                        failed += 1
                
                # Update cash
                cursor.execute("UPDATE portfolios SET cash_balance = ? WHERE id = ?", (cash_balance, portfolio_id))
                conn.commit()
                
        except Exception as e:
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}Summary: {executed} executed, {failed} failed{Style.RESET_ALL}")

    # Helper functions for calculations
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        if len(prices) < period + 1:
            return 50
        
        deltas = np.diff(prices)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        
        avg_gain = np.mean(gains[-period:])
        avg_loss = np.mean(losses[-period:])
        
        if avg_loss == 0:
            return 100
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def calculate_ema(self, prices, period):
        """Calculate EMA"""
        if len(prices) < period:
            return prices
        
        ema = np.zeros_like(prices)
        ema[0] = prices[0]
        multiplier = 2 / (period + 1)
        
        for i in range(1, len(prices)):
            ema[i] = (prices[i] * multiplier) + (ema[i-1] * (1 - multiplier))
        
        return ema

    def display_signals_with_scores(self, signals, holdings, scan_date, config):
        """Display signals with their weighted scores"""
        
        # Separate buy and sell
        buy_signals = [s for s in signals if s['type'] == 'BUY' and s['symbol'] not in holdings]
        sell_signals = [s for s in signals if s['type'] == 'SELL' and s['symbol'] in holdings]
        
        # Display BUY signals
        if buy_signals:
            print(f"\n{Fore.GREEN}{'='*120}{Style.RESET_ALL}")
            print(f"{Fore.GREEN} 📈 BUY SIGNALS (Not in Portfolio) - Using {'OPTIMIZED' if config else 'DEFAULT'} Config{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'='*120}{Style.RESET_ALL}")
            print(f"{'#':<4} {'Symbol':<8} {'Score':<8} {'Price':<10} {'Stop Loss':<10} {'Take Profit':<12} {'Indicator':<12} {'Reason':<45}")
            print("-" * 120)
            
            for i, signal in enumerate(buy_signals[:30], 1):  # Show top 30
                score = signal['weighted_score']
                
                # Color by score
                if score >= 80:
                    color = Fore.GREEN + Style.BRIGHT
                elif score >= 70:
                    color = Fore.YELLOW
                else:
                    color = Style.RESET_ALL
                
                print(f"{color}{i:<4} {signal['symbol']:<8} {score:<8.1f} "
                    f"${signal['price']:<9.2f} ${signal['stop_loss']:<9.2f} "
                    f"${signal['take_profit']:<11.2f} {signal['indicator']:<12} "
                    f"{signal['reason'][:45]:<45}{Style.RESET_ALL}")
        
        # Display SELL signals
        if sell_signals:
            print(f"\n{Fore.RED}{'='*120}{Style.RESET_ALL}")
            print(f"{Fore.RED} 📉 SELL SIGNALS (In Portfolio){Style.RESET_ALL}")
            print(f"{Fore.RED}{'='*120}{Style.RESET_ALL}")
            print(f"{'#':<4} {'Symbol':<8} {'Score':<8} {'Price':<10} {'Indicator':<12} {'Reason':<45}")
            print("-" * 120)
            
            for i, signal in enumerate(sell_signals[:30], 1):
                score = signal['weighted_score']
                print(f"{i:<4} {signal['symbol']:<8} {score:<8.1f} "
                    f"${signal['price']:<9.2f} {signal['indicator']:<12} "
                    f"{signal['reason'][:45]:<45}")
        
        # Summary
        print(f"\n{Fore.CYAN}SCAN SUMMARY for {scan_date}:{Style.RESET_ALL}")
        print(f"  Minimum Entry Score: {config.get('min_entry_score', 70)}")
        print(f"  Buy Signals: {len(buy_signals)} (actionable)")
        print(f"  Sell Signals: {len(sell_signals)} (actionable)")
        
        if buy_signals:
            avg_score = sum(s['weighted_score'] for s in buy_signals) / len(buy_signals)
            print(f"  Average Buy Score: {avg_score:.1f}")

    def scan_with_config(self, portfolio_name, scan_date, holdings, config):
        """Scan using specific configuration (optimized or default)"""
        all_signals = []
        
        try:
            # Get stocks to scan
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT DISTINCT symbol 
                    FROM price_data 
                    WHERE date <= ?
                    GROUP BY symbol
                    HAVING COUNT(*) >= 100
                    ORDER BY symbol
                """, (scan_date,))
                
                stocks = [row[0] for row in cursor.fetchall()]
            
            if not stocks:
                return []
            
            print(f"Analyzing {len(stocks)} stocks...")
            print("Progress: ", end="", flush=True)
            
            for i, symbol in enumerate(stocks):
                if (i + 1) % 20 == 0:
                    print(f"{i+1}...", end="", flush=True)
                
                # Analyze with configuration
                signals = self.analyze_stock_optimized(symbol, scan_date, config)
                
                for signal in signals:
                    # Calculate weighted score
                    score = self.calculate_weighted_score(signal, config)
                    signal['weighted_score'] = score
                    
                    # Only include if meets minimum score
                    if score >= config.get('min_entry_score', 70):
                        signal['in_portfolio'] = symbol in holdings
                        signal['scan_date'] = scan_date
                        
                        # Apply optimized stop loss and take profit
                        if signal['type'] == 'BUY':
                            signal['stop_loss'] = round(signal['price'] * (1 - config.get('stop_loss_pct', 0.05)), 2)
                            signal['take_profit'] = round(signal['price'] * (1 + config.get('profit_target_pct', 0.10)), 2)
                        
                        all_signals.append(signal)
            
            print(" Complete!")
            
            # Sort by score
            all_signals.sort(key=lambda x: x['weighted_score'], reverse=True)
            
        except Exception as e:
            print(f"\nError: {e}")
        
        return all_signals

    def calculate_weighted_score(self, signal, config):
        """Calculate weighted score using OPTIMIZED weights"""
        
        # Get weight for this indicator type
        weight_map = {
            'technical': config.get('technical_weight', 0.4),
            'momentum': config.get('momentum_weight', 0.25),
            'volume': config.get('volume_weight', 0.15),
            'sentiment': config.get('sentiment_weight', 0.1),
            'fundamental': config.get('fundamental_weight', 0.1)
        }
        
        indicator_type = signal.get('indicator_type', 'technical')
        weight = weight_map.get(indicator_type, 0.2)
        
        # Base strength from signal
        strength = signal.get('strength', 0.5)
        
        # Calculate weighted score (0-100)
        weighted_score = strength * weight * 100
        
        # Boost for multiple confirmations (would need to track this)
        confidence_boost = 1.0
        
        return round(weighted_score * confidence_boost, 2)

    def analyze_stock_optimized(self, symbol, scan_date, config):
        """Analyze stock using optimized configuration"""
        signals = []
        
        try:
            with self.db.get_connection() as conn:
                # Get price data
                query = """
                    SELECT date, open, high, low, close, volume
                    FROM price_data
                    WHERE symbol = ? AND date <= ?
                    ORDER BY date DESC
                    LIMIT 200
                """
                
                df = pd.read_sql_query(query, conn, params=(symbol, scan_date))
                
                if len(df) < 50:
                    return signals
                
                # Reverse for chronological order
                df = df.iloc[::-1].reset_index(drop=True)
                
                # Convert to numeric
                for col in ['open', 'high', 'low', 'close', 'volume']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                
                if len(df) < 50:
                    return signals
                
                close = df['close'].values
                high = df['high'].values
                low = df['low'].values
                volume = df['volume'].values
                current_price = close[-1]
                
                # RSI with OPTIMIZED thresholds
                rsi = self.calculate_rsi(close, 14)
                rsi_oversold = config.get('rsi_oversold', 30)
                rsi_overbought = config.get('rsi_overbought', 70)
                
                if rsi < rsi_oversold:
                    signals.append({
                        'symbol': symbol,
                        'type': 'BUY',
                        'indicator': 'RSI',
                        'value': round(rsi, 2),
                        'reason': f'RSI oversold at {rsi:.1f} (optimized threshold: {rsi_oversold})',
                        'strength': round((rsi_oversold - rsi) / rsi_oversold, 2),
                        'price': round(current_price, 2),
                        'indicator_type': 'technical'
                    })
                elif rsi > rsi_overbought:
                    signals.append({
                        'symbol': symbol,
                        'type': 'SELL',
                        'indicator': 'RSI',
                        'value': round(rsi, 2),
                        'reason': f'RSI overbought at {rsi:.1f} (optimized threshold: {rsi_overbought})',
                        'strength': round((rsi - rsi_overbought) / (100 - rsi_overbought), 2),
                        'price': round(current_price, 2),
                        'indicator_type': 'technical'
                    })
                
                # MACD with optimized parameters
                ema_12 = self.calculate_ema(close, 12)
                ema_26 = self.calculate_ema(close, 26)
                macd = ema_12 - ema_26
                signal_line = self.calculate_ema(macd, 9)
                macd_threshold = config.get('macd_threshold', 0.0)
                
                if len(macd) >= 2 and len(signal_line) >= 2:
                    if macd[-2] <= signal_line[-2] and macd[-1] > signal_line[-1] and macd[-1] > macd_threshold:
                        signals.append({
                            'symbol': symbol,
                            'type': 'BUY',
                            'indicator': 'MACD',
                            'value': round(macd[-1] - signal_line[-1], 4),
                            'reason': f'MACD bullish crossover (optimized threshold: {macd_threshold})',
                            'strength': round(min(1.0, abs(macd[-1] - signal_line[-1]) * 50), 2),
                            'price': round(current_price, 2),
                            'indicator_type': 'momentum'
                        })
                
                # Volume spike with OPTIMIZED threshold
                volume_threshold = config.get('volume_spike_threshold', 2.0)
                avg_volume = np.mean(volume[-20:])
                current_volume = volume[-1]
                
                if avg_volume > 0:
                    volume_ratio = current_volume / avg_volume
                    if volume_ratio > volume_threshold:
                        price_change = (close[-1] - close[-2]) / close[-2] * 100 if close[-2] > 0 else 0
                        if abs(price_change) > 2:
                            signals.append({
                                'symbol': symbol,
                                'type': 'BUY' if price_change > 0 else 'SELL',
                                'indicator': 'VOLUME',
                                'value': round(volume_ratio, 2),
                                'reason': f'Volume {volume_ratio:.1f}x average (optimized threshold: {volume_threshold}x)',
                                'strength': round(min(1.0, volume_ratio / (volume_threshold * 2)), 2),
                                'price': round(current_price, 2),
                                'indicator_type': 'volume'
                            })
                
                # Bollinger Bands with OPTIMIZED parameters
                bb_period = int(config.get('bb_period', 20))
                bb_std = config.get('bb_std', 2)
                
                if len(close) >= bb_period:
                    bb_middle = np.mean(close[-bb_period:])
                    bb_std_dev = np.std(close[-bb_period:])
                    bb_upper = bb_middle + (bb_std_dev * bb_std)
                    bb_lower = bb_middle - (bb_std_dev * bb_std)
                    
                    if current_price <= bb_lower:
                        signals.append({
                            'symbol': symbol,
                            'type': 'BUY',
                            'indicator': 'BB',
                            'value': round((bb_middle - current_price) / bb_middle * 100, 2),
                            'reason': f'Price at lower BB (optimized: {bb_period} period, {bb_std} std)',
                            'strength': round(min(1.0, (bb_middle - current_price) / bb_middle * 10), 2),
                            'price': round(current_price, 2),
                            'indicator_type': 'technical'
                        })
                
                # Moving average signals
                if len(close) >= 50:
                    sma_20 = np.mean(close[-20:])
                    sma_50 = np.mean(close[-50:])
                    
                    if sma_20 > sma_50 and current_price > sma_20:
                        signals.append({
                            'symbol': symbol,
                            'type': 'BUY',
                            'indicator': 'MA_TREND',
                            'value': round((current_price - sma_50) / sma_50 * 100, 2),
                            'reason': 'Uptrend confirmed (Price > SMA20 > SMA50)',
                            'strength': 0.7,
                            'price': round(current_price, 2),
                            'indicator_type': 'technical'
                        })
                
        except Exception as e:
            logger.debug(f"Error analyzing {symbol}: {e}")
        
        return signals

    def check_optimizer_status(self):
        """Check if optimizer has been run and configuration exists"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if trading_configs table exists
                cursor.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table' AND name='trading_configs'
                """)
                
                if not cursor.fetchone():
                    print(f"\n{Fore.YELLOW}⚠️  OPTIMIZER NOT RUN YET!{Style.RESET_ALL}")
                    print("The scanner will use DEFAULT parameters.")
                    print("To optimize: Main Menu > 5. Optimizer > 1. Run Optimization")
                    return False
                
                # Check for active configuration
                cursor.execute("""
                    SELECT COUNT(*) FROM trading_configs WHERE is_active = 1
                """)
                
                count = cursor.fetchone()[0]
                
                if count == 0:
                    print(f"\n{Fore.YELLOW}⚠️  No optimized configuration active!{Style.RESET_ALL}")
                    print("Using DEFAULT parameters. Run optimizer first for better results.")
                    return False
                else:
                    print(f"\n{Fore.GREEN}✅ Optimized configuration is ACTIVE{Style.RESET_ALL}")
                    return True
                    
        except Exception as e:
            print(f"{Fore.YELLOW}Configuration check failed: {e}{Style.RESET_ALL}")
            return False

    def load_optimized_configuration(self):
        """Load the best configuration from optimizer results"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get active optimized configuration
                cursor.execute("""
                    SELECT parameters, performance_score, total_return, sharpe_ratio, created_at
                    FROM trading_configs
                    WHERE is_active = 1
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                
                if result:
                    params = json.loads(result[0]) if isinstance(result[0], str) else result[0]
                    
                    print(f"\n{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
                    print(f"{Fore.GREEN} ✅ USING OPTIMIZED CONFIGURATION{Style.RESET_ALL}")
                    print(f"{Fore.GREEN}{'='*60}{Style.RESET_ALL}")
                    print(f"Performance Score: {result[1]:.2f}")
                    print(f"Total Return: {result[2]:.2%}")
                    print(f"Sharpe Ratio: {result[3]:.2f}")
                    print(f"Optimized on: {result[4]}")
                    
                    return params, True
                else:
                    print(f"\n{Fore.YELLOW}No optimized configuration found, using defaults{Style.RESET_ALL}")
                    return self.get_default_configuration(), False
                    
        except Exception as e:
            print(f"{Fore.YELLOW}Loading defaults due to error: {e}{Style.RESET_ALL}")
            return self.get_default_configuration(), False

    def get_default_configuration(self):
        """Default configuration if no optimized one exists"""
        return {
            'min_entry_score': 70.0,
            'min_confidence': 0.65,
            'max_positions': 10,
            'max_position_pct': 0.10,
            'stop_loss_pct': 0.05,
            'profit_target_pct': 0.10,
            'trailing_stop_pct': 0.03,
            'max_hold_days': 30,
            'technical_weight': 0.40,
            'momentum_weight': 0.25,
            'volume_weight': 0.15,
            'sentiment_weight': 0.10,
            'fundamental_weight': 0.10,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'macd_threshold': 0.0,
            'volume_spike_threshold': 2.0,
            'bb_period': 20,
            'bb_std': 2
        }

    def view_scan_results(self):
        """View latest scan results - SIMPLE VERSION"""
        try:
            print(f"\n{Fore.CYAN}Retrieving latest scan results...{Style.RESET_ALL}")
            
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get recent scan results
                cursor.execute("""
                    SELECT symbol, signal_type, signal_strength, reason, price, timestamp
                    FROM scan_results
                    WHERE DATE(timestamp) >= DATE('now', '-1 day')
                    ORDER BY timestamp DESC, signal_strength DESC
                    LIMIT 50
                """)
                
                results = cursor.fetchall()
                
                if not results:
                    # Try signals table
                    cursor.execute("""
                        SELECT symbol, signal_type, signal_strength, reason, 0, generated_date
                        FROM signals
                        WHERE status = 'ACTIVE'
                        ORDER BY generated_date DESC
                        LIMIT 50
                    """)
                    results = cursor.fetchall()
                
                if results:
                    # Convert to signal format
                    signals = []
                    for row in results:
                        signals.append({
                            'symbol': row[0],
                            'type': row[1],
                            'strength': row[2] or 0.5,
                            'reason': row[3] or 'Signal generated',
                            'price': row[4] or 0,
                            'indicator': 'SCAN'
                        })
                    
                    print(f"\n{Fore.GREEN}Found {len(signals)} recent signals{Style.RESET_ALL}")
                    self.display_scan_results(signals)
                else:
                    print(f"{Fore.YELLOW}No scan results found. Run a scan first.{Style.RESET_ALL}")
                    
        except Exception as e:
            print(f"{Fore.RED}Error viewing results: {e}{Style.RESET_ALL}")

    def run_daily_scan(self, execute=False):
        """Run daily scan and show recommendations - FIXED VERSION"""
        try:
            portfolio_name = input("Enter portfolio name (or press Enter for default): ").strip() or "default"
            
            print(f"\n{Fore.YELLOW}Running daily scan...{Style.RESET_ALL}")
            
            # Run the scan
            signals = self.scanner.scan(portfolio_name)
            
            if not signals:
                print(f"{Fore.YELLOW}No trading signals found.{Style.RESET_ALL}")
                print("Tips:")
                print("  • Make sure you have imported stock data (Main Menu > Option 1)")
                print("  • Market might be in neutral state")
                return
            
            print(f"\n{Fore.GREEN}Found {len(signals)} trading signals!{Style.RESET_ALL}")
            
            # Display scan results
            self.display_scan_results(signals)
            
            # Display trading recommendations
            self.display_trading_recommendations(signals, portfolio_name, execute)
            
        except Exception as e:
            print(f"{Fore.RED}Error during scan: {str(e)}{Style.RESET_ALL}")
            import traceback
            traceback.print_exc()
            logger.error(f"Scan error: {e}")

    def display_trading_recommendations(self, signals, portfolio_name, execute=False):
        """Display trading recommendations - FIXED TO WORK WITHOUT get_portfolio"""
        if not signals:
            return
        
        # Get cash balance directly from database
        cash_balance = 100000  # Default
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Check if portfolio exists
                cursor.execute("""
                    SELECT id, cash_balance 
                    FROM portfolios 
                    WHERE name = ?
                """, (portfolio_name,))
                
                result = cursor.fetchone()
                if result:
                    portfolio_id = result[0]
                    cash_balance = result[1] or 100000
                else:
                    # Create portfolio if it doesn't exist
                    print(f"{Fore.YELLOW}Creating portfolio '{portfolio_name}'...{Style.RESET_ALL}")
                    cursor.execute("""
                        INSERT INTO portfolios (name, initial_capital, cash_balance, created_date, status)
                        VALUES (?, ?, ?, datetime('now'), 'ACTIVE')
                    """, (portfolio_name, 100000, 100000))
                    conn.commit()
                    portfolio_id = cursor.lastrowid
                    cash_balance = 100000
                    
        except Exception as e:
            print(f"{Fore.YELLOW}Using default cash balance: ${cash_balance:,.2f}{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN} 💰 TRADING RECOMMENDATIONS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"Portfolio: {portfolio_name}")
        print(f"Available Cash: ${cash_balance:,.2f}")
        print("-" * 80)
        
        # Get top buy signals
        buy_signals = [s for s in signals if s.get('type') == 'BUY']
        buy_signals.sort(key=lambda x: x.get('strength', 0), reverse=True)
        buy_signals = buy_signals[:5]  # Top 5
        
        recommendations = []
        
        if buy_signals:
            print(f"\n{Fore.GREEN}🎯 RECOMMENDED BUYS (Top 5):{Style.RESET_ALL}")
            print(f"{'Symbol':<10} {'Shares':<10} {'Price':<10} {'Total Cost':<15} {'Reason':<30}")
            print("-" * 80)
            
            # Calculate position size (equal weight or $10k max per position)
            max_positions = min(len(buy_signals), 5)
            position_size = min(cash_balance / max_positions, 10000) if max_positions > 0 else 0
            
            for signal in buy_signals:
                price = signal.get('price', 1)
                if price > 0:
                    shares = int(position_size / price)
                    total = shares * price
                    
                    if total <= cash_balance:
                        rec = {
                            'symbol': signal.get('symbol', 'N/A'),
                            'type': 'BUY',
                            'shares': shares,
                            'price': price,
                            'total': total,
                            'reason': signal.get('reason', ''),
                            'strength': signal.get('strength', 0)
                        }
                        recommendations.append(rec)
                        
                        print(f"{signal.get('symbol', 'N/A'):<10} "
                            f"{shares:<10,} "
                            f"${price:<9.2f} "
                            f"${total:<14,.2f} "
                            f"{signal.get('reason', '')[:30]:<30}")
        
        # Get sell signals
        sell_signals = [s for s in signals if s.get('type') == 'SELL']
        sell_signals.sort(key=lambda x: x.get('strength', 0), reverse=True)
        sell_signals = sell_signals[:5]  # Top 5
        
        if sell_signals:
            print(f"\n{Fore.RED}⚠️ RECOMMENDED SELLS (Top 5):{Style.RESET_ALL}")
            print(f"{'Symbol':<10} {'Current Price':<15} {'Reason':<40}")
            print("-" * 80)
            
            for signal in sell_signals:
                rec = {
                    'symbol': signal.get('symbol', 'N/A'),
                    'type': 'SELL',
                    'shares': 'ALL',
                    'price': signal.get('price', 0),
                    'reason': signal.get('reason', ''),
                    'strength': signal.get('strength', 0)
                }
                recommendations.append(rec)
                
                print(f"{signal.get('symbol', 'N/A'):<10} "
                    f"${signal.get('price', 0):<14.2f} "
                    f"{signal.get('reason', '')[:40]:<40}")
        
        print("-" * 80)
        print(f"Total recommendations: {len(recommendations)}")
        
        # Execute trades if requested
        if execute and recommendations:
            confirm = input(f"\n{Fore.YELLOW}Execute these trades? (yes/no): {Style.RESET_ALL}").lower()
            if confirm == 'yes':
                print(f"\n{Fore.YELLOW}Executing trades...{Style.RESET_ALL}")
                self.execute_recommendations(recommendations, portfolio_name)
        elif recommendations and not execute:
            print(f"\n{Fore.CYAN}To execute these trades, run scan with 'Execute Trades' option{Style.RESET_ALL}")

    def display_scan_results(self, signals):
        """Display scan results - WORKING VERSION"""
        if not signals:
            return
        
        # Separate buy and sell signals
        buy_signals = [s for s in signals if s.get('type') == 'BUY']
        sell_signals = [s for s in signals if s.get('type') == 'SELL']
        
        # Sort by strength
        buy_signals.sort(key=lambda x: x.get('strength', 0), reverse=True)
        sell_signals.sort(key=lambda x: x.get('strength', 0), reverse=True)
        
        # Display BUY signals
        if buy_signals:
            print(f"\n{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.GREEN} 📈 BUY SIGNALS{Style.RESET_ALL}")
            print(f"{Fore.GREEN}{'='*80}{Style.RESET_ALL}")
            print(f"{'Symbol':<10} {'Price':<10} {'Indicator':<12} {'Strength':<10} {'Reason':<35}")
            print("-" * 80)
            
            for signal in buy_signals[:15]:  # Show top 15
                strength_bar = '█' * int(signal.get('strength', 0) * 10)
                print(f"{signal.get('symbol', 'N/A'):<10} "
                    f"${signal.get('price', 0):<9.2f} "
                    f"{signal.get('indicator', 'N/A'):<12} "
                    f"{signal.get('strength', 0):<10.2f} "
                    f"{signal.get('reason', 'N/A')[:35]:<35}")
        
        # Display SELL signals
        if sell_signals:
            print(f"\n{Fore.RED}{'='*80}{Style.RESET_ALL}")
            print(f"{Fore.RED} 📉 SELL SIGNALS{Style.RESET_ALL}")
            print(f"{Fore.RED}{'='*80}{Style.RESET_ALL}")
            print(f"{'Symbol':<10} {'Price':<10} {'Indicator':<12} {'Strength':<10} {'Reason':<35}")
            print("-" * 80)
            
            for signal in sell_signals[:15]:  # Show top 15
                print(f"{signal.get('symbol', 'N/A'):<10} "
                    f"${signal.get('price', 0):<9.2f} "
                    f"{signal.get('indicator', 'N/A'):<12} "
                    f"{signal.get('strength', 0):<10.2f} "
                    f"{signal.get('reason', 'N/A')[:35]:<35}")
        
        print(f"\n{Fore.CYAN}Summary: {len(buy_signals)} BUY signals, {len(sell_signals)} SELL signals{Style.RESET_ALL}")


    def run_scan(self, execute=False):
        """Run daily scan"""
        try:
            print(f"{Fore.YELLOW}Running daily scan...{Style.RESET_ALL}")
            
            # Get portfolio
            portfolio_name = input("Enter portfolio name (or press Enter for default): ") or "default"
            
            # Run scan
            recommendations = self.scanner.scan(portfolio_name)
            
            if not recommendations:
                print(f"{Fore.YELLOW}No trading signals found{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.GREEN}Trading Recommendations:{Style.RESET_ALL}")
            print(f"{'='*100}")
            print(f"{'Action':<8} {'Symbol':<10} {'Shares':<10} {'Price':<10} {'Reason':<50}")
            print(f"{'='*100}")
            
            for rec in recommendations:
                action_color = Fore.GREEN if rec['action'] == 'BUY' else Fore.RED
                print(f"{action_color}{rec['action']:<8}{Style.RESET_ALL} "
                      f"{rec['symbol']:<10} {rec['shares']:<10} "
                      f"${rec['price']:<9.2f} {rec['reason']:<50}")
            
            print(f"{'='*100}")
            
            if execute:
                if input("\nExecute trades? (y/n): ").lower() == 'y':
                    self.execute_recommendations(recommendations)
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def execute_recommendations(self, recommendations, portfolio_name):
        """Execute trading recommendations - SIMPLIFIED VERSION"""
        if not recommendations:
            return
        
        executed = 0
        failed = 0
        
        print(f"\n{Fore.YELLOW}Executing {len(recommendations)} trades...{Style.RESET_ALL}")
        
        for rec in recommendations:
            try:
                symbol = rec['symbol']
                
                if rec['type'] == 'BUY':
                    # Simple buy execution
                    with self.db.get_connection() as conn:
                        cursor = conn.cursor()
                        
                        # Get portfolio ID
                        cursor.execute("SELECT id, cash_balance FROM portfolios WHERE name = ?", (portfolio_name,))
                        result = cursor.fetchone()
                        
                        if result:
                            portfolio_id, cash_balance = result
                            
                            if rec['total'] <= cash_balance:
                                # Record transaction
                                cursor.execute("""
                                    INSERT INTO transactions 
                                    (portfolio_id, symbol, transaction_type, quantity, price, total_amount, transaction_date)
                                    VALUES (?, ?, 'BUY', ?, ?, ?, datetime('now'))
                                """, (portfolio_id, symbol, rec['shares'], rec['price'], rec['total']))
                                
                                # Update cash
                                cursor.execute("""
                                    UPDATE portfolios 
                                    SET cash_balance = cash_balance - ?
                                    WHERE id = ?
                                """, (rec['total'], portfolio_id))
                                
                                conn.commit()
                                
                                print(f"{Fore.GREEN}✓ Bought {rec['shares']:,} shares of {symbol} at ${rec['price']:.2f}{Style.RESET_ALL}")
                                executed += 1
                            else:
                                print(f"{Fore.RED}✗ Insufficient funds for {symbol}{Style.RESET_ALL}")
                                failed += 1
                        
                elif rec['type'] == 'SELL':
                    # Check if we have the position
                    with self.db.get_connection() as conn:
                        cursor = conn.cursor()
                        
                        # Get portfolio ID
                        cursor.execute("SELECT id FROM portfolios WHERE name = ?", (portfolio_name,))
                        result = cursor.fetchone()
                        
                        if result:
                            portfolio_id = result[0]
                            
                            # Check holdings (simplified - just mark as sell recommendation)
                            print(f"{Fore.YELLOW}📝 SELL recommendation recorded for {symbol}{Style.RESET_ALL}")
                            
                            # Save as a signal/recommendation
                            cursor.execute("""
                                INSERT INTO signals 
                                (symbol, signal_type, signal_strength, reason, generated_date, status)
                                VALUES (?, 'SELL', ?, ?, datetime('now'), 'ACTIVE')
                            """, (symbol, rec['strength'], rec['reason']))
                            
                            conn.commit()
                            executed += 1
                            
            except Exception as e:
                print(f"{Fore.RED}Error executing trade for {rec['symbol']}: {e}{Style.RESET_ALL}")
                failed += 1
        
        print(f"\n{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}Execution Summary:{Style.RESET_ALL}")
        print(f"  ✓ Successful: {executed}")
        print(f"  ✗ Failed: {failed}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")


    
    def backtest_menu(self):
        """Handle backtest operations - COMPLETE VERSION"""
        while True:
            print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print(f"{Fore.CYAN} 📊 BACKTEST MENU{Style.RESET_ALL}")
            print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
            print("1. Run Single Backtest")
            print("2. Compare Multiple Strategies")
            print("3. Walk-Forward Analysis")
            print("4. Monte Carlo Simulation")
            print("5. View Backtest Results")
            print("6. Export Results to CSV")
            print("0. Back to Main Menu")
            
            choice = input(f"\n{Fore.GREEN}Enter choice: {Style.RESET_ALL}")
            
            if choice == '1':
                self.run_single_backtest()
            elif choice == '2':
                self.compare_strategies()
            elif choice == '3':
                self.walk_forward_analysis()
            elif choice == '4':
                self.monte_carlo_simulation()
            elif choice == '5':
                self.view_backtest_results()
            elif choice == '6':
                self.export_backtest_results()
            elif choice == '0':
                break
    
    def run_single_backtest(self):
        """Run a single backtest with current configuration"""
        try:
            print(f"\n{Fore.CYAN}=== SINGLE BACKTEST ==={Style.RESET_ALL}")
            
            # Get date range
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD): ").strip()
            initial_capital = float(input("Enter initial capital (default 100000): ").strip() or "100000")
            
            print(f"\n{Fore.YELLOW}Running backtest from {start_date} to {end_date}...{Style.RESET_ALL}")
            
            # Check if we have optimized configuration
            config = self.load_backtest_configuration()
            
            # Run backtest
            results = self.execute_backtest(start_date, end_date, initial_capital, config)
            
            # Display results
            self.display_backtest_results(results)
            
            # Save results
            save = input("\nSave results? (y/n): ").lower()
            if save == 'y':
                self.save_backtest_results(results)
                print(f"{Fore.GREEN}✓ Results saved{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error running backtest: {e}{Style.RESET_ALL}")
            logger.error(f"Backtest error: {e}")

    def save_backtest_results(self, results):
        """Save backtest results to database"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS backtest_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        start_date TEXT,
                        end_date TEXT,
                        initial_capital REAL,
                        final_value REAL,
                        total_return REAL,
                        total_trades INTEGER,
                        win_rate REAL,
                        sharpe_ratio REAL,
                        max_drawdown REAL,
                        config TEXT,
                        trades TEXT,
                        equity_curve TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Save results
                import json
                cursor.execute("""
                    INSERT INTO backtest_results 
                    (start_date, end_date, initial_capital, final_value, total_return,
                    total_trades, win_rate, sharpe_ratio, max_drawdown, config, trades, equity_curve)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    results['start_date'],
                    results['end_date'],
                    results['initial_capital'],
                    results['final_value'],
                    results['total_return'],
                    results['metrics']['total_trades'],
                    results['metrics']['win_rate'],
                    results['metrics']['sharpe_ratio'],
                    results['metrics']['max_drawdown'],
                    json.dumps(results.get('config', {})),
                    json.dumps(results.get('trades', [])),
                    json.dumps(results.get('equity_curve', []))
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error saving backtest results: {e}")

    def display_backtest_results(self, results):
        """Display backtest results"""
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN} 📊 BACKTEST RESULTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        print(f"\nPeriod: {results['start_date']} to {results['end_date']}")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Value: ${results['final_value']:,.2f}")
        
        total_return = results['total_return'] * 100
        color = Fore.GREEN if total_return > 0 else Fore.RED
        print(f"Total Return: {color}{total_return:+.2f}%{Style.RESET_ALL}")
        
        metrics = results['metrics']
        print(f"\n{Fore.YELLOW}Performance Metrics:{Style.RESET_ALL}")
        print(f"  Total Trades: {metrics['total_trades']}")
        print(f"  Win Rate: {metrics['win_rate']:.2%}")
        print(f"  Average Win: ${metrics['avg_win']:,.2f}")
        print(f"  Average Loss: ${metrics['avg_loss']:,.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

    def compare_strategies(self):
        """Compare multiple trading strategies"""
        try:
            print(f"\n{Fore.CYAN}=== COMPARE STRATEGIES ==={Style.RESET_ALL}")
            
            # Get date range
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD): ").strip()
            initial_capital = float(input("Enter initial capital (default 100000): ").strip() or "100000")
            
            # Define strategies to compare
            strategies = self.get_strategy_configurations()
            
            print(f"\n{Fore.YELLOW}Comparing {len(strategies)} strategies...{Style.RESET_ALL}")
            print("Strategies:")
            for name in strategies.keys():
                print(f"  • {name}")
            
            comparison_results = {}
            
            # Run backtest for each strategy
            for strategy_name, config in strategies.items():
                print(f"\nTesting {strategy_name}...", end="", flush=True)
                
                try:
                    results = self.execute_backtest(start_date, end_date, initial_capital, config)
                    comparison_results[strategy_name] = results
                    print(f" {Fore.GREEN}✓{Style.RESET_ALL}")
                except Exception as e:
                    print(f" {Fore.RED}✗ {e}{Style.RESET_ALL}")
                    comparison_results[strategy_name] = None
            
            # Display comparison
            self.display_strategy_comparison(comparison_results)
            
        except Exception as e:
            print(f"{Fore.RED}Error comparing strategies: {e}{Style.RESET_ALL}")
            logger.error(f"Strategy comparison error: {e}")

    def display_strategy_comparison(self, comparison_results):
        """Display strategy comparison results"""
        
        print(f"\n{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        print(f"{Fore.CYAN} 📊 STRATEGY COMPARISON{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*100}{Style.RESET_ALL}")
        
        print(f"\n{'Strategy':<20} {'Return':<12} {'Win Rate':<12} {'Sharpe':<10} {'Max DD':<12} {'Trades':<10}")
        print("-" * 100)
        
        for strategy_name, results in comparison_results.items():
            if results:
                metrics = results['metrics']
                return_pct = results['total_return'] * 100
                color = Fore.GREEN if return_pct > 0 else Fore.RED
                
                print(f"{strategy_name:<20} {color}{return_pct:+.2f}%{Style.RESET_ALL}{'':4} "
                    f"{metrics['win_rate']:.2%}{'':6} {metrics['sharpe_ratio']:>8.2f}  "
                    f"{metrics['max_drawdown']:>10.2%}  {metrics['total_trades']:>8}")
            else:
                print(f"{strategy_name:<20} {'N/A':<12} {'N/A':<12} {'N/A':<10} {'N/A':<12} {'N/A':<10}")

    def walk_forward_analysis(self):
        """Perform walk-forward analysis"""
        try:
            print(f"\n{Fore.CYAN}=== WALK-FORWARD ANALYSIS ==={Style.RESET_ALL}")
            
            # Get parameters
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD): ").strip()
            window_size = int(input("Enter window size in days (default 90): ").strip() or "90")
            step_size = int(input("Enter step size in days (default 30): ").strip() or "30")
            
            print(f"\n{Fore.YELLOW}Running walk-forward analysis...{Style.RESET_ALL}")
            print(f"Window: {window_size} days, Step: {step_size} days")
            
            # Generate windows
            windows = self.generate_walk_forward_windows(start_date, end_date, window_size, step_size)
            
            print(f"Generated {len(windows)} windows")
            
            results = []
            for i, (train_start, train_end, test_start, test_end) in enumerate(windows, 1):
                print(f"\nWindow {i}/{len(windows)}:")
                print(f"  Train: {train_start} to {train_end}")
                print(f"  Test: {test_start} to {test_end}")
                
                # Optimize on training period
                print("  Optimizing...", end="", flush=True)
                optimized_config = self.quick_optimize(train_start, train_end)
                print(f" {Fore.GREEN}✓{Style.RESET_ALL}")
                
                # Test on out-of-sample period
                print("  Testing...", end="", flush=True)
                test_results = self.execute_backtest(test_start, test_end, 100000, optimized_config)
                results.append({
                    'window': i,
                    'train_period': f"{train_start} to {train_end}",
                    'test_period': f"{test_start} to {test_end}",
                    'config': optimized_config,
                    'results': test_results
                })
                print(f" {Fore.GREEN}✓{Style.RESET_ALL}")
            
            # Display walk-forward results
            self.display_walk_forward_results(results)
            
        except Exception as e:
            print(f"{Fore.RED}Error in walk-forward analysis: {e}{Style.RESET_ALL}")
            logger.error(f"Walk-forward error: {e}")

    def display_walk_forward_results(self, results):
        """Display walk-forward analysis results"""
        
        print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        print(f"{Fore.CYAN} WALK-FORWARD ANALYSIS RESULTS{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
        
        print(f"\n{'Window':<8} {'Train Period':<25} {'Test Return':<12} {'Win Rate':<10} {'Sharpe':<10}")
        print("-" * 80)
        
        total_return = 1.0
        
        for r in results:
            if r['results']:
                test_return = r['results']['total_return']
                total_return *= (1 + test_return)
                metrics = r['results']['metrics']
                
                color = Fore.GREEN if test_return > 0 else Fore.RED
                
                print(f"{r['window']:<8} {r['train_period']:<25} "
                    f"{color}{test_return:>10.2%}{Style.RESET_ALL} "
                    f"{metrics['win_rate']:>9.2%} {metrics['sharpe_ratio']:>9.2f}")
        
        final_return = total_return - 1
        color = Fore.GREEN if final_return > 0 else Fore.RED
        
        print("-" * 80)
        print(f"Cumulative Return: {color}{final_return:.2%}{Style.RESET_ALL}")

    def quick_optimize(self, start_date, end_date):
        """Quick optimization for walk-forward analysis"""
        # Simplified optimization for speed
        best_config = None
        best_return = -float('inf')
        
        configs = self.get_strategy_configurations()
        
        for name, config in configs.items():
            try:
                results = self.execute_backtest(start_date, end_date, 100000, config)
                if results['total_return'] > best_return:
                    best_return = results['total_return']
                    best_config = config
            except:
                continue
        
        return best_config or self.load_backtest_configuration()


    def generate_walk_forward_windows(self, start_date, end_date, window_size, step_size):
        """Generate walk-forward analysis windows"""
        from datetime import datetime, timedelta
        
        windows = []
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')
        
        current_start = start
        
        while current_start + timedelta(days=window_size + step_size) <= end:
            train_start = current_start.strftime('%Y-%m-%d')
            train_end = (current_start + timedelta(days=window_size)).strftime('%Y-%m-%d')
            test_start = (current_start + timedelta(days=window_size + 1)).strftime('%Y-%m-%d')
            test_end = (current_start + timedelta(days=window_size + step_size)).strftime('%Y-%m-%d')
            
            windows.append((train_start, train_end, test_start, test_end))
            current_start += timedelta(days=step_size)
        
        return windows

    def monte_carlo_simulation(self):
        """Run Monte Carlo simulation"""
        try:
            print(f"\n{Fore.CYAN}=== MONTE CARLO SIMULATION ==={Style.RESET_ALL}")
            
            # Get parameters
            start_date = input("Enter start date (YYYY-MM-DD): ").strip()
            end_date = input("Enter end date (YYYY-MM-DD): ").strip()
            num_simulations = int(input("Number of simulations (default 100): ").strip() or "100")
            initial_capital = float(input("Initial capital (default 100000): ").strip() or "100000")
            
            print(f"\n{Fore.YELLOW}Running {num_simulations} Monte Carlo simulations...{Style.RESET_ALL}")
            
            # Get base configuration
            config = self.load_backtest_configuration()
            
            simulation_results = []
            
            for i in range(1, num_simulations + 1):
                if i % 10 == 0:
                    print(f"Progress: {i}/{num_simulations}")
                
                # Add randomness to configuration
                randomized_config = self.randomize_configuration(config)
                
                # Run backtest
                try:
                    results = self.execute_backtest(start_date, end_date, initial_capital, randomized_config)
                    simulation_results.append(results)
                except:
                    continue
            
            # Analyze Monte Carlo results
            self.analyze_monte_carlo_results(simulation_results)
            
        except Exception as e:
            print(f"{Fore.RED}Error in Monte Carlo simulation: {e}{Style.RESET_ALL}")
            logger.error(f"Monte Carlo error: {e}")

    def analyze_monte_carlo_results(self, results):
        """Analyze Monte Carlo simulation results"""
        if not results:
            print(f"{Fore.YELLOW}No valid simulation results{Style.RESET_ALL}")
            return
        
        returns = [r['total_return'] for r in results]
        
        print(f"\n{Fore.CYAN}Monte Carlo Analysis:{Style.RESET_ALL}")
        print(f"Simulations: {len(results)}")
        print(f"Average Return: {np.mean(returns):.2%}")
        print(f"Median Return: {np.median(returns):.2%}")
        print(f"Best Return: {np.max(returns):.2%}")
        print(f"Worst Return: {np.min(returns):.2%}")
        print(f"Standard Deviation: {np.std(returns):.2%}")
        
        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        print(f"\nReturn Distribution:")
        for p in percentiles:
            value = np.percentile(returns, p)
            print(f"  {p}th percentile: {value:.2%}")
        
        # Calculate probability of profit
        profitable = sum(1 for r in returns if r > 0)
        print(f"\nProbability of Profit: {profitable/len(returns):.2%}")

    def randomize_configuration(self, config):
        """Add random variation to configuration for Monte Carlo"""
        import random
        
        randomized = config.copy()
        
        # Add ±10% variation to numeric parameters
        for key, value in randomized.items():
            if isinstance(value, (int, float)):
                variation = random.uniform(0.9, 1.1)
                randomized[key] = value * variation
        
        return randomized

    def view_backtest_results(self):
        """View saved backtest results"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get recent backtest results
                cursor.execute("""
                    SELECT id, start_date, end_date, initial_capital, final_value, 
                        total_return, sharpe_ratio, max_drawdown, created_at
                    FROM backtest_results
                    ORDER BY created_at DESC
                    LIMIT 20
                """)
                
                results = cursor.fetchall()
                
                if results:
                    print(f"\n{Fore.CYAN}Recent Backtest Results:{Style.RESET_ALL}")
                    print(f"{'ID':<5} {'Period':<25} {'Initial':<12} {'Final':<12} {'Return':<10} {'Sharpe':<8} {'Max DD':<10}")
                    print("-" * 100)
                    
                    for row in results:
                        period = f"{row[1]} to {row[2]}"
                        return_pct = row[5] * 100 if row[5] else 0
                        color = Fore.GREEN if return_pct > 0 else Fore.RED
                        
                        print(f"{row[0]:<5} {period:<25} ${row[3]:>11,.0f} ${row[4]:>11,.0f} "
                            f"{color}{return_pct:>8.2f}%{Style.RESET_ALL} {row[6]:>8.2f} {row[7]:>9.2%}")
                else:
                    print(f"{Fore.YELLOW}No backtest results found{Style.RESET_ALL}")
                    
        except Exception as e:
            print(f"{Fore.RED}Error viewing results: {e}{Style.RESET_ALL}")

    def export_backtest_results(self):
        """Export backtest results to CSV"""
        try:
            filename = input("Enter filename (default: backtest_results.csv): ").strip() or "backtest_results.csv"
            
            with self.db.get_connection() as conn:
                df = pd.read_sql_query("""
                    SELECT * FROM backtest_results
                    ORDER BY created_at DESC
                """, conn)
                
                if not df.empty:
                    df.to_csv(filename, index=False)
                    print(f"{Fore.GREEN}✓ Results exported to {filename}{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}No results to export{Style.RESET_ALL}")
                    
        except Exception as e:
            print(f"{Fore.RED}Error exporting results: {e}{Style.RESET_ALL}")

    # Helper functions

    def load_backtest_configuration(self):
        """Load configuration for backtesting"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Try to get active optimized configuration
                cursor.execute("""
                    SELECT parameters FROM trading_configs
                    WHERE is_active = 1
                    ORDER BY created_at DESC
                    LIMIT 1
                """)
                
                result = cursor.fetchone()
                
                if result:
                    import json
                    return json.loads(result[0]) if isinstance(result[0], str) else result[0]
                    
        except:
            pass
        
        # Return default configuration
        return {
            'min_entry_score': 70.0,
            'min_confidence': 0.65,
            'max_positions': 10,
            'max_position_pct': 0.10,
            'stop_loss_pct': 0.05,
            'profit_target_pct': 0.10,
            'rsi_oversold': 30,
            'rsi_overbought': 70,
            'technical_weight': 0.4,
            'momentum_weight': 0.3,
            'volume_weight': 0.2,
            'sentiment_weight': 0.1
        }

    def get_strategy_configurations(self):
        """Get different strategy configurations for comparison"""
        return {
            'Conservative': {
                'min_entry_score': 80,
                'max_positions': 5,
                'stop_loss_pct': 0.03,
                'profit_target_pct': 0.06,
                'rsi_oversold': 25,
                'rsi_overbought': 75
            },
            'Moderate': {
                'min_entry_score': 70,
                'max_positions': 10,
                'stop_loss_pct': 0.05,
                'profit_target_pct': 0.10,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            },
            'Aggressive': {
                'min_entry_score': 60,
                'max_positions': 15,
                'stop_loss_pct': 0.07,
                'profit_target_pct': 0.15,
                'rsi_oversold': 35,
                'rsi_overbought': 65
            },
            'Momentum': {
                'min_entry_score': 65,
                'max_positions': 8,
                'stop_loss_pct': 0.04,
                'profit_target_pct': 0.12,
                'momentum_weight': 0.5,
                'technical_weight': 0.3
            }
        }


    def execute_backtest(self, start_date, end_date, initial_capital, config):
        """Execute a backtest with given configuration"""
        
        # Initialize backtest portfolio
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'value': initial_capital,
            'trades': [],
            'equity_curve': []
        }
        
        try:
            with self.db.get_connection() as conn:
                # Get all trading days
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT date 
                    FROM price_data 
                    WHERE date BETWEEN ? AND ?
                    ORDER BY date
                """, (start_date, end_date))
                
                trading_days = [row[0] for row in cursor.fetchall()]
                
                if not trading_days:
                    raise ValueError("No data available for the specified period")
                
                print(f"Backtesting {len(trading_days)} trading days...")
                
                for day in trading_days:
                    # Get signals for this day
                    signals = self.get_historical_signals(day, config)
                    
                    # Process sell signals first
                    for signal in [s for s in signals if s['type'] == 'SELL']:
                        if signal['symbol'] in portfolio['positions']:
                            self.execute_backtest_sell(portfolio, signal, day)
                    
                    # Process buy signals
                    for signal in [s for s in signals if s['type'] == 'BUY']:
                        if portfolio['cash'] > 0 and len(portfolio['positions']) < config.get('max_positions', 10):
                            self.execute_backtest_buy(portfolio, signal, day, config)
                    
                    # Update portfolio value
                    portfolio_value = self.calculate_portfolio_value(portfolio, day)
                    portfolio['equity_curve'].append({
                        'date': day,
                        'value': portfolio_value
                    })
                
                # Calculate metrics
                metrics = self.calculate_backtest_metrics(portfolio, initial_capital)
                
                return {
                    'start_date': start_date,
                    'end_date': end_date,
                    'initial_capital': initial_capital,
                    'final_value': portfolio['value'],
                    'total_return': (portfolio['value'] - initial_capital) / initial_capital,
                    'trades': portfolio['trades'],
                    'equity_curve': portfolio['equity_curve'],
                    'metrics': metrics,
                    'config': config
                }
                
        except Exception as e:
            logger.error(f"Backtest execution error: {e}")
            raise

    def calculate_backtest_metrics(self, portfolio, initial_capital):
        """Calculate backtest performance metrics"""
        
        if not portfolio['trades']:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'sharpe_ratio': 0
            }
        
        # Calculate trade statistics
        trades_df = pd.DataFrame(portfolio['trades'])
        
        sell_trades = trades_df[trades_df['type'] == 'SELL']
        winning_trades = sell_trades[sell_trades['profit'] > 0] if 'profit' in sell_trades else pd.DataFrame()
        losing_trades = sell_trades[sell_trades['profit'] < 0] if 'profit' in sell_trades else pd.DataFrame()
        
        # Calculate metrics
        metrics = {
            'total_trades': len(sell_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(sell_trades) if len(sell_trades) > 0 else 0,
            'avg_win': winning_trades['profit'].mean() if len(winning_trades) > 0 else 0,
            'avg_loss': losing_trades['profit'].mean() if len(losing_trades) > 0 else 0,
            'profit_factor': abs(winning_trades['profit'].sum() / losing_trades['profit'].sum()) if len(losing_trades) > 0 and losing_trades['profit'].sum() != 0 else 0,
        }
        
        # Calculate max drawdown
        equity_curve = [e['value'] for e in portfolio['equity_curve']]
        if equity_curve:
            peak = equity_curve[0]
            max_dd = 0
            for value in equity_curve:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            metrics['max_drawdown'] = max_dd
        else:
            metrics['max_drawdown'] = 0
        
        # Calculate Sharpe ratio (simplified)
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            metrics['sharpe_ratio'] = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            metrics['sharpe_ratio'] = 0
        
        return metrics

    def calculate_portfolio_value(self, portfolio, date):
        """Calculate total portfolio value"""
        
        total_value = portfolio['cash']
        
        try:
            with self.db.get_connection() as conn:
                for symbol, position in portfolio['positions'].items():
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT close FROM price_data 
                        WHERE symbol = ? AND date = ?
                    """, (symbol, date))
                    
                    result = cursor.fetchone()
                    if result:
                        current_price = result[0]
                        total_value += position['shares'] * current_price
                        
                        # Check stop loss and take profit
                        if current_price <= position['stop_loss'] or current_price >= position['take_profit']:
                            # Trigger automatic sell
                            self.execute_backtest_sell(portfolio, {
                                'symbol': symbol,
                                'price': current_price
                            }, date)
                            
        except Exception as e:
            logger.error(f"Error calculating portfolio value: {e}")
        
        portfolio['value'] = total_value
        return total_value

    def get_historical_signals(self, date, config):
        """Get signals for a specific historical date"""
        signals = []
        
        try:
            with self.db.get_connection() as conn:
                # Get all stocks with data on this date
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT DISTINCT symbol 
                    FROM price_data 
                    WHERE date = ?
                """, (date,))
                
                symbols = [row[0] for row in cursor.fetchall()]
                
                for symbol in symbols:
                    # Get price history up to this date
                    df = pd.read_sql_query("""
                        SELECT date, open, high, low, close, volume
                        FROM price_data
                        WHERE symbol = ? AND date <= ?
                        ORDER BY date DESC
                        LIMIT 100
                    """, conn, params=(symbol, date))
                    
                    if len(df) < 50:
                        continue
                    
                    # Reverse for chronological order
                    df = df.iloc[::-1].reset_index(drop=True)
                    
                    # Calculate indicators and generate signals
                    stock_signals = self.generate_backtest_signals(symbol, df, config)
                    signals.extend(stock_signals)
                    
        except Exception as e:
            logger.error(f"Error getting historical signals: {e}")
        
        return signals

    def generate_backtest_signals(self, symbol, df, config):
        """Generate signals for backtesting"""
        signals = []
        
        try:
            close = df['close'].values
            volume = df['volume'].values
            current_price = close[-1]
            
            # RSI
            rsi = self.calculate_rsi(close, 14)
            rsi_oversold = config.get('rsi_oversold', 30)
            rsi_overbought = config.get('rsi_overbought', 70)
            
            if rsi < rsi_oversold:
                signals.append({
                    'symbol': symbol,
                    'type': 'BUY',
                    'price': current_price,
                    'strength': (rsi_oversold - rsi) / rsi_oversold
                })
            elif rsi > rsi_overbought:
                signals.append({
                    'symbol': symbol,
                    'type': 'SELL',
                    'price': current_price,
                    'strength': (rsi - rsi_overbought) / (100 - rsi_overbought)
                })
            
            # Add more indicators as needed
            
        except Exception as e:
            logger.debug(f"Error generating signals for {symbol}: {e}")
        
        return signals

    def execute_backtest_buy(self, portfolio, signal, date, config):
        """Execute buy order in backtest"""
        
        position_size = min(
            portfolio['cash'] * config.get('max_position_pct', 0.1),
            portfolio['cash'] / (config.get('max_positions', 10) - len(portfolio['positions']))
        )
        
        if position_size > signal['price']:
            shares = int(position_size / signal['price'])
            cost = shares * signal['price']
            
            portfolio['cash'] -= cost
            portfolio['positions'][signal['symbol']] = {
                'shares': shares,
                'entry_price': signal['price'],
                'entry_date': date,
                'stop_loss': signal['price'] * (1 - config.get('stop_loss_pct', 0.05)),
                'take_profit': signal['price'] * (1 + config.get('profit_target_pct', 0.1))
            }
            
            portfolio['trades'].append({
                'date': date,
                'symbol': signal['symbol'],
                'type': 'BUY',
                'shares': shares,
                'price': signal['price'],
                'total': cost
            })

    def execute_backtest_sell(self, portfolio, signal, date):
        """Execute sell order in backtest"""
        
        position = portfolio['positions'][signal['symbol']]
        proceeds = position['shares'] * signal['price']
        profit = proceeds - (position['shares'] * position['entry_price'])
        
        portfolio['cash'] += proceeds
        
        portfolio['trades'].append({
            'date': date,
            'symbol': signal['symbol'],
            'type': 'SELL',
            'shares': position['shares'],
            'price': signal['price'],
            'total': proceeds,
            'profit': profit,
            'profit_pct': profit / (position['shares'] * position['entry_price'])
        })
        
        del portfolio['positions'][signal['symbol']]


    def optimizer_menu(self):
        """Handle optimizer operations"""
        while True:
            print(f"\n{Fore.CYAN}Optimizer Menu:{Style.RESET_ALL}")
            print("1. Genetic Algorithm Optimization")
            print("2. Grid Search")
            print("3. Bayesian Optimization")
            print("4. View Best Parameters")
            print("5. Apply Best Parameters")
            print("0. Back to Main Menu")
            
            choice = input(f"\n{Fore.GREEN}Enter choice: {Style.RESET_ALL}")
            
            if choice == '1':
                self.run_genetic_optimization()
            elif choice == '2':
                self.run_grid_search()
            elif choice == '3':
                self.run_bayesian_optimization()
            elif choice == '4':
                self.view_best_parameters()
            elif choice == '5':
                self.apply_best_parameters()
            elif choice == '0':
                break
    
    def run_genetic_optimization(self):
        """Run genetic algorithm optimization"""
        try:
            print(f"\n{Fore.CYAN}Genetic Algorithm Optimization:{Style.RESET_ALL}")
            
            # Configuration
            population_size = int(input("Population size (default 50): ") or "50")
            generations = int(input("Number of generations (default 100): ") or "100")
            start_date = input("Start date (YYYY-MM-DD): ")
            end_date = input("End date (YYYY-MM-DD): ")
            
            print(f"\n{Fore.YELLOW}Running optimization...{Style.RESET_ALL}")
            print(f"This may take several minutes...")
            
            # Run optimization
            best_params = self.optimizer.optimize(
                population_size=population_size,
                generations=generations,
                start_date=start_date,
                end_date=end_date
            )
            
            # Display results
            print(f"\n{Fore.GREEN}Optimization Complete!{Style.RESET_ALL}")
            print(f"{'='*60}")
            print(f"Best Parameters Found:")
            print(f"{'='*60}")
            
            for param, value in best_params.items():
                if param not in ['fitness', 'expected_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']:
                    print(f"{param}: {value}")
            
            print(f"{'='*60}")
            print(f"Expected Annual Return: {best_params.get('expected_return', 0):.2f}%")
            print(f"Sharpe Ratio: {best_params.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {best_params.get('max_drawdown', 0):.2f}%")
            
            # Save parameters
            if input("\nSave parameters? (y/n): ").lower() == 'y':
                self.optimizer.save_parameters(best_params)
                print(f"{Fore.GREEN}Parameters saved{Style.RESET_ALL}")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def run_grid_search(self):
        """Run grid search optimization"""
        try:
            print(f"\n{Fore.CYAN}Grid Search Optimization:{Style.RESET_ALL}")
            
            # Define parameter grid
            param_grid = {
                'sma_short': [10, 20, 30],
                'sma_long': [40, 50, 60],
                'rsi_period': [10, 14, 20],
                'stop_loss_percent': [0.01, 0.02, 0.03]
            }
            
            start_date = input("Start date (YYYY-MM-DD): ")
            end_date = input("End date (YYYY-MM-DD): ")
            
            optimizer = GridSearchOptimizer(self.db)
            best_params = optimizer.optimize(
                param_grid=param_grid,
                start_date=start_date,
                end_date=end_date
            )
            
            print(f"{Fore.GREEN}Best parameters found:{Style.RESET_ALL}")
            for key, value in best_params['parameters'].items():
                print(f"  {key}: {value}")
                
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

    def run_bayesian_optimization(self):
        """Run Bayesian optimization"""
        try:
            print(f"\n{Fore.CYAN}Bayesian Optimization:{Style.RESET_ALL}")
            
            bounds = {
                'sma_short': (5, 50),
                'sma_long': (20, 200),
                'rsi_period': (7, 21),
                'stop_loss_percent': (0.01, 0.05)
            }
            
            start_date = input("Start date (YYYY-MM-DD): ")
            end_date = input("End date (YYYY-MM-DD): ")
            n_iter = int(input("Number of iterations (default 50): ") or "50")
            
            optimizer = BayesianOptimizer(self.db)
            best_params = optimizer.optimize(
                bounds=bounds,
                n_iter=n_iter,
                start_date=start_date,
                end_date=end_date
            )
            
            print(f"{Fore.GREEN}Best parameters found:{Style.RESET_ALL}")
            print(f"Best score: {best_params['best_score']:.4f}")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

    def view_best_parameters(self):
        """View best optimization parameters"""
        try:
            best_params = self.db.get_best_parameters()
            
            if not best_params:
                print(f"{Fore.YELLOW}No optimization results found{Style.RESET_ALL}")
                return
            
            print(f"\n{Fore.GREEN}Best Parameters:{Style.RESET_ALL}")
            print(f"{'='*60}")
            
            params = best_params.get('parameters', {})
            for param, value in params.items():
                print(f"{param}: {value}")
            
            print(f"{'='*60}")
            print(f"Fitness Score: {best_params.get('fitness_score', 0):.4f}")
            print(f"Total Return: {best_params.get('total_return', 0):.2f}%")
            print(f"Sharpe Ratio: {best_params.get('sharpe_ratio', 0):.2f}")
            print(f"Max Drawdown: {best_params.get('max_drawdown', 0):.2f}%")
            
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def broker_menu(self):
        """Handle broker operations"""
        while True:
            print(f"\n{Fore.CYAN}Broker Connection:{Style.RESET_ALL}")
            
            if self.ib_connector and self.ib_connector.is_connected():
                print(f"{Fore.GREEN}✓ Connected to Interactive Brokers{Style.RESET_ALL}")
            else:
                print(f"{Fore.YELLOW}✗ Not connected{Style.RESET_ALL}")
            
            print("\n1. Connect to Interactive Brokers")
            print("2. Disconnect")
            print("3. Execute Order")
            print("4. View Open Orders")
            print("5. Sync Positions")
            print("6. Account Summary")
            print("0. Back to Main Menu")
            
            choice = input(f"\n{Fore.GREEN}Enter choice: {Style.RESET_ALL}")
            
            if choice == '1':
                self.connect_broker()
            elif choice == '2':
                self.disconnect_broker()
            elif choice == '3':
                self.execute_order()
            elif choice == '4':
                self.view_open_orders()
            elif choice == '5':
                self.sync_positions()
            elif choice == '6':
                self.account_summary()
            elif choice == '0':
                break
    
    def connect_broker(self):
        """Connect to Interactive Brokers"""
        try:
            print(f"{Fore.YELLOW}Connecting to Interactive Brokers...{Style.RESET_ALL}")
            
            host = input("TWS Host (default: 127.0.0.1): ") or "127.0.0.1"
            port = int(input("TWS Port (default: 7497): ") or "7497")
            client_id = int(input("Client ID (default: 1): ") or "1")
            
            self.ib_connector = IBConnector()
            if self.ib_connector.connect(host, port, client_id):
                print(f"{Fore.GREEN}Successfully connected to Interactive Brokers{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Failed to connect{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")
    
    def run(self):
        """Main loop"""
        print(f"\n{Fore.GREEN}Welcome to the Trading System!{Style.RESET_ALL}")
        print(f"Version 1.0.0 - Professional Trading Platform\n")
        
        while True:
            self.display_menu()
            choice = input(f"\n{Fore.GREEN}Enter your choice: {Style.RESET_ALL}")
            
            if choice.startswith('1'):
                self.import_menu()
            elif choice.startswith('2'):
                self.portfolio_menu()
            elif choice.startswith('3'):
                self.scanner_menu()
            elif choice.startswith('4'):
                self.backtest_menu()
            elif choice.startswith('5'):
                self.optimizer_menu()
            elif choice.startswith('6'):
                self.broker_menu()
            elif choice.startswith('7'):
                self.settings_menu()
            elif choice == '0':
                print(f"\n{Fore.YELLOW}Thank you for using the Trading System!{Style.RESET_ALL}")
                print(f"{Fore.GREEN}Good luck with your trading!{Style.RESET_ALL}")
                break
            else:
                print(f"{Fore.RED}Invalid choice. Please try again.{Style.RESET_ALL}")
    
    def settings_menu(self):
        """Settings and configuration menu"""
        while True:
            print(f"\n{Fore.CYAN}Settings Menu:{Style.RESET_ALL}")
            print("1. Configure Trading Rules")
            print("2. Risk Management Settings")
            print("3. Database Maintenance")
            print("4. Performance Optimization")
            print("5. Export/Import Settings")
            print("0. Back to Main Menu")
            
            choice = input(f"\n{Fore.GREEN}Enter choice: {Style.RESET_ALL}")
            
            if choice == '1':
                self.configure_trading_rules()
            elif choice == '2':
                self.risk_management_settings()
            elif choice == '3':
                self.database_maintenance()
            elif choice == '4':
                self.performance_optimization()
            elif choice == '5':
                self.export_import_settings()
            elif choice == '0':
                break
    
    def configure_trading_rules(self):
        """Configure trading rules"""
        print(f"\n{Fore.CYAN}Trading Rules Configuration:{Style.RESET_ALL}")
        print("Current configuration saved in config.json")
        print("Edit the file to modify trading rules")
    
    def risk_management_settings(self):
        """Configure risk management"""
        print(f"\n{Fore.CYAN}Risk Management Settings:{Style.RESET_ALL}")
        print(f"Stop Loss: {self.config.trading_rules.stop_loss_percent*100:.1f}%")
        print(f"Take Profit: {self.config.trading_rules.take_profit_percent*100:.1f}%")
        print(f"Max Position Size: {self.config.trading_rules.max_position_size*100:.1f}%")
        print(f"Max Drawdown: {self.config.trading_rules.max_drawdown*100:.1f}%")
    
    def database_maintenance(self):
        """Database maintenance"""
        try:
            print(f"\n{Fore.CYAN}Database Maintenance:{Style.RESET_ALL}")
            print("1. Clean old data")
            print("2. Optimize database")
            print("3. Backup database")
            
            choice = input("Enter choice: ")
            
            if choice == '1':
                days = int(input("Delete data older than (days): ") or "365")
                self.db.cleanup_old_data(days)
                print(f"{Fore.GREEN}Old data cleaned{Style.RESET_ALL}")
            elif choice == '2':
                with self.db.get_connection() as conn:
                    conn.execute("VACUUM")
                print(f"{Fore.GREEN}Database optimized{Style.RESET_ALL}")
            elif choice == '3':
                import shutil
                shutil.copy2(self.db.db_path, f"{self.db.db_path}.backup")
                print(f"{Fore.GREEN}Database backed up{Style.RESET_ALL}")
                
        except Exception as e:
            print(f"{Fore.RED}Error: {str(e)}{Style.RESET_ALL}")

def main():
    """Entry point"""
    try:
        system = TradingSystem()
        system.run()
    except KeyboardInterrupt:
        print(f"\n\n{Fore.YELLOW}System interrupted by user{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        print(f"\n{Fore.RED}Fatal error: {str(e)}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main()