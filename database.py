"""
Database Management Module
Handles all database operations with optimized performance
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
import logging
from contextlib import contextmanager
import json
import threading
from queue import Queue

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Centralized database management"""
    
    def __init__(self, db_path: str = "trading_system.db"):
        self.db_path = db_path
        self._local = threading.local()
        self.connection_pool = Queue(maxsize=10)
        self._initialize_pool()
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        for _ in range(5):
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            self.connection_pool.put(conn)
    
    @contextmanager
    def get_connection(self):
        """Get connection from pool"""
        conn = self.connection_pool.get()
        try:
            yield conn
        finally:
            self.connection_pool.put(conn)
    
    def initialize(self):
        """Initialize database schema"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Stocks table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS stocks (
                    symbol TEXT PRIMARY KEY,
                    name TEXT,
                    sector TEXT,
                    industry TEXT,
                    market_cap REAL,
                    shares_outstanding REAL,
                    country TEXT,
                    ipo_date DATE,
                    last_updated TIMESTAMP,
                    metadata TEXT
                )
            ''')
            
            # Price data table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS price_data (
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
            
            # Technical indicators table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS technical_indicators (
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
            
            # Portfolio table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS portfolios (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE,
                    created_date TIMESTAMP,
                    initial_capital REAL,
                    current_value REAL,
                    cash_balance REAL,
                    status TEXT,
                    metadata TEXT
                )
            ''')
            
            # Positions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS positions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER,
                    symbol TEXT,
                    shares REAL,
                    average_price REAL,
                    current_price REAL,
                    market_value REAL,
                    unrealized_pnl REAL,
                    realized_pnl REAL,
                    opened_date TIMESTAMP,
                    closed_date TIMESTAMP,
                    status TEXT,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
                )
            ''')
            
            # Transactions table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS transactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    portfolio_id INTEGER,
                    symbol TEXT,
                    transaction_type TEXT,
                    shares REAL,
                    price REAL,
                    commission REAL,
                    transaction_date TIMESTAMP,
                    notes TEXT,
                    FOREIGN KEY (portfolio_id) REFERENCES portfolios(id),
                    FOREIGN KEY (symbol) REFERENCES stocks(symbol)
                )
            ''')
            
            # Signals table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS signals (
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
            
            # Backtest results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS backtest_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    strategy_name TEXT,
                    start_date DATE,
                    end_date DATE,
                    initial_capital REAL,
                    final_value REAL,
                    total_return REAL,
                    annual_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    win_rate REAL,
                    total_trades INTEGER,
                    parameters TEXT,
                    equity_curve TEXT,
                    created_date TIMESTAMP
                )
            ''')
            
            # Optimization results table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS optimization_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    optimization_type TEXT,
                    parameters TEXT,
                    fitness_score REAL,
                    total_return REAL,
                    sharpe_ratio REAL,
                    max_drawdown REAL,
                    created_date TIMESTAMP
                )
            ''')
            
            # Create indexes for performance
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_price_symbol_date ON price_data(symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_indicators_symbol_date ON technical_indicators(symbol, date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_signals_symbol_date ON signals(symbol, generated_date)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_positions_portfolio ON positions(portfolio_id, status)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_transactions_portfolio ON transactions(portfolio_id, transaction_date)')
            
            conn.commit()
            logger.info("Database initialized successfully")
    
    def get_stock_info(self, symbol: str) -> Optional[Dict]:
        """Get stock information"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM stocks WHERE symbol = ?', (symbol,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """Get latest price for a stock"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT close FROM price_data 
                WHERE symbol = ? 
                ORDER BY date DESC 
                LIMIT 1
            ''', (symbol,))
            
            result = cursor.fetchone()
            return result[0] if result else None
    
    def get_price_history(self, symbol: str, start_date: str = None, 
                         end_date: str = None) -> pd.DataFrame:
        """Get price history for a stock"""
        query = 'SELECT * FROM price_data WHERE symbol = ?'
        params = [symbol]
        
        if start_date:
            query += ' AND date >= ?'
            params.append(start_date)
        
        if end_date:
            query += ' AND date <= ?'
            params.append(end_date)
        
        query += ' ORDER BY date'
        
        with self.get_connection() as conn:
            df = pd.read_sql_query(query, conn, params=params)
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
        return df
    
    def insert_price_data(self, symbol: str, data: pd.DataFrame):
        """Insert price data"""
        with self.get_connection() as conn:
            data['symbol'] = symbol
            data.to_sql('price_data', conn, if_exists='append', index=True)
            conn.commit()
    
    def get_portfolio(self, name: str = "default") -> Optional[Dict]:
        """Get portfolio by name"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT * FROM portfolios WHERE name = ?', (name,))
            row = cursor.fetchone()
            
            if row:
                return dict(row)
            return None
    
    def create_portfolio(self, name: str, initial_capital: float) -> int:
        """Create new portfolio"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO portfolios (name, created_date, initial_capital, 
                                      current_value, cash_balance, status)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (name, datetime.now(), initial_capital, initial_capital, 
                 initial_capital, 'active'))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_positions(self, portfolio_id: int, status: str = 'open') -> List[Dict]:
        """Get positions for portfolio"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM positions 
                WHERE portfolio_id = ? AND status = ?
            ''', (portfolio_id, status))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def execute_transaction(self, portfolio_id: int, symbol: str, 
                           transaction_type: str, shares: float, price: float,
                           commission: float = 0.0, notes: str = ""):
        """Execute a transaction"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Insert transaction
            cursor.execute('''
                INSERT INTO transactions (portfolio_id, symbol, transaction_type,
                                        shares, price, commission, transaction_date, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (portfolio_id, symbol, transaction_type, shares, price, 
                 commission, datetime.now(), notes))
            
            # Update position
            if transaction_type == 'BUY':
                self._update_position_buy(cursor, portfolio_id, symbol, shares, price)
            elif transaction_type == 'SELL':
                self._update_position_sell(cursor, portfolio_id, symbol, shares, price)
            
            # Update portfolio cash
            total_cost = shares * price + commission
            cash_change = -total_cost if transaction_type == 'BUY' else total_cost - commission
            
            cursor.execute('''
                UPDATE portfolios 
                SET cash_balance = cash_balance + ?
                WHERE id = ?
            ''', (cash_change, portfolio_id))
            
            conn.commit()
    
    def _update_position_buy(self, cursor, portfolio_id: int, symbol: str, 
                            shares: float, price: float):
        """Update position for buy transaction"""
        # Check if position exists
        cursor.execute('''
            SELECT * FROM positions 
            WHERE portfolio_id = ? AND symbol = ? AND status = 'open'
        ''', (portfolio_id, symbol))
        
        position = cursor.fetchone()
        
        if position:
            # Update existing position
            new_shares = position['shares'] + shares
            new_avg_price = ((position['shares'] * position['average_price']) + 
                           (shares * price)) / new_shares
            
            cursor.execute('''
                UPDATE positions 
                SET shares = ?, average_price = ?
                WHERE id = ?
            ''', (new_shares, new_avg_price, position['id']))
        else:
            # Create new position
            cursor.execute('''
                INSERT INTO positions (portfolio_id, symbol, shares, average_price,
                                     current_price, market_value, unrealized_pnl,
                                     realized_pnl, opened_date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (portfolio_id, symbol, shares, price, price, shares * price,
                 0, 0, datetime.now(), 'open'))
    
    def _update_position_sell(self, cursor, portfolio_id: int, symbol: str,
                             shares: float, price: float):
        """Update position for sell transaction"""
        cursor.execute('''
            SELECT * FROM positions 
            WHERE portfolio_id = ? AND symbol = ? AND status = 'open'
        ''', (portfolio_id, symbol))
        
        position = cursor.fetchone()
        
        if position:
            new_shares = position['shares'] - shares
            realized_pnl = shares * (price - position['average_price'])
            
            if new_shares <= 0:
                # Close position
                cursor.execute('''
                    UPDATE positions 
                    SET shares = 0, realized_pnl = realized_pnl + ?,
                        closed_date = ?, status = 'closed'
                    WHERE id = ?
                ''', (realized_pnl, datetime.now(), position['id']))
            else:
                # Partial sell
                cursor.execute('''
                    UPDATE positions 
                    SET shares = ?, realized_pnl = realized_pnl + ?
                    WHERE id = ?
                ''', (new_shares, realized_pnl, position['id']))
    
    def save_signal(self, symbol: str, signal_type: str, signal_strength: float,
                   reason: str, expiry_hours: int = 24):
        """Save trading signal"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO signals (symbol, signal_type, signal_strength, reason,
                                   generated_date, expiry_date, status)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (symbol, signal_type, signal_strength, reason,
                 datetime.now(), datetime.now() + timedelta(hours=expiry_hours),
                 'active'))
            
            conn.commit()
    
    def get_active_signals(self) -> List[Dict]:
        """Get active trading signals"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM signals 
                WHERE status = 'active' AND expiry_date > ?
                ORDER BY signal_strength DESC
            ''', (datetime.now(),))
            
            return [dict(row) for row in cursor.fetchall()]
    
    def save_backtest_results(self, results: Dict):
        """Save backtest results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Handle both dict and BacktestResult object
            if hasattr(results, '__dict__'):
                results_dict = results.__dict__
            else:
                results_dict = results
            
            cursor.execute('''
                INSERT INTO backtest_results 
                (strategy_name, start_date, end_date, initial_capital, final_value,
                 total_return, annual_return, sharpe_ratio, max_drawdown, win_rate,
                 total_trades, parameters, equity_curve, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                results_dict.get('strategy_name', 'default'),
                results_dict.get('start_date'),
                results_dict.get('end_date'),
                results_dict.get('initial_capital'),
                results_dict.get('final_value'),
                results_dict.get('total_return'),
                results_dict.get('annual_return'),
                results_dict.get('sharpe_ratio'),
                results_dict.get('max_drawdown'),
                results_dict.get('win_rate'),
                results_dict.get('total_trades'),
                json.dumps(results_dict.get('parameters', {})),
                json.dumps(results_dict.get('equity_curve', [])),
                datetime.now()
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def save_optimization_results(self, results: Dict):
        """Save optimization results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO optimization_results 
                (optimization_type, parameters, fitness_score, total_return,
                 sharpe_ratio, max_drawdown, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                results.get('optimization_type', 'genetic_algorithm'),
                json.dumps(results.get('parameters', {})),
                results.get('fitness_score', 0),
                results.get('total_return', 0),
                results.get('sharpe_ratio', 0),
                results.get('max_drawdown', 0),
                datetime.now()
            ))
            
            conn.commit()
            return cursor.lastrowid
    
    def get_best_parameters(self) -> Optional[Dict]:
        """Get best optimization parameters"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM optimization_results 
                ORDER BY fitness_score DESC 
                LIMIT 1
            ''')
            
            row = cursor.fetchone()
            if row:
                result = dict(row)
                result['parameters'] = json.loads(result['parameters'])
                return result
            return None
    
    def get_backtest_results(self, limit: int = 10) -> List[Dict]:
        """Get recent backtest results"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                SELECT * FROM backtest_results 
                ORDER BY created_date DESC 
                LIMIT ?
            ''', (limit,))
            
            results = []
            for row in cursor.fetchall():
                result = dict(row)
                result['parameters'] = json.loads(result.get('parameters', '{}'))
                result['equity_curve'] = json.loads(result.get('equity_curve', '[]'))
                results.append(result)
            
            return results
    
    def cleanup_old_data(self, days: int = 365):
        """Clean up old data"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            cutoff_date = datetime.now() - timedelta(days=days)
            
            # Clean old signals
            cursor.execute('''
                DELETE FROM signals 
                WHERE generated_date < ? AND status != 'active'
            ''', (cutoff_date,))
            
            # Clean old closed positions (keep recent ones for analysis)
            cursor.execute('''
                DELETE FROM positions 
                WHERE closed_date < ? AND status = 'closed'
                AND portfolio_id NOT IN (
                    SELECT id FROM portfolios WHERE status = 'active'
                )
            ''', (cutoff_date,))
            
            # Clean old backtest results (keep best ones)
            cursor.execute('''
                DELETE FROM backtest_results 
                WHERE created_date < ? 
                AND id NOT IN (
                    SELECT id FROM backtest_results 
                    ORDER BY sharpe_ratio DESC 
                    LIMIT 100
                )
            ''', (cutoff_date,))
            
            conn.commit()
            logger.info(f"Cleaned up data older than {days} days")
    
    def get_all_symbols(self) -> List[str]:
        """Get all stock symbols in database"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT symbol FROM stocks ORDER BY symbol')
            return [row[0] for row in cursor.fetchall()]
    
    def update_portfolio_value(self, portfolio_id: int):
        """Update current portfolio value"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Get cash balance
            cursor.execute('SELECT cash_balance FROM portfolios WHERE id = ?', (portfolio_id,))
            cash = cursor.fetchone()[0]
            
            # Get positions value
            cursor.execute('''
                SELECT SUM(shares * current_price) 
                FROM positions 
                WHERE portfolio_id = ? AND status = 'open'
            ''', (portfolio_id,))
            
            positions_value = cursor.fetchone()[0] or 0
            total_value = cash + positions_value
            
            # Update portfolio
            cursor.execute('''
                UPDATE portfolios 
                SET current_value = ? 
                WHERE id = ?
            ''', (total_value, portfolio_id))
            
            conn.commit()
    
    def get_portfolio_history(self, portfolio_id: int, days: int = 30) -> pd.DataFrame:
        """Get portfolio value history"""
        with self.get_connection() as conn:
            query = '''
                SELECT DATE(transaction_date) as date, 
                       SUM(CASE WHEN transaction_type = 'BUY' THEN -shares * price 
                                WHEN transaction_type = 'SELL' THEN shares * price 
                                ELSE 0 END) as cash_flow
                FROM transactions 
                WHERE portfolio_id = ? 
                AND transaction_date >= date('now', '-{} days')
                GROUP BY DATE(transaction_date)
                ORDER BY date
            '''.format(days)
            
            df = pd.read_sql_query(query, conn, params=[portfolio_id])
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df.set_index('date', inplace=True)
            
            return df