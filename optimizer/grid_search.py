"""
Grid Search Optimizer
Systematic parameter optimization
"""

import itertools
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from datetime import datetime

from backtest.engine import BacktestEngine
from config import TradingRules

logger = logging.getLogger(__name__)

class GridSearchOptimizer:
    """Grid search parameter optimization"""
    
    def __init__(self, db):
        self.db = db
        self.backtest_engine = BacktestEngine(db)
        
    def optimize(self, param_grid: Dict[str, List[Any]],
                start_date: str, end_date: str,
                initial_capital: float = 100000,
                objective: str = 'sharpe_ratio',
                n_jobs: int = 4) -> Dict:
        """Run grid search optimization"""
        
        logger.info(f"Starting grid search optimization")
        
        # Generate all parameter combinations
        param_combinations = self._generate_combinations(param_grid)
        total_combinations = len(param_combinations)
        
        logger.info(f"Testing {total_combinations} parameter combinations")
        
        # Run backtests in parallel
        results = []
        
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            futures = []
            
            for i, params in enumerate(param_combinations):
                future = executor.submit(
                    self._evaluate_parameters,
                    params, start_date, end_date, 
                    initial_capital, objective
                )
                futures.append((future, params, i))
            
            # Collect results
            for future, params, idx in futures:
                try:
                    result = future.result(timeout=60)
                    results.append({
                        'parameters': params,
                        'objective_value': result[objective],
                        **result
                    })
                    
                    if (idx + 1) % 10 == 0:
                        logger.info(f"Progress: {idx + 1}/{total_combinations}")
                        
                except Exception as e:
                    logger.error(f"Error evaluating parameters: {e}")
        
        # Find best parameters
        best_result = max(results, key=lambda x: x['objective_value'])
        
        # Save results
        self._save_results(results, best_result, objective)
        
        logger.info(f"Optimization complete. Best {objective}: {best_result['objective_value']:.4f}")
        
        return best_result
    
    def _generate_combinations(self, param_grid: Dict[str, List[Any]]) -> List[Dict]:
        """Generate all parameter combinations"""
        
        keys = param_grid.keys()
        values = param_grid.values()
        
        combinations = []
        for combination in itertools.product(*values):
            param_dict = dict(zip(keys, combination))
            
            # Validate combination
            if self._validate_combination(param_dict):
                combinations.append(param_dict)
        
        return combinations
    
    def _validate_combination(self, params: Dict) -> bool:
        """Validate parameter combination"""
        
        # Check SMA constraint
        if 'sma_short' in params and 'sma_long' in params:
            if params['sma_short'] >= params['sma_long']:
                return False
        
        # Check RSI constraint
        if 'rsi_oversold' in params and 'rsi_overbought' in params:
            if params['rsi_oversold'] >= params['rsi_overbought']:
                return False
        
        # Check risk constraint
        if 'stop_loss_percent' in params and 'take_profit_percent' in params:
            if params['stop_loss_percent'] >= params['take_profit_percent']:
                return False
        
        return True
    
    def _evaluate_parameters(self, params: Dict, start_date: str,
                           end_date: str, initial_capital: float,
                           objective: str) -> Dict:
        """Evaluate a single parameter set"""
        
        try:
            # Create strategy with parameters
            strategy = TradingRules()
            for key, value in params.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
            
            # Run backtest
            result = self.backtest_engine.run(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                strategy=strategy
            )
            
            # Return metrics
            return {
                'total_return': result.total_return,
                'annual_return': result.annual_return,
                'sharpe_ratio': result.sharpe_ratio,
                'max_drawdown': result.max_drawdown,
                'win_rate': result.win_rate,
                'profit_factor': result.profit_factor,
                'total_trades': result.total_trades
            }
            
        except Exception as e:
            logger.error(f"Error in backtest: {e}")
            return {objective: -999}
    
    def _save_results(self, results: List[Dict], best_result: Dict, objective: str):
        """Save optimization results"""
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Save optimization run
            cursor.execute('''
                INSERT INTO optimization_runs 
                (optimization_id, optimization_type, strategy_name, 
                 start_date, end_date, objective, created_date,
                 total_iterations, best_fitness)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                f"grid_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'grid',
                'GridSearch',
                datetime.now().date(),
                datetime.now().date(),
                objective,
                datetime.now(),
                len(results),
                best_result['objective_value']
            ))
            
            optimization_id = cursor.lastrowid
            
            # Save individual results
            for i, result in enumerate(results):
                cursor.execute('''
                    INSERT INTO optimization_results 
                    (optimization_id, iteration, parameters, fitness_score,
                     total_return, sharpe_ratio, max_drawdown)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    optimization_id,
                    i,
                    json.dumps(result['parameters']),
                    result.get('objective_value', 0),
                    result.get('total_return', 0),
                    result.get('sharpe_ratio', 0),
                    result.get('max_drawdown', 0)
                ))
            
            conn.commit()
    
    def run_walk_forward(self, param_grid: Dict[str, List[Any]],
                        start_date: str, end_date: str,
                        in_sample_ratio: float = 0.7,
                        window_size: int = 252) -> Dict:
        """Run walk-forward optimization"""
        
        logger.info("Starting walk-forward optimization")
        
        # Split data into windows
        windows = self._create_walk_forward_windows(
            start_date, end_date, in_sample_ratio, window_size
        )
        
        results = []
        
        for window in windows:
            # Optimize on in-sample
            in_sample_best = self.optimize(
                param_grid,
                window['in_sample_start'],
                window['in_sample_end'],
                objective='sharpe_ratio'
            )
            
            # Test on out-of-sample
            out_sample_result = self._evaluate_parameters(
                in_sample_best['parameters'],
                window['out_sample_start'],
                window['out_sample_end'],
                100000,
                'sharpe_ratio'
            )
            
            results.append({
                'window': window,
                'in_sample_performance': in_sample_best,
                'out_sample_performance': out_sample_result
            })
        
        # Calculate overall performance
        overall = self._calculate_walk_forward_performance(results)
        
        return overall
    
    def _create_walk_forward_windows(self, start_date: str, end_date: str,
                                    in_sample_ratio: float,
                                    window_size: int) -> List[Dict]:
        """Create walk-forward windows"""
        
        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)
        
        windows = []
        current_start = start
        
        while current_start < end:
            window_end = min(current_start + pd.Timedelta(days=window_size), end)
            
            # Split into in-sample and out-of-sample
            split_point = current_start + pd.Timedelta(
                days=int((window_end - current_start).days * in_sample_ratio)
            )
            
            windows.append({
                'in_sample_start': current_start.strftime('%Y-%m-%d'),
                'in_sample_end': split_point.strftime('%Y-%m-%d'),
                'out_sample_start': split_point.strftime('%Y-%m-%d'),
                'out_sample_end': window_end.strftime('%Y-%m-%d')
            })
            
            # Move to next window
            current_start = split_point
        
        return windows
    
    def _calculate_walk_forward_performance(self, results: List[Dict]) -> Dict:
        """Calculate walk-forward performance"""
        
        in_sample_returns = []
        out_sample_returns = []
        
        for result in results:
            in_sample_returns.append(
                result['in_sample_performance'].get('total_return', 0)
            )
            out_sample_returns.append(
                result['out_sample_performance'].get('total_return', 0)
            )
        
        return {
            'num_windows': len(results),
            'avg_in_sample_return': np.mean(in_sample_returns),
            'avg_out_sample_return': np.mean(out_sample_returns),
            'consistency': np.corrcoef(in_sample_returns, out_sample_returns)[0, 1],
            'robustness_ratio': (np.mean(out_sample_returns) / 
                               np.mean(in_sample_returns)) if np.mean(in_sample_returns) != 0 else 0,
            'detailed_results': results
        }