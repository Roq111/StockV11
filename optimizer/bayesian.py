"""
Bayesian Optimizer
Smart parameter optimization using Bayesian methods
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import logging
from datetime import datetime
import json
from scipy.stats import norm
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

class BayesianOptimizer:
    """Bayesian optimization for trading strategies"""
    
    def __init__(self, db):
        self.db = db
        self.observations = []
        self.best_params = None
        self.best_score = -np.inf
        
    def optimize(self, bounds: Dict[str, Tuple[float, float]],
                n_iter: int = 50,
                start_date: str = None,
                end_date: str = None,
                initial_capital: float = 100000,
                objective: str = 'sharpe_ratio',
                init_points: int = 5,
                acquisition: str = 'ei') -> Dict:
        """Run Bayesian optimization"""
        
        logger.info(f"Starting Bayesian optimization with {n_iter} iterations")
        
        # Initialize with random points
        self._initialize_random_points(
            bounds, init_points, start_date, end_date, 
            initial_capital, objective
        )
        
        # Bayesian optimization loop
        for i in range(n_iter - init_points):
            # Find next point to evaluate
            next_params = self._get_next_point(bounds, acquisition)
            
            # Evaluate the point
            score = self._evaluate_objective(
                next_params, start_date, end_date, 
                initial_capital, objective
            )
            
            # Update observations
            self.observations.append({
                'params': next_params,
                'score': score
            })
            
            # Update best if needed
            if score > self.best_score:
                self.best_score = score
                self.best_params = next_params
                logger.info(f"New best score: {score:.4f}")
            
            if (i + 1) % 10 == 0:
                logger.info(f"Progress: {i + 1 + init_points}/{n_iter}")
        
        # Final results
        final_results = {
            'best_params': self.best_params,
            'best_score': self.best_score,
            'n_iterations': n_iter,
            'objective': objective,
            'all_observations': self.observations
        }
        
        # Save results
        self._save_results(final_results, objective)
        
        return final_results
    
    def _initialize_random_points(self, bounds: Dict[str, Tuple[float, float]],
                                 n_points: int, start_date: str,
                                 end_date: str, initial_capital: float,
                                 objective: str):
        """Initialize with random points"""
        
        for _ in range(n_points):
            # Generate random parameters
            params = {}
            for param, (low, high) in bounds.items():
                if isinstance(low, int) and isinstance(high, int):
                    params[param] = np.random.randint(low, high + 1)
                else:
                    params[param] = np.random.uniform(low, high)
            
            # Evaluate
            score = self._evaluate_objective(
                params, start_date, end_date, 
                initial_capital, objective
            )
            
            # Store observation
            self.observations.append({
                'params': params,
                'score': score
            })
            
            # Update best
            if score > self.best_score:
                self.best_score = score
                self.best_params = params
    
    def _get_next_point(self, bounds: Dict[str, Tuple[float, float]],
                       acquisition: str = 'ei') -> Dict:
        """Get next point to evaluate using acquisition function"""
        
        # Build Gaussian Process model
        X = np.array([self._params_to_array(obs['params'], bounds) 
                     for obs in self.observations])
        y = np.array([obs['score'] for obs in self.observations])
        
        # Fit simple GP (simplified - in production use scikit-learn GaussianProcessRegressor)
        mean = np.mean(y)
        std = np.std(y) if len(y) > 1 else 1.0
        
        # Optimize acquisition function
        def acq_func(x):
            # Expected Improvement
            if acquisition == 'ei':
                return -self._expected_improvement(x, X, y, mean, std)
            # Upper Confidence Bound
            elif acquisition == 'ucb':
                return -self._upper_confidence_bound(x, X, y, mean, std)
            else:
                return -self._probability_of_improvement(x, X, y, mean, std)
        
        # Random restarts for optimization
        best_x = None
        best_acq = np.inf
        
        for _ in range(10):
            # Random starting point
            x0 = np.random.uniform(0, 1, len(bounds))
            
            # Optimize
            result = minimize(
                acq_func,
                x0,
                bounds=[(0, 1)] * len(bounds),
                method='L-BFGS-B'
            )
            
            if result.fun < best_acq:
                best_acq = result.fun
                best_x = result.x
        
        # Convert back to parameters
        next_params = self._array_to_params(best_x, bounds)
        
        return next_params
    
    def _expected_improvement(self, x: np.ndarray, X: np.ndarray,
                            y: np.ndarray, mean: float, std: float) -> float:
        """Calculate expected improvement"""
        
        # Predict at point (simplified)
        distances = np.sum((X - x) ** 2, axis=1)
        weights = np.exp(-distances)
        weights /= weights.sum() if weights.sum() > 0 else 1
        
        mu = np.dot(weights, y)
        sigma = std * (1 - weights.max())  # Simplified uncertainty
        
        # Calculate EI
        if sigma > 0:
            z = (mu - self.best_score) / sigma
            ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
        else:
            ei = 0
        
        return ei
    
    def _upper_confidence_bound(self, x: np.ndarray, X: np.ndarray,
                               y: np.ndarray, mean: float, std: float,
                               kappa: float = 2.0) -> float:
        """Calculate upper confidence bound"""
        
        # Predict at point
        distances = np.sum((X - x) ** 2, axis=1)
        weights = np.exp(-distances)
        weights /= weights.sum() if weights.sum() > 0 else 1
        
        mu = np.dot(weights, y)
        sigma = std * (1 - weights.max())
        
        return mu + kappa * sigma
    
    def _probability_of_improvement(self, x: np.ndarray, X: np.ndarray,
                                   y: np.ndarray, mean: float, std: float) -> float:
        """Calculate probability of improvement"""
        
        # Predict at point
        distances = np.sum((X - x) ** 2, axis=1)
        weights = np.exp(-distances)
        weights /= weights.sum() if weights.sum() > 0 else 1
        
        mu = np.dot(weights, y)
        sigma = std * (1 - weights.max())
        
        # Calculate POI
        if sigma > 0:
            z = (mu - self.best_score) / sigma
            poi = norm.cdf(z)
        else:
            poi = 0
        
        return poi
    
    def _params_to_array(self, params: Dict, bounds: Dict) -> np.ndarray:
        """Convert parameters to normalized array"""
        
        array = []
        for key, (low, high) in bounds.items():
            value = params[key]
            # Normalize to [0, 1]
            normalized = (value - low) / (high - low) if high > low else 0.5
            array.append(normalized)
        
        return np.array(array)
    
    def _array_to_params(self, array: np.ndarray, bounds: Dict) -> Dict:
        """Convert normalized array to parameters"""
        
        params = {}
        for i, (key, (low, high)) in enumerate(bounds.items()):
            # Denormalize
            value = low + array[i] * (high - low)
            
            # Convert to int if needed
            if isinstance(low, int) and isinstance(high, int):
                value = int(round(value))
            
            params[key] = value
        
        return params
    
    def _evaluate_objective(self, params: Dict, start_date: str,
                          end_date: str, initial_capital: float,
                          objective: str) -> float:
        """Evaluate objective function"""
        
        try:
            from backtest.engine import BacktestEngine
            from config import TradingRules
            
            # Create strategy with parameters
            strategy = TradingRules()
            for key, value in params.items():
                if hasattr(strategy, key):
                    setattr(strategy, key, value)
            
            # Run backtest
            engine = BacktestEngine(self.db)
            result = engine.run(
                start_date=start_date,
                end_date=end_date,
                initial_capital=initial_capital,
                strategy=strategy
            )
            
            # Get objective value
            if objective == 'sharpe_ratio':
                return result.sharpe_ratio
            elif objective == 'total_return':
                return result.total_return
            elif objective == 'calmar_ratio':
                return result.total_return / max(result.max_drawdown, 1)
            else:
                return result.sharpe_ratio
                
        except Exception as e:
            logger.error(f"Error in evaluation: {e}")
            return -999
    
    def _save_results(self, results: Dict, objective: str):
        """Save optimization results"""
        
        with self.db.get_connection() as conn:
            cursor = conn.cursor()
            
            # Save best parameters
            cursor.execute('''
                INSERT INTO optimization_results 
                (optimization_type, parameters, fitness_score,
                 total_return, sharpe_ratio, max_drawdown, created_date)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                'bayesian',
                json.dumps(results['best_params']),
                results['best_score'],
                0,  # Would need to run final backtest for these
                0,
                0,
                datetime.now()
            ))
            
            conn.commit()
    
    def plot_convergence(self) -> Dict:
        """Plot optimization convergence"""
        
        if not self.observations:
            return {}
        
        scores = [obs['score'] for obs in self.observations]
        best_scores = []
        
        current_best = -np.inf
        for score in scores:
            if score > current_best:
                current_best = score
            best_scores.append(current_best)
        
        return {
            'iterations': list(range(len(scores))),
            'scores': scores,
            'best_scores': best_scores,
            'final_best': self.best_score
        }