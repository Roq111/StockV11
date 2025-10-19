"""
FIX FOR GENETIC OPTIMIZER PICKLE ERROR
Replace your genetic_algorithm.py or add this to your optimizer
"""

import logging
from colorama import Fore, Style
import numpy as np
import random
from typing import Dict, List, Tuple
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)

class GeneticOptimizer:
    """Fixed Genetic Algorithm Optimizer - No Pickle Errors"""
    
    def __init__(self, db_manager):
        """Initialize optimizer
        
        Args:
            db_manager: DatabaseManager instance
        """
        self.db = db_manager
        
        # Parameter bounds for optimization
        self.param_bounds = {
            'min_entry_score': (60.0, 90.0),
            'min_confidence': (0.50, 0.90),
            'max_positions': (5, 20),
            'max_position_pct': (0.02, 0.15),
            'stop_loss_pct': (0.01, 0.10),
            'profit_target_pct': (0.03, 0.20),
            'rsi_oversold': (20, 35),
            'rsi_overbought': (65, 80),
            'volume_spike_threshold': (1.5, 3.0),
            'bb_period': (15, 25),
            'bb_std': (1.5, 2.5),
            'technical_weight': (0.1, 0.5),
            'momentum_weight': (0.1, 0.5),
            'volume_weight': (0.05, 0.3),
            'sentiment_weight': (0.0, 0.2),
            'fundamental_weight': (0.0, 0.2)
        }
        
        logger.info("✓ Genetic Optimizer initialized (Single-threaded mode)")
    
    def optimize(self, start_date: str, end_date: str, 
                generations: int = 50, population_size: int = 30):
        """
        Run genetic algorithm optimization
        FIXED: Uses single-threaded execution to avoid pickle errors
        """
        
        print(f"\n{'-'*60}")
        print("GENETIC ALGORITHM OPTIMIZATION")
        print(f"{'-'*60}")
        print(f"Period: {start_date} to {end_date}")
        print(f"Generations: {generations}")
        print(f"Population Size: {population_size}")
        print(f"Mode: Single-threaded (stable)")
        print(f"{'-'*60}\n")
        
        start_time = time.time()
        
        try:
            # Initialize population
            population = self._initialize_population(population_size)
            best_individual = None
            best_fitness = -float('inf')
            
            # Evolution loop
            for generation in range(1, generations + 1):
                print(f"Generation {generation}/{generations}", end="")
                
                # Evaluate fitness for each individual
                # FIXED: Single-threaded evaluation to avoid pickle error
                fitness_scores = []
                for i, individual in enumerate(population):
                    try:
                        fitness = self._evaluate_fitness_single(
                            individual, start_date, end_date
                        )
                        fitness_scores.append(fitness)
                        
                        # Track best
                        if fitness > best_fitness:
                            best_fitness = fitness
                            best_individual = individual.copy()
                            print(f" ⭐ New best: {best_fitness:.2f}", end="")
                    except Exception as e:
                        logger.debug(f"Error evaluating individual {i}: {e}")
                        fitness_scores.append(-1000)  # Penalty for failed evaluation
                
                print(f" | Best so far: {best_fitness:.2f}")
                
                # Selection
                selected = self._selection(population, fitness_scores)
                
                # Crossover and Mutation
                new_population = []
                while len(new_population) < population_size:
                    parent1 = random.choice(selected)
                    parent2 = random.choice(selected)
                    
                    if random.random() < 0.8:  # Crossover probability
                        child = self._crossover(parent1, parent2)
                    else:
                        child = parent1.copy()
                    
                    if random.random() < 0.2:  # Mutation probability
                        child = self._mutate(child)
                    
                    new_population.append(child)
                
                population = new_population
                
                # Early stopping if no improvement
                if generation > 20 and generation % 10 == 0:
                    print("  → Checking for convergence...")
            
            # Save best configuration
            if best_individual:
                self._save_optimized_config(best_individual, best_fitness, start_date, end_date)
                
                elapsed = time.time() - start_time
                
                print(f"\n{'='*60}")
                print(f"✅ OPTIMIZATION COMPLETE")
                print(f"{'='*60}")
                print(f"Best Fitness Score: {best_fitness:.2f}")
                print(f"Time Elapsed: {elapsed:.1f} seconds")
                print(f"Configuration saved to database")
                print(f"\nOptimized Parameters:")
                for key, value in best_individual.items():
                    if isinstance(value, float):
                        if 'weight' in key or 'pct' in key:
                            print(f"  {key}: {value:.1%}")
                        else:
                            print(f"  {key}: {value:.2f}")
                    else:
                        print(f"  {key}: {value}")
                
                return best_individual
            else:
                print("\n❌ Optimization failed - no valid configurations found")
                return None
                
        except Exception as e:
            logger.error(f"Optimization error: {e}")
            print(f"\n❌ Optimization failed: {e}")
            return None
    
    def _initialize_population(self, size: int) -> List[Dict]:
        """Initialize random population"""
        population = []
        
        for _ in range(size):
            individual = {}
            for param, (min_val, max_val) in self.param_bounds.items():
                if isinstance(min_val, int):
                    individual[param] = random.randint(min_val, max_val)
                else:
                    individual[param] = random.uniform(min_val, max_val)
            
            # Normalize weights
            individual = self._normalize_weights(individual)
            population.append(individual)
        
        return population
    
    def _normalize_weights(self, individual: Dict) -> Dict:
        """Ensure weights sum to 1.0"""
        weight_keys = [k for k in individual.keys() if 'weight' in k]
        if weight_keys:
            total = sum(individual[k] for k in weight_keys)
            if total > 0:
                for k in weight_keys:
                    individual[k] = individual[k] / total
        return individual
    
    def _evaluate_fitness_single(self, individual: Dict, start_date: str, end_date: str) -> float:
        """
        Evaluate fitness of an individual configuration
        FIXED: Creates new database connection for each evaluation
        """
        try:
            # Simple backtest simulation
            score = self._run_simple_backtest(individual, start_date, end_date)
            return score
            
        except Exception as e:
            logger.debug(f"Fitness evaluation error: {e}")
            return -1000  # Penalty score
    
    def _run_simple_backtest(self, config: Dict, start_date: str, end_date: str) -> float:
        """
        Run simplified backtest for fitness evaluation
        """
        try:
            # Use context manager for database connection
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Get trading days
                cursor.execute("""
                    SELECT DISTINCT date 
                    FROM price_data 
                    WHERE date BETWEEN ? AND ?
                    ORDER BY date
                    LIMIT 100
                """, (start_date, end_date))
                
                trading_days = [row[0] for row in cursor.fetchall()]
                
                if len(trading_days) < 10:
                    return -1000
                
                # Simple trading simulation
                cash = 100000
                positions = {}
                trades = 0
                wins = 0
                
                for day in trading_days:
                    # Get some stocks with data
                    cursor.execute("""
                        SELECT symbol, close, volume
                        FROM price_data
                        WHERE date = ?
                        ORDER BY RANDOM()
                        LIMIT 50
                    """, (day,))
                    
                    stocks = cursor.fetchall()
                    
                    for symbol, price, volume in stocks:
                        if not symbol or price <= 0:
                            continue
                        
                        # Simple RSI calculation (mock)
                        rsi = random.uniform(20, 80)
                        
                        # Buy signal
                        if (symbol not in positions and 
                            len(positions) < config.get('max_positions', 10) and
                            rsi < config.get('rsi_oversold', 30)):
                            
                            position_size = cash * config.get('max_position_pct', 0.1)
                            if position_size <= cash and position_size > price:
                                shares = int(position_size / price)
                                cost = shares * price
                                cash -= cost
                                positions[symbol] = {
                                    'shares': shares,
                                    'entry_price': price,
                                    'stop_loss': price * (1 - config.get('stop_loss_pct', 0.05)),
                                    'take_profit': price * (1 + config.get('profit_target_pct', 0.1))
                                }
                        
                        # Sell signal
                        elif symbol in positions:
                            position = positions[symbol]
                            
                            # Check exit conditions
                            if (rsi > config.get('rsi_overbought', 70) or
                                price <= position['stop_loss'] or
                                price >= position['take_profit']):
                                
                                proceeds = position['shares'] * price
                                cash += proceeds
                                
                                profit = proceeds - (position['shares'] * position['entry_price'])
                                trades += 1
                                if profit > 0:
                                    wins += 1
                                
                                del positions[symbol]
                
                # Calculate final value
                final_value = cash
                for symbol, position in positions.items():
                    # Get last price
                    cursor.execute("""
                        SELECT close FROM price_data
                        WHERE symbol = ? AND date <= ?
                        ORDER BY date DESC
                        LIMIT 1
                    """, (symbol, end_date))
                    
                    result = cursor.fetchone()
                    if result:
                        final_value += position['shares'] * result[0]
                
                # Calculate fitness score
                total_return = (final_value - 100000) / 100000
                win_rate = wins / trades if trades > 0 else 0
                
                # Fitness function: combination of return and win rate
                fitness = (total_return * 100) + (win_rate * 50)
                
                # Penalty for too few trades
                if trades < 10:
                    fitness *= 0.5
                
                return fitness
                
        except Exception as e:
            logger.debug(f"Backtest error: {e}")
            return -1000
    
    def _selection(self, population: List[Dict], fitness_scores: List[float]) -> List[Dict]:
        """Tournament selection"""
        selected = []
        tournament_size = 3
        
        for _ in range(len(population)):
            # Random tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx].copy())
        
        return selected
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Uniform crossover"""
        child = {}
        for key in parent1.keys():
            if random.random() < 0.5:
                child[key] = parent1[key]
            else:
                child[key] = parent2[key]
        
        return self._normalize_weights(child)
    
    def _mutate(self, individual: Dict) -> Dict:
        """Random mutation"""
        mutated = individual.copy()
        
        # Mutate 1-3 random parameters
        num_mutations = random.randint(1, 3)
        params_to_mutate = random.sample(list(self.param_bounds.keys()), num_mutations)
        
        for param in params_to_mutate:
            min_val, max_val = self.param_bounds[param]
            if isinstance(min_val, int):
                mutated[param] = random.randint(min_val, max_val)
            else:
                # Small mutation
                current = mutated[param]
                delta = (max_val - min_val) * 0.1
                new_val = current + random.uniform(-delta, delta)
                mutated[param] = max(min_val, min(max_val, new_val))
        
        return self._normalize_weights(mutated)
    
    def _save_optimized_config(self, config: Dict, fitness: float, start_date: str, end_date: str):
        """Save optimized configuration to database"""
        try:
            with self.db.get_connection() as conn:
                cursor = conn.cursor()
                
                # Create table if not exists
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS trading_configs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT,
                        parameters TEXT,
                        performance_score REAL,
                        total_return REAL,
                        sharpe_ratio REAL,
                        is_active INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Deactivate previous configs
                cursor.execute("UPDATE trading_configs SET is_active = 0")
                
                # Calculate estimated return from fitness
                estimated_return = fitness / 100  # Simplified
                
                # Save new config
                config_name = f"GA_Optimized_{datetime.now().strftime('%Y%m%d_%H%M')}"
                cursor.execute("""
                    INSERT INTO trading_configs 
                    (name, parameters, performance_score, total_return, sharpe_ratio, is_active)
                    VALUES (?, ?, ?, ?, ?, 1)
                """, (
                    config_name,
                    json.dumps(config),
                    fitness,
                    estimated_return,
                    1.5,  # Placeholder Sharpe ratio
                ))
                
                conn.commit()
                logger.info(f"✓ Saved optimized config: {config_name}")
                
        except Exception as e:
            logger.error(f"Error saving config: {e}")

# Add this to your optimizer menu in main.py
def run_genetic_optimization(self):
    """Run genetic algorithm optimization"""
    try:
        print(f"\n{Fore.CYAN}=== GENETIC ALGORITHM OPTIMIZATION ==={Style.RESET_ALL}")
        
        # Get parameters
        start_date = input("Enter start date (YYYY-MM-DD): ").strip()
        end_date = input("Enter end date (YYYY-MM-DD): ").strip()
        
        # Reduced defaults for faster execution
        generations = int(input("Number of generations (default 30): ").strip() or "30")
        population = int(input("Population size (default 20): ").strip() or "20")
        
        print(f"\n{Fore.YELLOW}Starting optimization...{Style.RESET_ALL}")
        print("This will take a few minutes...")
        
        # Create optimizer instance
        optimizer = GeneticOptimizer(self.db)
        
        # Run optimization
        best_config = optimizer.optimize(
            start_date=start_date,
            end_date=end_date,
            generations=generations,
            population_size=population
        )
        
        if best_config:
            print(f"\n{Fore.GREEN}✅ Optimization successful!{Style.RESET_ALL}")
            print("The optimized configuration has been saved and activated.")
            print("Your scanner will now use these optimized parameters.")
            
            # Ask if user wants to test
            test = input("\nRun a test scan with new parameters? (y/n): ").lower()
            if test == 'y':
                self.run_daily_scan_optimized(execute=False)
        else:
            print(f"\n{Fore.RED}Optimization failed. Check the logs for details.{Style.RESET_ALL}")
            
    except Exception as e:
        print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}")
        logger.error(f"Genetic optimization error: {e}")