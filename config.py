"""
Configuration Module
All system configuration and trading rules
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Any
import json
from datetime import datetime

@dataclass
class TradingRules:
    """Trading strategy rules and parameters"""
    
    # Technical Indicators
    sma_short: int = 20
    sma_long: int = 50
    ema_period: int = 12
    rsi_period: int = 14
    rsi_oversold: float = 30
    rsi_overbought: float = 70
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14
    
    # Volume Indicators
    volume_ma_period: int = 20
    volume_spike_threshold: float = 2.0
    
    # Risk Management
    max_position_size: float = 0.1  # 10% of portfolio
    stop_loss_percent: float = 0.02  # 2% stop loss
    take_profit_percent: float = 0.05  # 5% take profit
    trailing_stop_percent: float = 0.03  # 3% trailing stop
    max_daily_loss: float = 0.05  # 5% max daily loss
    max_drawdown: float = 0.15  # 15% max drawdown
    
    # Position Management
    min_holding_period: int = 1  # days
    max_holding_period: int = 30  # days
    max_positions: int = 10
    position_sizing_method: str = "kelly"  # kelly, equal, risk_parity
    
    # Entry Conditions
    min_volume: float = 1000000  # Minimum daily volume
    min_price: float = 5.0  # Minimum stock price
    max_price: float = 10000.0  # Maximum stock price
    
    # Market Conditions
    use_market_filter: bool = True
    market_index: str = "SPY"
    bull_market_sma: int = 200
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
    
    def from_dict(self, params: Dict):
        """Load from dictionary"""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
    
    def validate(self) -> bool:
        """Validate parameters"""
        if self.sma_short >= self.sma_long:
            return False
        if self.stop_loss_percent >= self.take_profit_percent:
            return False
        if self.max_position_size > 0.25:  # Max 25% per position
            return False
        return True

@dataclass
class Config:
    """System configuration"""
    
    # Database
    db_path: str = "trading_system.db"
    backup_path: str = "backups/"
    
    # Data Sources
    data_provider: str = "yfinance"  # yfinance, alpha_vantage, polygon
    api_keys: Dict[str, str] = field(default_factory=dict)
    
    # Performance
    cache_enabled: bool = True
    cache_ttl: int = 3600  # seconds
    max_workers: int = 4
    batch_size: int = 100
    
    # Logging
    log_level: str = "INFO"
    log_path: str = "logs/"
    log_rotation: str = "daily"
    
    # Trading
    trading_rules: TradingRules = field(default_factory=TradingRules)
    paper_trading: bool = True
    
    # Broker
    broker_type: str = "interactive_brokers"
    broker_config: Dict[str, Any] = field(default_factory=dict)
    
    # Notifications
    enable_notifications: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["email", "telegram"])
    
    def __post_init__(self):
        """Initialize configuration"""
        # Create directories if they don't exist
        os.makedirs(self.backup_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        # Load from file if exists
        config_file = "config.json"
        if os.path.exists(config_file):
            self.load_from_file(config_file)
    
    def save_to_file(self, filename: str = "config.json"):
        """Save configuration to file"""
        config_dict = {
            "db_path": self.db_path,
            "data_provider": self.data_provider,
            "api_keys": self.api_keys,
            "cache_enabled": self.cache_enabled,
            "trading_rules": self.trading_rules.to_dict(),
            "paper_trading": self.paper_trading,
            "broker_config": self.broker_config,
            "enable_notifications": self.enable_notifications
        }
        
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def load_from_file(self, filename: str = "config.json"):
        """Load configuration from file"""
        try:
            with open(filename, 'r') as f:
                config_dict = json.load(f)
            
            for key, value in config_dict.items():
                if key == "trading_rules":
                    self.trading_rules.from_dict(value)
                elif hasattr(self, key):
                    setattr(self, key, value)
        except Exception as e:
            print(f"Error loading config: {e}")

# Global configuration instance
config = Config()