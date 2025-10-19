"""
Data Models Module
Define data structures and models for the trading system
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any
from enum import Enum

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class SignalType(Enum):
    """Signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class PositionType(Enum):
    """Position types"""
    LONG = "long"
    SHORT = "short"

@dataclass
class Stock:
    """Stock information model"""
    symbol: str
    name: str
    sector: str = ""
    industry: str = ""
    market_cap: float = 0
    shares_outstanding: float = 0
    country: str = "US"
    exchange: str = "NASDAQ"
    ipo_date: Optional[datetime] = None
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'name': self.name,
            'sector': self.sector,
            'industry': self.industry,
            'market_cap': self.market_cap,
            'shares_outstanding': self.shares_outstanding,
            'country': self.country,
            'exchange': self.exchange,
            'ipo_date': self.ipo_date.isoformat() if self.ipo_date else None,
            'active': self.active
        }

@dataclass
class PriceData:
    """Price data model"""
    symbol: str
    date: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    adjusted_close: Optional[float] = None
    dividends: float = 0
    splits: float = 0
    
    @property
    def typical_price(self) -> float:
        """Calculate typical price (HLC/3)"""
        return (self.high + self.low + self.close) / 3
    
    @property
    def range(self) -> float:
        """Calculate daily range"""
        return self.high - self.low
    
    @property
    def change(self) -> float:
        """Calculate daily change"""
        return self.close - self.open
    
    @property
    def change_percent(self) -> float:
        """Calculate daily change percentage"""
        return (self.change / self.open) * 100 if self.open != 0 else 0

@dataclass
class Signal:
    """Trading signal model"""
    symbol: str
    signal_type: SignalType
    signal_strength: float
    action: str
    reason: str
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    generated_date: datetime = field(default_factory=datetime.now)
    expiry_date: Optional[datetime] = None
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_valid(self) -> bool:
        """Check if signal is still valid"""
        if self.expiry_date:
            return datetime.now() < self.expiry_date
        return True
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'signal_type': self.signal_type.value,
            'signal_strength': self.signal_strength,
            'action': self.action,
            'reason': self.reason,
            'target_price': self.target_price,
            'stop_loss': self.stop_loss,
            'generated_date': self.generated_date.isoformat(),
            'confidence': self.confidence
        }

@dataclass
class Position:
    """Position model"""
    portfolio_id: int
    symbol: str
    shares: float
    average_price: float
    position_type: PositionType = PositionType.LONG
    opened_date: datetime = field(default_factory=datetime.now)
    closed_date: Optional[datetime] = None
    current_price: float = 0
    unrealized_pnl: float = 0
    realized_pnl: float = 0
    status: str = "open"
    
    @property
    def cost_basis(self) -> float:
        """Calculate cost basis"""
        return self.shares * self.average_price
    
    @property
    def market_value(self) -> float:
        """Calculate current market value"""
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl_percent(self) -> float:
        """Calculate unrealized P&L percentage"""
        if self.cost_basis == 0:
            return 0
        return (self.unrealized_pnl / self.cost_basis) * 100
    
    def update_price(self, current_price: float):
        """Update current price and P&L"""
        self.current_price = current_price
        if self.position_type == PositionType.LONG:
            self.unrealized_pnl = (current_price - self.average_price) * self.shares
        else:  # SHORT
            self.unrealized_pnl = (self.average_price - current_price) * self.shares

@dataclass
class Order:
    """Order model"""
    portfolio_id: int
    symbol: str
    order_type: OrderType
    action: str
    quantity: float
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_date: datetime = field(default_factory=datetime.now)
    executed_date: Optional[datetime] = None
    executed_price: Optional[float] = None
    broker_order_id: Optional[str] = None
    commission: float = 0
    notes: str = ""
    
    def is_executable(self, current_price: float) -> bool:
        """Check if order can be executed at current price"""
        if self.status != OrderStatus.PENDING:
            return False
        
        if self.order_type == OrderType.MARKET:
            return True
        elif self.order_type == OrderType.LIMIT:
            if self.action == "BUY":
                return current_price <= self.limit_price
            else:  # SELL
                return current_price >= self.limit_price
        elif self.order_type == OrderType.STOP:
            if self.action == "BUY":
                return current_price >= self.stop_price
            else:  # SELL
                return current_price <= self.stop_price
        
        return False

@dataclass
class Portfolio:
    """Portfolio model"""
    id: int
    name: str
    initial_capital: float
    cash_balance: float
    created_date: datetime = field(default_factory=datetime.now)
    status: str = "active"
    positions: List[Position] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def positions_value(self) -> float:
        """Calculate total positions value"""
        return sum(p.market_value for p in self.positions if p.status == "open")
    
    @property
    def total_value(self) -> float:
        """Calculate total portfolio value"""
        return self.cash_balance + self.positions_value
    
    @property
    def total_return(self) -> float:
        """Calculate total return"""
        if self.initial_capital == 0:
            return 0
        return ((self.total_value - self.initial_capital) / self.initial_capital) * 100
    
    @property
    def buying_power(self) -> float:
        """Calculate buying power (simplified)"""
        return self.cash_balance * 2  # Assumes 2:1 margin

@dataclass
class Transaction:
    """Transaction model"""
    portfolio_id: int
    symbol: str
    transaction_type: str
    shares: float
    price: float
    total_amount: float
    commission: float = 0
    transaction_date: datetime = field(default_factory=datetime.now)
    notes: str = ""
    
    @property
    def net_amount(self) -> float:
        """Calculate net amount after commission"""
        if self.transaction_type == "BUY":
            return -(self.total_amount + self.commission)
        else:
            return self.total_amount - self.commission