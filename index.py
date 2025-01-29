import os
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Set
from dataclasses import dataclass
import threading
import queue
import joblib
import logging.config
import logging.handlers
import json
from pathlib import Path
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest, LimitOrderRequest, StopOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce, OrderType
from alpaca.data.live import StockDataStream
import yfinance as yf
import talib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv

load_dotenv()

Path("logs").mkdir(exist_ok=True)
Path("logs/strategies").mkdir(exist_ok=True) 

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] - %(name)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'simple': {
            'format': '%(asctime)s [%(levelname)s] - %(message)s'
        },
        'json': {
            'format': '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'color': {
            '()': 'colorlog.ColoredFormatter',
            'format': '%(log_color)s%(asctime)s [%(levelname)s] - %(message)s',
            'log_colors': {
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white'
            }
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'color',
            'stream': 'ext://sys.stdout'
        },
        'file_main': {
            'class': 'logging.handlers.TimedRotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': 'logs/trading.log',
            'when': 'midnight',
            'interval': 1,
            'backupCount': 30,
            'encoding': 'utf-8'
        },
        'file_errors': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': 'logs/errors.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 20,
            'encoding': 'utf-8'
        },
        'file_trades': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',  # Changed to JSON format for better analysis
            'filename': 'logs/trades.log',
            'maxBytes': 10485760,
            'backupCount': 50,
            'encoding': 'utf-8'
        },
        'performance_monitor': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'INFO',
            'formatter': 'json',  # Changed to JSON format
            'filename': 'logs/performance.log',
            'maxBytes': 10485760,
            'backupCount': 20,
            'encoding': 'utf-8'
        },
        'strategy_signals': {  # New handler for strategy signals
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'json',
            'filename': 'logs/strategies/signals.log',
            'maxBytes': 10485760,
            'backupCount': 20,
            'encoding': 'utf-8'
        },
        'market_data': {  # New handler for market data
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'json',
            'filename': 'logs/market_data.log',
            'maxBytes': 10485760,
            'backupCount': 20,
            'encoding': 'utf-8'
        }
    },
    'loggers': {
        'trading': {
            'level': 'DEBUG',
            'handlers': ['console', 'file_main'],
            'propagate': False
        },
        'trading.errors': {
            'level': 'ERROR',
            'handlers': ['file_errors', 'console'],
            'propagate': False
        },
        'trading.trades': {
            'level': 'INFO',
            'handlers': ['file_trades', 'console'],
            'propagate': False
        },
        'trading.performance': {
            'level': 'INFO',
            'handlers': ['performance_monitor'],
            'propagate': False
        },
        'trading.strategy': {  # New logger for strategy signals
            'level': 'DEBUG',
            'handlers': ['strategy_signals'],
            'propagate': False
        },
        'trading.market_data': {  # New logger for market data
            'level': 'DEBUG',
            'handlers': ['market_data'],
            'propagate': False
        }
    },
    'root': {
        'level': 'INFO',
        'handlers': ['console', 'file_main']
    }
}

# Initialize logging configuration
logging.config.dictConfig(LOGGING_CONFIG)

logger = logging.getLogger('trading')
error_logger = logging.getLogger('trading.errors')
trade_logger = logging.getLogger('trading.trades')
performance_logger = logging.getLogger('trading.performance')
strategy_logger = logging.getLogger('trading.strategy')
market_logger = logging.getLogger('trading.market_data')

@dataclass
class TradingParameters:
    """Trading parameters configuration"""
    capital_per_stock: float
    max_position_size: int
    risk_per_trade: float
    max_drawdown: float
    portfolio_stop_loss: float
    portfolio_take_profit: float
    max_trades_per_day: int
    min_volume: int
    max_spread_pct: float
    tickers: Set[str]

class Strategy:
    """Enhanced base strategy class with logging"""
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f'trading.strategy.{name.lower()}')
    
    def log_signal(self, ticker: str, signal: str, confidence: float, metrics: dict):
        """Log strategy signals with detailed metrics"""
        self.logger.debug(json.dumps({
            'strategy': self.name,
            'ticker': ticker,
            'signal': signal,
            'confidence': confidence,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }))

class BollingerBandsStrategy(Strategy):
    def __init__(self, window: int = 20, num_std: float = 2.0):
        super().__init__(name="Bollinger_Bands")
        self.window = window
        self.num_std = num_std
    
    def generate_signal(self, data: pd.DataFrame) -> Tuple[str, float, dict]:
        df = data.to_dict('records') 
        parsed_dict = {}
        for entry in df:
            for _, value in entry.items():
                parsed_dict[value[0]] = value[1]
        ticker = parsed_dict['symbol']

        bb_upper, bb_middle, bb_lower = talib.BBANDS(
            np.array([parsed_dict['close']]),
            timeperiod=self.window,
            nbdevup=self.num_std,
            nbdevdn=self.num_std
        )
        
        close = parsed_dict['close']
        signal = "HOLD"
        confidence = 0.0
        
        if close < bb_lower[-1]:
            signal = "BUY"
            confidence = min(0.95, (bb_lower[-1] - close) / bb_lower[-1] * 1.5) 
        elif close > bb_upper[-1]:
            signal = "SELL"
            confidence = min(0.95, (close - bb_upper[-1]) / bb_upper[-1] * 1.5)
            
        metrics = {
            "bb_upper": float(bb_upper[-1]),
            "bb_middle": float(bb_middle[-1]),
            "bb_lower": float(bb_lower[-1]),
            "close": float(close),
            "volatility": float(bb_upper[-1] - bb_lower[-1]) / bb_middle[-1]
        }
        
        self.log_signal(ticker, signal, confidence, metrics)
        return signal, confidence, metrics

class MachineLearningStrategy(Strategy):
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(name="Machine_Learning")
        self.model = self.load_model(model_path) if model_path else self.train_model()
        self.scaler = StandardScaler()
        
    def prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        features = pd.DataFrame()
        df = data.to_dict('records') 
        parsed_dict = {}
        for entry in df:
            for _, value in entry.items():
                parsed_dict[value[0]] = value[1]

        close_prices = np.array(parsed_dict['close'])
        high_prices = np.array(parsed_dict['high'])
        low_prices = np.array(parsed_dict['low'])
        volume = np.array(parsed_dict['volume'])

        features['rsi'] = talib.RSI(close_prices)  
        features['macd'], _, _ = talib.MACD(close_prices) 
        features['atr'] = talib.ATR(high_prices, low_prices, close_prices) 
        features['cci'] = talib.CCI(high_prices, low_prices, close_prices)

        features['mom'] = talib.MOM(close_prices, timeperiod=10)  
        features['roc'] = talib.ROC(close_prices, timeperiod=10) 

        features['obv'] = talib.OBV(close_prices, volume)
        features['ad'] = talib.AD(high_prices, low_prices, close_prices, volume) 

        return self.scaler.fit_transform(features.dropna())

    def train_model(self) -> RandomForestClassifier:
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42
        )
        return model

    def load_model(self, path: str) -> RandomForestClassifier:
        return joblib.load(path)

    def generate_signal(self, data: pd.DataFrame) -> Tuple[str, float, dict]:
        features = self.prepare_features(data)
        prediction = self.model.predict_proba(features[-1].reshape(1, -1))
        
        confidence = np.max(prediction)
        signal = "BUY" if prediction[0][1] > 0.6 else "SELL" if prediction[0][0] > 0.6 else "HOLD"
        
        metrics = {
            "prediction_probabilities": prediction.tolist(),
            "feature_importance": dict(zip(
                ['rsi', 'macd', 'atr', 'cci', 'mom', 'roc', 'obv', 'ad'],
                self.model.feature_importances_
            ))
        }
        
        return signal, confidence, metrics

class AggressiveMomentumStrategy(Strategy):
    def __init__(self, 
                 short_lookback: int = 5, 
                 momentum_threshold: float = 0.005, 
                 min_confidence: float = 0.2):
        super().__init__(name="Aggressive_Momentum")
        self.short_lookback = short_lookback
        self.momentum_threshold = momentum_threshold
        self.min_confidence = min_confidence
        
    def calculate_momentum_score(self, data: pd.DataFrame) -> float:
        """Calculate short-term momentum score for frequent trading"""
        try:

            features = pd.DataFrame()
            df = data.to_dict('records') 
            parsed_dict = {}
            for entry in df:
                for _, value in entry.items():
                    parsed_dict[value[0]] = value[1]

            closed_prices =  parsed_dict['close']
            high_prices = parsed_dict['high']
            low_prices = parsed_dict['low']
            volume = parsed_dict['volume']

            returns_1min = closed_prices.pct_change(periods=1)
            returns_5min = closed_prices.pct_change(periods=5)
            
            roc = talib.ROC(closed_prices, timeperiod=self.short_lookback)
            rsi = talib.RSI(closed_prices, timeperiod=7)  
            
            volume_ma = volume.rolling(window=5).mean()
            volume_spike = volume / volume_ma
            
            momentum_score = (
                0.4 * returns_1min.iloc[-1] + 
                0.3 * returns_5min.iloc[-1] +
                0.2 * (roc.iloc[-1] / 100) +
                0.1 * (rsi.iloc[-1] / 100 - 0.5)
            ) * volume_spike.iloc[-1]
            
            return momentum_score
            
        except Exception as e:
            self.logger.error(f"Error calculating momentum score: {e}")
            return 0.0
        
    def generate_signal(self, data: pd.DataFrame) -> Tuple[str, float, dict]:
        """Generate more frequent trading signals"""
        try:
            ticker = data.get('symbol', 'Unknown')
            momentum_score = self.calculate_momentum_score(data)
            
            signal = "HOLD"
            confidence = abs(momentum_score) + 0.2  
            
            if momentum_score > self.momentum_threshold:
                signal = "BUY"
            elif momentum_score < -self.momentum_threshold:
                signal = "SELL"
                
            confidence = max(confidence, self.min_confidence)
            confidence = min(confidence, 0.95)
            
            metrics = {
                "momentum_score": float(momentum_score),
                "volume_spike": float(data['Volume'].iloc[-1] / data['Volume'].rolling(window=5).mean().iloc[-1]),
                "rsi": float(talib.RSI(data['Close'], timeperiod=7)[-1])
            }
            
            self.log_signal(ticker, signal, confidence, metrics)
            return signal, confidence, metrics
            
        except Exception as e:
            self.logger.error(f"Error generating momentum signal: {e}")
            return "BUY", self.min_confidence, {} 
class ScalpingStrategy(Strategy):
    def __init__(self, 
                 min_profit_pct: float = 0.001, 
                 max_loss_pct: float = 0.002,
                 lookback_periods: int = 3):
        super().__init__(name="Scalping")
        self.min_profit_pct = min_profit_pct
        self.max_loss_pct = max_loss_pct
        self.lookback_periods = lookback_periods
        self.last_signal = {}
        
    def generate_signal(self, data: pd.DataFrame) -> Tuple[str, float, dict]:
        """Generate rapid scalping signals based on tiny price movements"""
        try:
            df = data.to_dict('records') 
            parsed_dict = {}
            for entry in df:
                for key, value in entry.items():
                    parsed_dict[key] = value

            ticker = parsed_dict['symbol']
            current_price = parsed_dict['close']
            volume = parsed_dict['volume']

            price_change_1min = (parsed_dict['close'] - parsed_dict['previous_close']) / parsed_dict['previous_close']

            volume_surge = volume > data['Volume'].mean()

            in_position = self.last_signal.get(ticker) == "BUY"

            signal = "HOLD"
            confidence = 0.9 

            if in_position:
                if price_change_1min > self.min_profit_pct:
                    signal = "SELL"
                elif price_change_1min < -self.max_loss_pct:
                    signal = "SELL"  # Stop loss
            else:
                lookback_prices = [entry['close'] for entry in df[-self.lookback_periods:]]
                price_increasing = all(x < y for x, y in zip(lookback_prices, lookback_prices[1:]))

                if price_increasing and volume_surge:
                    signal = "BUY"

            if signal != "HOLD":
                self.last_signal[ticker] = signal

            metrics = {
                "price_change_1min": float(price_change_1min),
                "in_position": in_position,
                "current_price": float(current_price),
                "volume_surge": volume_surge
            }

            return signal, confidence, metrics
            
        except Exception as e:
            self.logger.error(f"Error generating scalping signal: {e}")
            return "HOLD", 0.9, {}

class AdvancedTradingBot:
    def __init__(self, 
                 config_path: str = "config.json"):
        
        # Load configuration
        self.config = self.load_config(config_path)
        self.config['min_confidence'] = 0.2 
        self.config['execution']['min_trade_spacing_seconds'] = 1  
        self.config['risk_management']['max_loss_per_trade'] = 0.02 
        self.config['execution']['use_limit_orders'] = False
        
        # Initialize clients
        self.trading_client = TradingClient(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY'),
            paper=self.config['paper_trading']
        )
        
        self.stock_stream = StockDataStream(
            os.getenv('ALPACA_API_KEY'),
            os.getenv('ALPACA_SECRET_KEY')
        )
        
        self.strategies = {
            "bollinger": BollingerBandsStrategy(
                window=self.config['technical_parameters']['bollinger_window'],
                num_std=self.config['technical_parameters']['bollinger_std']
            ),
            # "ml": MachineLearningStrategy(),
            "momentum": AggressiveMomentumStrategy(),
            "scalping": ScalpingStrategy()

        }
        self.strategy_weights = self.config['strategy_weights']
        
        self.ticker_data = {ticker: pd.DataFrame() for ticker in self.config['trading_parameters']['tickers']}
        
        self.positions = {
            ticker: [] for ticker in self.config['trading_parameters']['tickers']
        }
        self.update_positions()
        self.position_details = {}
        self.orders = {}
        self.trade_log = []
        self.daily_trades = 0
        self.last_trade_reset = datetime.now()
        
        self.initial_portfolio_value = self.get_portfolio_value()
        self.max_drawdown_value = self.initial_portfolio_value * (1 - self.config['trading_parameters']['max_drawdown'])
        
        self.order_queue = queue.Queue()
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        
        threading.Thread(target=self.setup_data_streaming).start()


    def update_positions(self):
        """Enhanced position tracking with timestamps"""
        try:
            positions = self.trading_client.get_all_positions()
            
            self.positions = {
                ticker: [] for ticker in self.config['trading_parameters']['tickers']
            }
            self.position_details = {}
            
            for p in positions:
                position_id = f"{p.symbol}_{len(self.positions[p.symbol])}"
                position_info = {
                    'qty': float(p.qty),
                    'avg_entry_price': float(p.avg_entry_price),
                    'current_price': float(p.current_price),
                    'market_value': float(p.market_value),
                    'unrealized_pl': float(p.unrealized_pl),
                    'unrealized_plpc': float(p.unrealized_plpc),
                    'position_id': position_id,
                    'entry_time': getattr(p, 'created_at', datetime.now()),
                }
                
                self.positions[p.symbol].append(position_info)
                self.position_details[position_id] = position_info
                
                # # Log position details
                # logger.info(f"Updated position for {p.symbol}: {position_info}")
                
        except Exception as e:
            logger.error(f"Error updating positions: {e}")

    def can_trade(self, ticker: str, signal: str) -> bool:
        """Modified to allow multiple positions per symbol"""
        try:
            current_positions = self.positions.get(ticker, [])
            max_positions = self.config['trading_parameters']['max_positions_per_symbol']
            
            if signal == "BUY":
                # Allow buying if under max positions
                if len(current_positions) >= max_positions:
                    logger.info(f"Max positions ({max_positions}) reached for {ticker}")
                    return False
                    
            elif signal == "SELL":
                # Allow selling if we have any positions
                if not current_positions:
                    logger.info(f"No positions to sell for {ticker}")
                    return False
                    
            return True
            
        except Exception as e:
            logger.error(f"Error checking trade possibility: {e}")
            return False

    def should_sell_position(self, position: dict) -> bool:
        """Determine if a position should be sold based on multiple criteria"""
        try:
            current_time = datetime.now()
            position_age = (current_time - position['entry_time']).total_seconds()
            
            reasons = []
            
            # 1. Profit target reached
            if float(position['unrealized_plpc']) >= self.config['trading_parameters']['profit_target']:
                reasons.append(f"Profit target reached: {position['unrealized_plpc']:.4f}")
            
            # 2. Stop loss hit
            if float(position['unrealized_plpc']) <= self.config['trading_parameters']['stop_loss']:
                reasons.append(f"Stop loss hit: {position['unrealized_plpc']:.4f}")
            
            # 3. Position age exceeded
            if position_age >= self.config['trading_parameters']['max_position_age']:
                reasons.append(f"Position age exceeded: {position_age:.1f} seconds")
            
            # Log if we're selling and why
            if reasons:
                logger.info(f"Selling position {position['position_id']} for reasons: {', '.join(reasons)}")
                return True
                
            return False
            
        except Exception as e:
            logger.error(f"Error in should_sell_position: {e}")
            return False


    def load_config(self, path: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(path, 'r') as f:
                config = json.load(f)
            print(f"Configuration loaded from {path}")
            return config
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise

    def get_portfolio_value(self) -> float:
        """Get current portfolio value"""
        try:
            account = self.trading_client.get_account()
            return float(account.portfolio_value)
        except Exception as e:
            print(f"Error getting portfolio value: {e}")
            return 0.0

    def check_risk_limits(self) -> bool:
        """Simplified risk checks to allow more trading"""
        return True 

    def close_all_positions(self):
        """Close all open positions"""
        try:
            positions = self.trading_client.get_all_positions()
            for position in positions:
                self.submit_order(
                    symbol=position.symbol,
                    qty=abs(float(position.qty)),
                    side=OrderSide.SELL if float(position.qty) > 0 else OrderSide.BUY,
                    type=OrderType.MARKET
                )
            print("All positions closed")
        except Exception as e:
            logger.error(f"Error closing positions: {e}")

    def calculate_position_size(self, ticker: str, price: float) -> int:
        """Enhanced position sizing for multiple positions"""
        try:
            account = self.trading_client.get_account()
            buying_power = float(account.buying_power)
            
            # More aggressive position sizing
            risk_amount = buying_power * self.config['trading_parameters']['risk_per_trade'] * 1.5
            
            # Calculate base position size
            position_size = int(risk_amount / price)
            
            # Scale down size based on existing positions
            existing_positions = len(self.positions.get(ticker, []))
            if existing_positions > 0:
                position_size = int(position_size * (0.8 ** existing_positions))
            
            # Ensure minimum viable position size
            min_size = max(1, int(self.config['trading_parameters']['min_position_value'] / price))
            position_size = max(position_size, min_size)
            
            return min(
                position_size,
                int(self.config['trading_parameters']['max_position_size']),
                int(buying_power / price)
            )
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 1  # Return minimum size instead of 0


    def submit_order(self, symbol: str, qty: float, side: OrderSide, 
                    type: OrderType = OrderType.MARKET, 
                    limit_price: Optional[float] = None,
                    stop_price: Optional[float] = None) -> bool:
        """Submit an order to Alpaca"""
        try:
            # Create base order data
            order_data = {
                "symbol": symbol,
                "qty": qty,
                "side": side,
                "time_in_force": TimeInForce.GTC  # Now included in the order request
            }
            
            # Create the appropriate order request based on type
            if type == OrderType.MARKET:
                order = MarketOrderRequest(**order_data)
            elif type == OrderType.LIMIT and limit_price is not None:
                order_data["limit_price"] = limit_price
                order = LimitOrderRequest(**order_data)
            elif type == OrderType.STOP and stop_price is not None:
                order_data["stop_price"] = stop_price
                order = StopOrderRequest(**order_data)
            else:
                logger.error(f"Invalid order type or missing price for {type}")
                return False
                
            # Submit the order
            result = self.trading_client.submit_order(order)
            
            logger.info(f"Order submitted - Symbol: {symbol}, Side: {side}, Qty: {qty}, Type: {type}")
            logger.info(f"Order result: {result}")
                
            return True
            
        except Exception as e:
            logger.error(f"Order submission failed: {e}")
            return False

    def setup_data_streaming(self):
        """Set up real-time data streaming for tracked tickers"""
        async def handle_data(data):
            self.data_queue.put(data)

        try:
            self.stock_stream.subscribe_bars(handle_data, *self.config['trading_parameters']['tickers'])
            print(f"Subscribed to data stream for {len(self.config['trading_parameters']['tickers'])} tickers")
            self.stock_stream.run()
        except Exception as e:
            logger.error(f"Error setting up data stream: {e}")

    def process_data_stream(self):
        """Enhanced data processing with regular position checks"""
        last_position_check = datetime.now()
        check_interval = 10 
        
        while not self.stop_event.is_set():
            try:
                current_time = datetime.now()
                if (current_time - last_position_check).total_seconds() >= check_interval:
                    self.update_positions()
                    
                    for ticker in self.positions:
                        for position in self.positions[ticker]:
                            if self.should_sell_position(position):
                                self.generate_orders(ticker, "SELL", float(position['current_price']))
                    
                    last_position_check = current_time
                
                data = self.data_queue.get(timeout=1)
                ticker = data.symbol
                
                self.ticker_data[ticker] = pd.concat(
                    [self.ticker_data[ticker], pd.Series(data, name=len(self.ticker_data[ticker]))],
                    ignore_index=True
                )
                
                signal, confidence = self.aggregate_signals(ticker, self.ticker_data[ticker])
                self.generate_orders(ticker, signal, data.close)
                    
            except queue.Empty:
                continue
            except Exception as e:
                error_logger.exception(f"Data processing error: {str(e)}")


    def aggregate_signals(self, ticker: str, data: pd.DataFrame) -> Tuple[str, float]:
        """More aggressive signal generation"""
        signals = []
        confidences = []
        
        for strategy_name, strategy in self.strategies.items():
            signal, confidence, metrics = strategy.generate_signal(data)
            weight = self.strategy_weights[strategy_name]
            
            if strategy_name in ['momentum', 'scalping']:
                confidence *= 1.2
                
            signals.append(signal)
            confidences.append(confidence * weight)
        
        buy_signals = signals.count("BUY")
        sell_signals = signals.count("SELL")
        
        if buy_signals >= sell_signals:
            return "BUY", max(self.config['min_confidence'], sum(confidences) / len(confidences))
        else:
            return "SELL", max(self.config['min_confidence'], sum(confidences) / len(confidences))


    def generate_orders(self, ticker: str, signal: str, price: float):
        """Enhanced order generation with more aggressive selling"""
        try:
            self.update_positions()
            
            if ticker in self.positions and self.positions[ticker]:
                for position in self.positions[ticker]:
                    if self.should_sell_position(position):
                        order_data = {
                            "symbol": ticker,
                            "qty": position['qty'],
                            "side": OrderSide.SELL,
                            "type": OrderType.MARKET
                        }
                        logger.info(f"Generating SELL order for {ticker}: {order_data}")
                        self.order_queue.put(order_data)
                        return  
            
            if signal == "BUY" and self.can_trade(ticker, signal):
                position_size = self.calculate_position_size(ticker, price)
                if position_size > 0:
                    order_data = {
                        "symbol": ticker,
                        "qty": position_size,
                        "side": OrderSide.BUY,
                        "type": OrderType.MARKET
                    }
                    logger.info(f"Generating BUY order for {ticker}: {order_data}")
                    self.order_queue.put(order_data)
                    
        except Exception as e:
            logger.error(f"Error generating orders: {e}")

    def process_orders(self):
        """Improved order processing"""
        while not self.stop_event.is_set():
            try:
                order = self.order_queue.get(timeout=1)
                
                success = self.submit_order(**order)
                
                if success:
                    self.trade_log.append({
                        'timestamp': datetime.now(),
                        'symbol': order['symbol'],
                        'side': order['side'],
                        'qty': order['qty'],
                        'type': order['type']
                    })
                    self.daily_trades += 1
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Order processing error: {e}")


    def is_market_hours(self) -> bool:
        """Check if it's currently market hours"""
        try:
            clock = self.trading_client.get_clock()
            logger.info(f"Market hours: {clock.is_open}")
            return clock.is_open
        except Exception as e:
            logger.error(f"Error checking market hours: {e}")
            return False

    def update_trade_log(self, order: dict, result: dict):
        """Enhanced trade logging with detailed metrics"""
        try:
            entry_price = float(result.filled_avg_price) if hasattr(result, 'filled_avg_price') else None
            
            trade_info = {
                'timestamp': datetime.now().isoformat(),
                'symbol': order['symbol'],
                'side': str(order['side']),
                'qty': float(order['qty']),
                'type': str(order['type']),
                'entry_price': entry_price,
                'status': str(result.status) if hasattr(result, 'status') else None,
                'order_id': str(result.id) if hasattr(result, 'id') else None,
                'portfolio_value': float(self.get_portfolio_value()),
                'market_conditions': {
                    'is_market_hours': self.is_market_hours(),
                    'current_positions': len(self.positions)
                }
            }
            
            trade_logger.info(json.dumps(trade_info))
            self.trade_log.append(trade_info)
            self.log_performance_metrics()
            
        except Exception as e:
            error_logger.exception(f"Error updating trade log: {str(e)}")

    def log_performance_metrics(self):
        """Log performance metrics"""
        try:
            current_value = self.get_portfolio_value()
            total_return = (current_value - self.initial_portfolio_value) / self.initial_portfolio_value
            drawdown = (self.initial_portfolio_value - current_value) / self.initial_portfolio_value
            
            if self.trade_log:
                trades_df = pd.DataFrame(self.trade_log)
                profitable_trades = len(trades_df[trades_df['side'] == OrderSide.SELL])
                win_rate = profitable_trades / len(trades_df) if len(trades_df) > 0 else 0
            else:
                win_rate = 0
            
            metrics = {
                'timestamp': datetime.now(),
                'portfolio_value': current_value,
                'total_return': total_return,
                'drawdown': drawdown,
                'win_rate': win_rate,
                'daily_trades': self.daily_trades
            }
            
            performance_print(f"Performance metrics: {metrics}")
            
            if (drawdown > self.config['trading_parameters']['max_drawdown'] or 
                win_rate < self.config['performance_monitoring']['min_win_rate']):
                logger.warning("Performance thresholds breached - consider adjusting strategy")
                
        except Exception as e:
            print(f"Error logging performance metrics: {e}")

    def run(self):
        """Run the trading bot"""
        logger.info("Starting Advanced Trading Bot...")
        
        # Start processing threads
        data_thread = threading.Thread(target=self.process_data_stream)
        order_thread = threading.Thread(target=self.process_orders)
        
        data_thread.start()
        order_thread.start()
        
        try:
            while True:
                # Reset daily counters if needed
                if (datetime.now() - self.last_trade_reset).days >= 1:
                    self.daily_trades = 0
                    self.last_trade_reset = datetime.now()
                
                # Check risk limits
                if not self.check_risk_limits():
                    continue

                
                
                time.sleep(1)
                
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.stop_event.set()
            data_thread.join()
            order_thread.join()
            
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            self.stop_event.set()
            data_thread.join()
            order_thread.join()

def main():
    # Initialize and run trading bot
    config_path = "config.json"
    
    try:
        # Create instance of trading bot
        trading_bot = AdvancedTradingBot(
            config_path=config_path,
        )
        
        # Run the bot
        trading_bot.run()
        
    except KeyboardInterrupt:
        print("Shutting down trading bot...")
    except Exception as e:
        print(f"Fatal error: {e}")
        raise

if __name__ == "__main__":
    main()