"""
GEOPOLITICAL VOLATILITY BOT - ALPACA VERSION
Complete Production System for African Markets
Author: AI Architect
Version: 2.0.0 - Alpaca Compatible
"""

import os
import json
import asyncio
import logging
import hashlib
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from enum import Enum
import abc

# Third-party imports
import aiohttp # pyright: ignore[reportMissingImports]
import asyncpg # pyright: ignore[reportMissingImports]
import redis.asyncio as redis # pyright: ignore[reportMissingImports]
import numpy as np # pyright: ignore[reportMissingImports]
import pandas as pd # pyright: ignore[reportMissingModuleSource]
from sentence_transformers import SentenceTransformer # pyright: ignore[reportMissingImports]
import faiss # pyright: ignore[reportMissingImports]
from sklearn.preprocessing import StandardScaler # pyright: ignore[reportMissingModuleSource]
from transformers import pipeline # pyright: ignore[reportMissingImports]

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION - MODIFY THESE VALUES
# ============================================================================

CONFIG = {
    # Supabase PostgreSQL - Get from Supabase Dashboard > Project Settings > Database
    "database_url": os.getenv("DATABASE_URL", "postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT].supabase.co:5432/postgres"),
    
    # Upstash Redis - Get from Upstash Console
    "redis_host": os.getenv("REDIS_HOST", "[YOUR-REDIS-HOST].upstash.io"),
    "redis_port": int(os.getenv("REDIS_PORT", 6379)),
    "redis_password": os.getenv("REDIS_PASSWORD", "[YOUR-REDIS-PASSWORD]"),
    
    # NewsAPI - Get from https://newsapi.org
    "newsapi_key": os.getenv("NEWSAPI_KEY", "[YOUR-NEWSAPI-KEY]"),
    
    # ALPACA API - Get from https://alpaca.markets > Paper Trading > API Keys
    "alpaca_api_key": os.getenv("ALPACA_API_KEY", "[YOUR-ALPACA-KEY-ID]"),
    "alpaca_secret_key": os.getenv("ALPACA_SECRET_KEY", "[YOUR-ALPACA-SECRET-KEY]"),
    "alpaca_paper": os.getenv("ALPACA_PAPER", "true").lower() == "true",  # Keep true for practice
    
    # Bot Settings
    "min_severity": 6,  # 1-10, higher = only major events
    "min_confidence": 0.65,  # 0.0-1.0, higher = more selective
    "paper_trading": os.getenv("PAPER_TRADING", "true").lower() == "true",
    
    # ALPACA ETF INSTRUMENTS (Replace forex pairs)
    # These ETFs move like the underlying assets:
    # GLD = Gold price, USO = Oil price, UUP = Dollar strength, etc.
    "instruments": [
        "GLD",      # SPDR Gold Shares (like XAU/USD)
        "USO",      # WTI Crude Oil ETF (like USOIL)
        "UUP",      # US Dollar Index Bullish Fund (like DXY)
        "VIXY",     # VIX Short-Term Futures (volatility)
        "SPY",      # S&P 500 ETF (market sentiment)
        "FXE",      # Euro Trust (like EUR/USD)
        "FXY",      # Japanese Yen Trust (like USD/JPY)
        "FXB",      # British Pound Sterling Trust (like GBP/USD)
    ],
    
    # How often to check for news (seconds)
    # NewsAPI free = 100 requests/day = every 14 minutes minimum
    "check_interval": 900,  # 15 minutes (conservative for free tier)
    
    # Risk Management
    "portfolio_value": 100000,  # Your account size in USD
    "risk_per_trade": 0.01,     # Risk 1% per trade
    "max_position_size": 0.05,  # Max 5% in one position
}

# ============================================================================
# DATA MODELS
# ============================================================================

class EventType(Enum):
    CONFLICT = "conflict"
    SANCTIONS = "sanctions"
    TRADE_WAR = "trade_war"
    POLICY = "policy"
    UNKNOWN = "unknown"

@dataclass
class GeoEvent:
    id: str
    timestamp: datetime
    headline: str
    text: str
    source: str
    embedding: Optional[np.ndarray] = None
    event_type: EventType = EventType.UNKNOWN
    severity: int = 5
    sentiment: float = 0.0
    countries: List[str] = None
    commodities: List[str] = None
    
    def __post_init__(self):
        if self.countries is None:
            self.countries = []
        if self.commodities is None:
            self.commodities = []
        if not self.id:
            self.id = hashlib.md5(f"{self.timestamp}{self.headline}".encode()).hexdigest()[:16]

@dataclass
class Signal:
    id: str
    event_id: str
    instrument: str
    direction: str  # 'buy' or 'sell'
    entry: float
    stop_loss: float
    take_profit: float
    size: float  # Dollar amount to risk
    confidence: float
    created_at: datetime

# ============================================================================
# NLP ENGINE (Processes News Text)
# ============================================================================

class SimpleNLP:
    def __init__(self):
        logger.info("Loading NLP models... (this takes 30-60 seconds first time)")
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.sentiment = pipeline(
            "sentiment-analysis", 
            model="ProsusAI/finbert",
            device=-1  # Use CPU
        )
        
        # Event type keywords
        self.event_keywords = {
            EventType.CONFLICT: ["war", "attack", "invasion", "bombing", "missile", "killed", "military", "troops", "invasion"],
            EventType.SANCTIONS: ["sanctions", "embargo", "freeze", "ban", "restricted", "blacklist", "asset freeze"],
            EventType.TRADE_WAR: ["tariff", "trade war", "import duty", "export ban", "wto", "trade dispute"],
            EventType.POLICY: ["interest rate", "fed", "ecb", "boe", "hike", "cut", "qe", "taper", "central bank"],
        }
        
        # Country mappings
        self.countries = {
            "usa": "US", "united states": "US", "america": "US", "u.s.": "US",
            "china": "CN", "chinese": "CN",
            "russia": "RU", "russian": "RU",
            "uk": "UK", "britain": "UK", "united kingdom": "UK", "british": "UK",
            "eu": "EU", "europe": "EU", "european union": "EU",
            "japan": "JP", "japanese": "JP",
            "germany": "DE", "german": "DE",
            "france": "FR", "french": "FR",
            "ukraine": "UA", "ukrainian": "UA",
            "israel": "IL", "israeli": "IL",
            "iran": "IR", "iranian": "IR",
            "saudi arabia": "SA", "saudi": "SA",
            "india": "IN", "indian": "IN",
            "brazil": "BR", "brazilian": "BR",
            "canada": "CA", "canadian": "CA",
            "australia": "AU", "australian": "AU",
            "south africa": "ZA", "south african": "ZA",
            "nigeria": "NG", "nigerian": "NG",
            "kenya": "KE", "kenyan": "KE",
            "ghana": "GH", "ghanaian": "GH",
        }
        
        # Commodity mappings
        self.commodities = {
            "oil": "OIL", "crude": "OIL", "brent": "OIL", "wti": "OIL", "petroleum": "OIL",
            "gold": "GOLD", "precious metal": "GOLD", "bullion": "GOLD",
            "silver": "SILVER",
            "natural gas": "GAS", "lng": "GAS",
            "copper": "COPPER",
            "wheat": "WHEAT", "grain": "WHEAT",
            "corn": "CORN",
            "soybeans": "SOY", "soy": "SOY",
        }
    
    def process(self, headline: str, text: str) -> GeoEvent:
        """Process a news article into structured data"""
        full_text = f"{headline}. {text}"
        
        # Create vector embedding for similarity search
        embedding = self.embedder.encode(full_text)
        
        # Classify event type
        event_type = self._classify(full_text)
        
        # Analyze sentiment (financial context)
        try:
            sent_result = self.sentiment(full_text[:512])[0]
            sentiment = 1.0 if sent_result['label'] == 'positive' else -1.0
            sentiment *= sent_result['score']
        except Exception as e:
            logger.warning(f"Sentiment analysis failed: {e}")
            sentiment = 0.0
        
        # Extract mentioned countries
        countries = self._extract_countries(full_text)
        
        # Extract mentioned commodities
        commodities = self._extract_commodities(full_text)
        
        # Calculate severity score
        severity = self._calculate_severity(event_type, sentiment, countries, commodities)
        
        return GeoEvent(
            id="",
            timestamp=datetime.utcnow(),
            headline=headline,
            text=text,
            source="",
            embedding=embedding,
            event_type=event_type,
            severity=severity,
            sentiment=sentiment,
            countries=countries,
            commodities=commodities
        )
    
    def _classify(self, text: str) -> EventType:
        """Determine event type from keywords"""
        text_lower = text.lower()
        scores = {}
        
        for event_type, keywords in self.event_keywords.items():
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                scores[event_type] = score
        
        return max(scores, key=scores.get) if scores else EventType.UNKNOWN
    
    def _extract_countries(self, text: str) -> List[str]:
        """Find country mentions in text"""
        text_lower = text.lower()
        found = []
        for name, code in self.countries.items():
            if name in text_lower:
                found.append(code)
        return list(set(found))
    
    def _extract_commodities(self, text: str) -> List[str]:
        """Find commodity mentions in text"""
        text_lower = text.lower()
        found = []
        for name, code in self.commodities.items():
            if name in text_lower:
                found.append(code)
        return list(set(found))
    
    def _calculate_severity(self, event_type, sentiment, countries, commodities) -> int:
        """Calculate 1-10 severity score"""
        base = 5
        
        # Event type weight
        weights = {
            EventType.CONFLICT: 3, 
            EventType.SANCTIONS: 2, 
            EventType.TRADE_WAR: 2, 
            EventType.POLICY: 1,
            EventType.UNKNOWN: 0
        }
        base += weights.get(event_type, 0)
        
        # Major powers involvement increases severity
        major_powers = ["US", "CN", "RU", "EU"]
        if any(c in major_powers for c in countries):
            base += 1
        
        # Strategic commodities
        strategic = ["OIL", "GOLD", "GAS"]
        if any(c in strategic for c in commodities):
            base += 1
        
        # Extreme sentiment indicates shock events
        base += abs(sentiment) * 2
        
        return min(10, max(1, int(base)))

# ============================================================================
# PATTERN MATCHING DATABASE (FAISS + PostgreSQL)
# ============================================================================

class PatternDB:
    def __init__(self, db_url: str):
        self.db_url = db_url
        self.index = None
        self.events = {}
        self.nlp = SimpleNLP()
    
    async def initialize(self):
        """Load historical events into memory"""
        try:
            conn = await asyncpg.connect(self.db_url)
            
            # Check if pgvector extension exists
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            
            # Create tables if not exist
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS historical_events (
                    event_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    description TEXT NOT NULL,
                    embedding VECTOR(384),
                    event_type TEXT,
                    severity INTEGER,
                    market_impact_data JSONB,
                    event_timestamp TIMESTAMPTZ,
                    outcome_accuracy FLOAT,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    signal_id TEXT,
                    instrument TEXT,
                    direction TEXT,
                    entry_price FLOAT,
                    exit_price FLOAT,
                    pnl FLOAT,
                    status TEXT,
                    opened_at TIMESTAMPTZ,
                    closed_at TIMESTAMPTZ
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS incoming_events (
                    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    raw_text TEXT,
                    headline TEXT,
                    source TEXT,
                    sentiment_score FLOAT,
                    severity_score INTEGER,
                    processed BOOLEAN DEFAULT FALSE,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                )
            """)
            
            # Load existing events into FAISS
            rows = await conn.fetch("""
                SELECT event_id, description, embedding, event_type, severity, market_impact_data
                FROM historical_events 
                WHERE embedding IS NOT NULL
            """)
            
            if rows:
                embeddings = []
                for row in rows:
                    emb = np.frombuffer(row['embedding'], dtype=np.float32)
                    embeddings.append(emb)
                    self.events[str(row['event_id'])] = {
                        'description': row['description'],
                        'type': row['event_type'],
                        'severity': row['severity'],
                        'impact': json.loads(row['market_impact_data']) if row['market_impact_data'] else {}
                    }
                
                embeddings = np.array(embeddings)
                faiss.normalize_L2(embeddings)
                
                self.index = faiss.IndexFlatIP(384)
                self.index.add(embeddings)
                
                logger.info(f"Loaded {len(rows)} historical events into FAISS index")
            else:
                logger.warning("No historical events found. Run backfill first.")
                self.index = faiss.IndexFlatIP(384)
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def find_similar(self, event: GeoEvent, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """Find similar historical events using vector similarity"""
        if self.index is None or self.index.ntotal == 0:
            return []
        
        query = event.embedding.reshape(1, -1).astype('float32')
        faiss.normalize_L2(query)
        
        distances, indices = self.index.search(query, top_k)
        
        results = []
        event_ids = list(self.events.keys())
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(event_ids):
                continue
            
            event_id = event_ids[idx]
            event_data = self.events[event_id]
            results.append((event_id, float(dist), event_data))
        
        return results
    
    async def save_event(self, event: GeoEvent, impact: dict):
        """Save processed event to database"""
        conn = await asyncpg.connect(self.db_url)
        
        try:
            await conn.execute("""
                INSERT INTO historical_events 
                (event_id, description, embedding, event_type, severity, market_impact_data, event_timestamp, outcome_accuracy)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 0.9)
                ON CONFLICT (event_id) DO UPDATE SET
                    market_impact_data = EXCLUDED.market_impact_data,
                    outcome_accuracy = EXCLUDED.outcome_accuracy
            """, 
                event.id,
                event.headline,
                event.embedding.tobytes(),
                event.event_type.value,
                event.severity,
                json.dumps(impact),
                event.timestamp
            )
        finally:
            await conn.close()

# ============================================================================
# VOLATILITY PREDICTION ENGINE
# ============================================================================

class VolatilityPredictor:
    def __init__(self, pattern_db: PatternDB):
        self.patterns = pattern_db
        
        # Map commodities/geopolitical factors to ETF instruments
        self.instrument_map = {
            "GOLD": ["GLD"],
            "OIL": ["USO"],
            "GAS": ["UNG"],  # US Natural Gas Fund
            "US": ["UUP", "SPY"],  # Dollar and stocks
            "CN": ["FXE", "FXY"],  # Euro/Yen as alternatives
            "RU": ["USO", "GLD"],  # Oil/Gold on Russia tension
            "EU": ["FXE", "VGK"],  # Euro ETF, Europe ETF
            "CONFLICT": ["GLD", "UUP", "VIXY"],  # Safe havens
            "SANCTIONS": ["USO", "UNG", "GLD"],
        }
    
    def predict(self, event: GeoEvent) -> Dict[str, dict]:
        """Predict market impact on each instrument"""
        
        # Find similar historical events
        similar = self.patterns.find_similar(event, top_k=5)
        
        if not similar:
            return self._heuristic_predict(event)
        
        # Aggregate predictions from similar events
        instrument_votes = {}
        
        for _, similarity, hist_data in similar:
            impact = hist_data.get('impact', {})
            
            for inst, data in impact.items():
                if inst not in instrument_votes:
                    instrument_votes[inst] = []
                
                instrument_votes[inst].append({
                    'vol_spike': data.get('vol_spike', 1.5),
                    'direction': data.get('direction', 'neutral'),
                    'weight': similarity
                })
        
        # Calculate weighted predictions
        predictions = {}
        
        for inst, votes in instrument_votes.items():
            if not votes:
                continue
                
            total_weight = sum(v['weight'] for v in votes)
            if total_weight == 0:
                continue
            
            avg_vol = sum(v['vol_spike'] * v['weight'] for v in votes) / total_weight
            
            # Direction voting
            dir_votes = {'up': 0, 'down': 0, 'neutral': 0}
            for v in votes:
                dir_votes[v['direction']] += v['weight']
            
            predicted_dir = max(dir_votes, key=dir_votes.get)
            
            predictions[inst] = {
                'volatility_multiplier': avg_vol,
                'direction': predicted_dir,
                'confidence': min(0.9, total_weight / len(votes)),
                'time_to_peak': 48  # hours
            }
        
        # Add commodity-specific predictions
        for comm in event.commodities:
            for inst in self.instrument_map.get(comm, []):
                if inst not in predictions:
                    predictions[inst] = {
                        'volatility_multiplier': 2.0 if event.severity >= 7 else 1.5,
                        'direction': 'up' if comm in ["GOLD", "OIL", "GAS"] else 'neutral',
                        'confidence': 0.6,
                        'time_to_peak': 24
                    }
        
        # Add event-type predictions
        if event.event_type == EventType.CONFLICT:
            for inst in self.instrument_map.get("CONFLICT", []):
                if inst not in predictions:
                    predictions[inst] = {
                        'volatility_multiplier': 1.8,
                        'direction': 'up',
                        'confidence': 0.55,
                        'time_to_peak': 24
                    }
        
        return predictions
    
    def _heuristic_predict(self, event: GeoEvent) -> Dict[str, dict]:
        """Fallback predictions when no historical data exists"""
        predictions = {}
        
        # Safe havens on conflict
        if event.event_type == EventType.CONFLICT:
            predictions["GLD"] = {
                'volatility_multiplier': 2.0,
                'direction': 'up',
                'confidence': 0.5,
                'time_to_peak': 24
            }
            predictions["UUP"] = {
                'volatility_multiplier': 1.5,
                'direction': 'up',
                'confidence': 0.5,
                'time_to_peak': 24
            }
            predictions["VIXY"] = {
                'volatility_multiplier': 2.5,
                'direction': 'up',
                'confidence': 0.4,
                'time_to_peak': 12
            }
        
        # Oil on Middle East tension
        if event.event_type == EventType.CONFLICT:
            middle_east = ["IR", "SA", "IQ", "IL"]
            if any(c in event.countries for c in middle_east):
                predictions["USO"] = {
                    'volatility_multiplier': 2.5,
                    'direction': 'up',
                    'confidence': 0.6,
                    'time_to_peak': 12
                }
        
        # Sanctions effects
        if event.event_type == EventType.SANCTIONS:
            if "RU" in event.countries:
                predictions["USO"] = {
                    'volatility_multiplier': 2.0,
                    'direction': 'up',
                    'confidence': 0.6,
                    'time_to_peak': 24
                }
                predictions["GLD"] = {
                    'volatility_multiplier': 1.8,
                    'direction': 'up',
                    'confidence': 0.5,
                    'time_to_peak': 48
                }
        
        # Fed/central bank policy
        if event.event_type == EventType.POLICY:
            predictions["UUP"] = {
                'volatility_multiplier': 1.5,
                'direction': 'up',
                'confidence': 0.5,
                'time_to_peak': 6
            }
            predictions["SPY"] = {
                'volatility_multiplier': 1.8,
                'direction': 'down',
                'confidence': 0.4,
                'time_to_peak': 6
            }
        
        return predictions

# ============================================================================
# TRADE SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    def __init__(self, predictor: VolatilityPredictor):
        self.predictor = predictor
        self.risk_per_trade = CONFIG['risk_per_trade']
        self.max_position = CONFIG['max_position_size']
    
    def generate(self, event: GeoEvent, current_prices: Dict[str, float], portfolio_value: float) -> List[Signal]:
        """Generate trade signals from predictions"""
        
        predictions = self.predictor.predict(event)
        signals = []
        
        for instrument, pred in predictions.items():
            # Skip low confidence predictions
            if pred['confidence'] < CONFIG['min_confidence']:
                continue
            
            if instrument not in current_prices:
                continue
            
            price = current_prices[instrument]
            direction = pred['direction']
            
            if direction == 'neutral':
                continue
            
            # Calculate stop loss and take profit levels
            # Wider stops for higher volatility predictions
            stop_pct = 0.02 * pred['volatility_multiplier']  # 2% base * multiplier
            target_pct = stop_pct * 3  # 1:3 risk/reward ratio
            
            if direction == 'up':
                entry = price
                stop = price * (1 - stop_pct)
                target = price * (1 + target_pct)
            else:
                entry = price
                stop = price * (1 + stop_pct)
                target = price * (1 - target_pct)
            
            # Position sizing based on risk
            risk_amount = portfolio_value * self.risk_per_trade
            stop_distance = abs(entry - stop)
            
            if stop_distance > 0:
                position_value = risk_amount / (stop_distance / entry)
            else:
                position_value = 0
            
            # Cap at maximum position size
            max_value = portfolio_value * self.max_position
            position_value = min(position_value, max_value)
            
            signal = Signal(
                id=f"{event.id}_{instrument}",
                event_id=event.id,
                instrument=instrument,
                direction='buy' if direction == 'up' else 'sell',
                entry=entry,
                stop_loss=stop,
                take_profit=target,
                size=position_value,
                confidence=pred['confidence'],
                created_at=datetime.utcnow()
            )
            
            signals.append(signal)
        
        # Sort by confidence, take top 3
        signals.sort(key=lambda x: x.confidence, reverse=True)
        return signals[:3]

# ============================================================================
# ALPACA BROKER INTERFACE (REPLACES OANDA)
# ============================================================================

class AlpacaBroker:
    """
    Alpaca Markets API Integration
    - Paper trading: https://paper-api.alpaca.markets
    - Live trading: https://api.alpaca.markets
    - Free for stocks/ETFs, no forex
    """
    
    def __init__(self, api_key: str, secret_key: str, paper: bool = True):
        self.api_key = api_key
        self.secret_key = secret_key
        
        if paper:
            self.base_url = "https://paper-api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
            logger.info("Using Alpaca PAPER trading (fake money)")
        else:
            self.base_url = "https://api.alpaca.markets"
            self.data_url = "https://data.alpaca.markets"
            logger.warning("Using Alpaca LIVE trading (real money!)")
        
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        }
    
    async def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Get current prices for stocks/ETFs"""
        if not symbols:
            return {}
        
        url = f"{self.data_url}/v2/stocks/snapshots"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(
                    url,
                    headers=self.headers,
                    params={"symbols": ",".join(symbols)}
                ) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"Alpaca API error: {error}")
                        return {}
                    
                    data = await resp.json()
                    
                    prices = {}
                    for symbol, snapshot in data.items():
                        if snapshot and 'latestTrade' in snapshot:
                            prices[symbol] = float(snapshot['latestTrade']['p'])
                        elif snapshot and 'dailyBar' in snapshot:
                            prices[symbol] = float(snapshot['dailyBar']['c'])
                    
                    return prices
                    
            except Exception as e:
                logger.error(f"Failed to get prices: {e}")
                return {}
    
    async def place_order(self, signal: Signal) -> bool:
        """Place market order"""
        
        # Calculate quantity (Alpaca uses shares, not dollar amount)
        qty = int(signal.size / signal.entry)
        if qty < 1:
            logger.warning(f"Position size too small for {signal.instrument}")
            return False
        
        if signal.direction == 'sell':
            qty = -qty  # Negative for short
        
        order = {
            "symbol": signal.instrument,
            "qty": abs(qty),
            "side": "buy" if signal.direction == 'buy' else "sell",
            "type": "market",
            "time_in_force": "day"
        }
        
        url = f"{self.base_url}/v2/orders"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(url, headers=self.headers, json=order) as resp:
                    if resp.status == 200:
                        result = await resp.json()
                        fill_price = result.get('filled_avg_price', signal.entry)
                        logger.info(f"✅ ORDER FILLED: {signal.instrument} {signal.direction} {abs(qty)} shares @ ${fill_price}")
                        
                        # Place stop loss and take profit as separate orders
                        await self._place_risk_orders(result['id'], signal, abs(qty))
                        return True
                    else:
                        error = await resp.text()
                        logger.error(f"❌ Order failed: {error}")
                        return False
                        
            except Exception as e:
                logger.error(f"Order execution failed: {e}")
                return False
    
    async def _place_risk_orders(self, parent_order_id: str, signal: Signal, qty: int):
        """Place stop loss and take profit orders"""
        
        # Stop loss order
        stop_order = {
            "symbol": signal.instrument,
            "qty": qty,
            "side": "sell" if signal.direction == 'buy' else "buy",
            "type": "stop",
            "stop_price": str(round(signal.stop_loss, 2)),
            "time_in_force": "gtc"  # Good till cancelled
        }
        
        # Take profit order (limit)
        profit_order = {
            "symbol": signal.instrument,
            "qty": qty,
            "side": "sell" if signal.direction == 'buy' else "buy",
            "type": "limit",
            "limit_price": str(round(signal.take_profit, 2)),
            "time_in_force": "gtc"
        }
        
        url = f"{self.base_url}/v2/orders"
        
        async with aiohttp.ClientSession() as session:
            # Place stop loss
            try:
                async with session.post(url, headers=self.headers, json=stop_order) as resp:
                    if resp.status == 200:
                        logger.info(f"🛡️ Stop loss set at ${signal.stop_loss:.2f}")
                    else:
                        logger.warning(f"Failed to set stop loss: {await resp.text()}")
            except Exception as e:
                logger.error(f"Stop loss order failed: {e}")
            
            # Place take profit
            try:
                async with session.post(url, headers=self.headers, json=profit_order) as resp:
                    if resp.status == 200:
                        logger.info(f"🎯 Take profit set at ${signal.take_profit:.2f}")
                    else:
                        logger.warning(f"Failed to set take profit: {await resp.text()}")
            except Exception as e:
                logger.error(f"Take profit order failed: {e}")
    
    async def get_account(self) -> dict:
        """Get account information"""
        url = f"{self.base_url}/v2/account"
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, headers=self.headers) as resp:
                if resp.status == 200:
                    return await resp.json()
                return {}

class PaperBroker:
    """Local paper trading for testing without any API"""
    
    def __init__(self, initial_balance: float = 100000):
        self.balance = initial_balance
        self.positions = {}
        self.trades = []
        self.prices = {
            "GLD": 180.50,
            "USO": 75.25,
            "UUP": 28.40,
            "VIXY": 45.80,
            "SPY": 445.20,
            "FXE": 105.30,
            "FXY": 85.60,
            "FXB": 125.40
        }
        logger.info(f"Paper broker initialized with ${initial_balance:,.2f}")
    
    def set_price(self, symbol: str, price: float):
        self.prices[symbol] = price
    
    async def get_prices(self, symbols: List[str]) -> Dict[str, float]:
        """Return simulated prices with small random movement"""
        result = {}
        for symbol in symbols:
            base = self.prices.get(symbol, 100.0)
            # Add tiny random movement (-0.1% to +0.1%)
            noise = np.random.uniform(-0.001, 0.001)
            result[symbol] = base * (1 + noise)
        return result
    
    async def place_order(self, signal: Signal) -> bool:
        """Simulate order execution"""
        qty = int(signal.size / signal.entry)
        if qty < 1:
            logger.warning(f"Position too small: ${signal.size:.2f}")
            return False
        
        cost = qty * signal.entry
        
        if cost > self.balance:
            logger.warning(f"Insufficient funds: need ${cost:.2f}, have ${self.balance:.2f}")
            return False
        
        self.balance -= cost
        
        self.positions[signal.id] = {
            'signal': signal,
            'qty': qty,
            'entry_time': datetime.utcnow()
        }
        
        self.trades.append({
            'action': 'OPEN',
            'signal': signal,
            'qty': qty,
            'time': datetime.utcnow()
        })
        
        logger.info(f"📝 PAPER TRADE: {signal.direction} {qty} shares of {signal.instrument} @ ${signal.entry:.2f}")
        logger.info(f"   Stop: ${signal.stop_loss:.2f} | Target: ${signal.take_profit:.2f}")
        logger.info(f"   Remaining balance: ${self.balance:,.2f}")
        
        return True
    
    async def get_account(self) -> dict:
        return {
            'cash': str(self.balance),
            'portfolio_value': str(self.balance + sum(
                pos['qty'] * self.prices.get(pos['signal'].instrument, 0)
                for pos in self.positions.values()
            ))
        }

# ============================================================================
# MAIN BOT ORCHESTRATOR
# ============================================================================

class GeoBot:
    def __init__(self):
        self.nlp = SimpleNLP()
        self.db = PatternDB(CONFIG['database_url'])
        self.predictor = VolatilityPredictor(self.db)
        self.generator = SignalGenerator(self.predictor)
        
        # Initialize broker based on config
        if CONFIG['paper_trading']:
            self.broker = PaperBroker(CONFIG['portfolio_value'])
        else:
            self.broker = AlpacaBroker(
                CONFIG['alpaca_api_key'],
                CONFIG['alpaca_secret_key'],
                paper=CONFIG['alpaca_paper']
            )
        
        self.redis = None
        self.seen_events = set()
    
    async def initialize(self):
        """Setup all connections"""
        logger.info("Initializing Geopolitical Volatility Bot...")
        
        # Initialize database and load historical patterns
        await self.db.initialize()
        
        # Connect to Redis for deduplication
        try:
            self.redis = redis.Redis(
                host=CONFIG['redis_host'],
                port=CONFIG['redis_port'],
                password=CONFIG['redis_password'],
                ssl=True,
                decode_responses=True
            )
            await self.redis.ping()
            logger.info("Redis connected")
            
            # Load seen events
            seen = await self.redis.smembers("seen_events")
            self.seen_events = set(seen)
            logger.info(f"Loaded {len(self.seen_events)} previously seen events")
            
        except Exception as e:
            logger.warning(f"Redis not available (using memory only): {e}")
            self.redis = None
        
        # Check account balance
        try:
            account = await self.broker.get_account()
            cash = float(account.get('cash', 0))
            logger.info(f"Account cash: ${cash:,.2f}")
        except Exception as e:
            logger.warning(f"Could not fetch account: {e}")
        
        logger.info("Bot initialization complete")
    
    async def run(self):
        """Main trading loop"""
        logger.info("🚀 Starting main loop...")
        logger.info(f"Monitoring {len(CONFIG['instruments'])} instruments: {', '.join(CONFIG['instruments'])}")
        logger.info(f"Checking for news every {CONFIG['check_interval']} seconds")
        
        while True:
            try:
                # Fetch latest news
                events = await self.fetch_news()
                logger.info(f"Fetched {len(events)} news articles")
                
                for event_data in events:
                    event_id = event_data['id']
                    
                    # Skip if already processed
                    if event_id in self.seen_events:
                        continue
                    
                    # Mark as seen
                    self.seen_events.add(event_id)
                    if self.redis:
                        await self.redis.sadd("seen_events", event_id)
                    
                    # Process with NLP
                    event = self.nlp.process(
                        event_data['headline'],
                        event_data['text']
                    )
                    event.source = event_data['source']
                    event.timestamp = datetime.fromisoformat(
                        event_data['timestamp'].replace('Z', '+00:00')
                    )
                    
                    logger.info(f"\n{'='*60}")
                    logger.info(f"📰 NEW EVENT: {event.headline[:80]}...")
                    logger.info(f"   Type: {event.event_type.value.upper()}")
                    logger.info(f"   Severity: {event.severity}/10")
                    logger.info(f"   Sentiment: {event.sentiment:+.2f}")
                    logger.info(f"   Countries: {', '.join(event.countries) if event.countries else 'None'}")
                    logger.info(f"   Commodities: {', '.join(event.commodities) if event.commodities else 'None'}")
                    
                    # Filter by severity threshold
                    if event.severity < CONFIG['min_severity']:
                        logger.info(f"   ⚠️  Severity {event.severity} below threshold {CONFIG['min_severity']}, skipping")
                        continue
                    
                    # Get current market prices
                    prices = await self.broker.get_prices(CONFIG['instruments'])
                    
                    if not prices:
                        logger.error("Could not fetch prices, skipping")
                        continue
                    
                    logger.info(f"   Prices: {', '.join([f'{k}=${v:.2f}' for k, v in list(prices.items())[:3]])}...")
                    
                    # Generate trading signals
                    portfolio = CONFIG['portfolio_value']
                    signals = self.generator.generate(event, prices, portfolio)
                    
                    if signals:
                        logger.info(f"   🎯 Generated {len(signals)} trading signals:")
                        
                        for i, signal in enumerate(signals, 1):
                            logger.info(f"      {i}. {signal.direction.upper()} {signal.instrument}")
                            logger.info(f"         Entry: ${signal.entry:.2f} | Stop: ${signal.stop_loss:.2f} | Target: ${signal.take_profit:.2f}")
                            logger.info(f"         Size: ${signal.size:,.2f} | Confidence: {signal.confidence:.1%}")
                            
                            # Execute trade
                            success = await self.broker.place_order(signal)
                            
                            if success:
                                await self.save_signal(signal)
                                logger.info(f"         ✅ TRADE EXECUTED")
                            else:
                                logger.info(f"         ❌ Trade failed")
                    else:
                        logger.info(f"   ℹ️  No signals generated (confidence too low)")
                    
                    logger.info(f"{'='*60}\n")
                
                # Wait before next check
                logger.info(f"Sleeping for {CONFIG['check_interval']} seconds...")
                await asyncio.sleep(CONFIG['check_interval'])
                
            except Exception as e:
                logger.error(f"❌ Error in main loop: {e}", exc_info=True)
                await asyncio.sleep(60)  # Wait 1 minute on error
    
    async def fetch_news(self) -> List[dict]:
        """Fetch news from NewsAPI"""
        if not CONFIG['newsapi_key'] or CONFIG['newsapi_key'].startswith('['):
            logger.error("No valid NewsAPI key configured")
            return []
        
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": "geopolitics OR sanctions OR war OR conflict OR 'trade war' OR 'interest rate' OR 'central bank'",
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 20,
            "apiKey": CONFIG['newsapi_key']
        }
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(url, params=params, timeout=30) as resp:
                    if resp.status != 200:
                        error = await resp.text()
                        logger.error(f"NewsAPI error {resp.status}: {error}")
                        return []
                    
                    data = await resp.json()
                    
                    if data.get('status') != 'ok':
                        logger.error(f"NewsAPI error: {data.get('message')}")
                        return []
                    
                    articles = []
                    for art in data.get('articles', []):
                        # Create unique ID from URL
                        article_id = hashlib.md5(art['url'].encode()).hexdigest()[:16]
                        
                        articles.append({
                            'id': article_id,
                            'headline': art['title'] or "No title",
                            'text': f"{art.get('description', '')} {art.get('content', '')}",
                            'source': art['source']['name'] if art.get('source') else "Unknown",
                            'timestamp': art['publishedAt']
                        })
                    
                    return articles
                    
            except Exception as e:
                logger.error(f"Failed to fetch news: {e}")
                return []
    
    async def save_signal(self, signal: Signal):
        """Save signal to database"""
        try:
            conn = await asyncpg.connect(CONFIG['database_url'])
            
            await conn.execute("""
                INSERT INTO trades (signal_id, instrument, direction, entry_price, 
                                  stop_loss, take_profit, size, status, opened_at)
                VALUES ($1, $2, $3, $4, $5, $6, $7, 'open', $8)
                ON CONFLICT DO NOTHING
            """, 
                signal.id, 
                signal.instrument, 
                signal.direction, 
                signal.entry,
                signal.stop_loss, 
                signal.take_profit, 
                signal.size, 
                signal.created_at
            )
            
            await conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save signal: {e}")

# ============================================================================
# BACKFILL UTILITY - Run once to populate historical data
# ============================================================================

async def backfill_historical_events():
    """Add sample historical events to train the pattern matcher"""
    
    logger.info("Starting historical data backfill...")
    
    # Sample major geopolitical events and their market impacts
    # These train the bot to recognize patterns
    
    sample_events = [
        {
            "description": "Russia launches full-scale invasion of Ukraine, major war in Europe begins",
            "event_type": "conflict",
            "severity": 10,
            "market_impact": {
                "GLD": {"vol_spike": 2.5, "direction": "up"},      # Gold safe haven
                "USO": {"vol_spike": 3.5, "direction": "up"},      # Oil spike
                "UUP": {"vol_spike": 1.5, "direction": "up"},      # Dollar strength
                "SPY": {"vol_spike": 2.0, "direction": "down"},     # Stocks fall
                "FXE": {"vol_spike": 2.0, "direction": "down"}      # Euro falls
            },
            "timestamp": "2022-02-24"
        },
        {
            "description": "Federal Reserve announces 75 basis point interest rate hike to combat inflation",
            "event_type": "policy",
            "severity": 7,
            "market_impact": {
                "UUP": {"vol_spike": 1.8, "direction": "up"},       # Dollar up
                "GLD": {"vol_spike": 1.5, "direction": "down"},     # Gold down
                "SPY": {"vol_spike": 2.0, "direction": "down"},     # Stocks down
                "FXE": {"vol_spike": 1.5, "direction": "down"}     # Euro down vs USD
            },
            "timestamp": "2022-06-15"
        },
        {
            "description": "United States and European Union impose sweeping sanctions on Russian oil exports",
            "event_type": "sanctions",
            "severity": 8,
            "market_impact": {
                "USO": {"vol_spike": 3.0, "direction": "up"},      # Oil prices surge
                "GLD": {"vol_spike": 1.8, "direction": "up"},      # Gold safe haven
                "UUP": {"vol_spike": 1.5, "direction": "up"},       # Dollar up
                "SPY": {"vol_spike": 1.5, "direction": "down"}     # Stocks nervous
            },
            "timestamp": "2022-03-08"
        },
        {
            "description": "Hamas attacks Israel, Israel declares war, Middle East conflict escalates",
            "event_type": "conflict",
            "severity": 9,
            "market_impact": {
                "GLD": {"vol_spike": 2.0, "direction": "up"},       # Gold safe haven
                "USO": {"vol_spike": 2.5, "direction": "up"},      # Oil risk premium
                "UUP": {"vol_spike": 1.5, "direction": "up"},      # Dollar safe haven
                "VIXY": {"vol_spike": 3.0, "direction": "up"},     # Volatility spikes
                "SPY": {"vol_spike": 1.8, "direction": "down"}     # Risk off
            },
            "timestamp": "2023-10-07"
        },
        {
            "description": "China announces major stimulus package to boost slowing economy",
            "event_type": "policy",
            "severity": 6,
            "market_impact": {
                "SPY": {"vol_spike": 1.5, "direction": "up"},      # Global stocks up
                "GLD": {"vol_spike": 1.2, "direction": "up"},     # Commodities up
                "FXE": {"vol_spike": 1.0, "direction": "up"}       # Risk currencies up
            },
            "timestamp": "2024-09-24"
        },
        {
            "description": "Saudi Arabia and Russia extend oil production cuts, supply tightens",
            "event_type": "policy",
            "severity": 7,
            "market_impact": {
                "USO": {"vol_spike": 2.0, "direction": "up"},       # Oil up
                "GLD": {"vol_spike": 1.0, "direction": "up"},       # Inflation hedge
                "UUP": {"vol_spike": 1.2, "direction": "up"}       # Dollar up (inflation)
            },
            "timestamp": "2023-07-03"
        }
    ]
    
    # Initialize NLP for embeddings
    nlp = SimpleNLP()
    
    # Connect to database
    conn = await asyncpg.connect(CONFIG['database_url'])
    
    inserted = 0
    
    try:
        for evt in sample_events:
            # Generate embedding
            embedding = nlp.embedder.encode(evt['description'])
            
            # Insert into database
            await conn.execute("""
                INSERT INTO historical_events 
                (description, embedding, event_type, severity, market_impact_data, event_timestamp, outcome_accuracy)
                VALUES ($1, $2, $3, $4, $5, $6, 0.85)
                ON CONFLICT DO NOTHING
            """, 
                evt['description'], 
                embedding.tobytes(), 
                evt['event_type'],
                evt['severity'], 
                json.dumps(evt['market_impact']), 
                evt['timestamp']
            )
            
            inserted += 1
            logger.info(f"Added: {evt['description'][:60]}...")
        
        logger.info(f"✅ Backfill complete! Added {inserted} historical events")
        
    finally:
        await conn.close()

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    import sys
    
    # Check for backfill command
    if len(sys.argv) > 1 and sys.argv[1] == "backfill":
        print("Running historical data backfill...")
        asyncio.run(backfill_historical_events())
        print("Done! You can now run the bot normally.")
    
    # Check for test command (paper trading only)
    elif len(sys.argv) > 1 and sys.argv[1] == "test":
        print("Running in TEST mode (local paper trading, no APIs needed)")
        CONFIG['paper_trading'] = True
        
        # Override with dummy config for testing
        CONFIG['database_url'] = "postgresql://localhost/test"  # Will fail gracefully
        
        bot = GeoBot()
        
        # Test NLP
        test_headline = "Major conflict breaks out in Middle East, oil prices surge"
        test_text = "Tensions escalate as military action begins. Sanctions expected."
        
        print(f"\nTesting NLP on: '{test_headline}'")
        event = bot.nlp.process(test_headline, test_text)
        
        print(f"Event Type: {event.event_type.value}")
        print(f"Severity: {event.severity}/10")
        print(f"Countries: {event.countries}")
        print(f"Commodities: {event.commodities}")
        print(f"Sentiment: {event.sentiment:+.2f}")
        
        # Test predictions (without database)
        print(f"\nTesting predictions...")
        predictions = bot.predictor._heuristic_predict(event)
        
        for inst, pred in predictions.items():
            print(f"  {inst}: {pred['direction'].upper()} (vol: {pred['volatility_multiplier']:.1f}x, conf: {pred['confidence']:.0%})")
        
        print("\nTest complete!")
    
    # Normal operation
    else:
        # Validate config
        errors = []
        
        if CONFIG['database_url'].startswith('[') or 'YOUR-' in CONFIG['database_url']:
            errors.append("DATABASE_URL not configured")
        if CONFIG['newsapi_key'].startswith('[') or 'YOUR-' in CONFIG['newsapi_key']:
            errors.append("NEWSAPI_KEY not configured")
        
        if errors:
            print("❌ CONFIGURATION ERRORS:")
            for err in errors:
                print(f"   - {err}")
            print("\nPlease edit CONFIG at the top of bot.py with your actual API keys")
            print("Or set environment variables before running")
            sys.exit(1)
        
        # Run bot
        bot = GeoBot()
        
        try:
            asyncio.run(bot.initialize())
            asyncio.run(bot.run())
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)
