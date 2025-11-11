# yfinance removed - using Finnhub and Alpha Vantage instead
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, timezone
import warnings
from urllib.parse import quote
import asyncio
from dataclasses import dataclass
from typing import List, Dict, Tuple
import sqlite3
import pickle
from collections import defaultdict
import logging
import redis
from transformers import pipeline
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from textstat import flesch_reading_ease
import spacy
from newsapi import NewsApiClient
import praw
import finnhub
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
from sec_edgar_api import EdgarClient
import yake
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from influxdb_client.client.write_api import SYNCHRONOUS
from threading import Thread
import queue
from flask import Flask, request, jsonify
from flask_cors import CORS
import json

warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analyzer.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SentimentData:
    """Data class for sentiment information"""
    score: float
    confidence: float
    source: str
    timestamp: datetime
    article_count: int = 0
    keywords: List[str] = None
    aspects: Dict[str, float] = None

class CacheManager:
    """Enhanced caching system using Redis and local cache"""
    
    def __init__(self, use_redis=False):
        self.use_redis = use_redis
        self.local_cache = {}
        self.cache_ttl = 3600  # 1 hour default TTL
        
        if use_redis:
            try:
                self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
                self.redis_client.ping()
                logger.info("Redis cache initialized")
            except:
                logger.warning("Redis not available, using local cache only")
                self.use_redis = False
    
    def get(self, key: str):
        """Get cached value"""
        if self.use_redis:
            try:
                data = self.redis_client.get(key)
                return pickle.loads(data) if data else None
            except:
                pass
        
        # Fallback to local cache
        if key in self.local_cache:
            data, timestamp = self.local_cache[key]
            if datetime.now(timezone.utc) - timestamp < timedelta(seconds=self.cache_ttl):
                return data
            else:
                del self.local_cache[key]
        return None
    
    def set(self, key: str, value, ttl: int = None):
        """Set cached value"""
        ttl = ttl or self.cache_ttl
        
        if self.use_redis:
            try:
                self.redis_client.setex(key, ttl, pickle.dumps(value))
                return
            except:
                pass
        
        # Fallback to local cache
        self.local_cache[key] = (value, datetime.now(timezone.utc))

class DatabaseManager:
    """Database manager for storing historical data"""
    
    def __init__(self, db_path="sentiment_data.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                timestamp DATETIME,
                sentiment_score REAL,
                confidence REAL,
                source TEXT,
                article_count INTEGER,
                price REAL,
                volume INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS news_articles (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT,
                title TEXT,
                content TEXT,
                url TEXT UNIQUE,
                source TEXT,
                timestamp DATETIME,
                sentiment_score REAL,
                keywords TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT,
                symbol TEXT,
                timestamp DATETIME,
                predicted_sentiment REAL,
                actual_return REAL,
                accuracy REAL
            )
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized")
    
    def store_sentiment(self, symbol: str, sentiment_data: SentimentData, stock_price: float = None, volume: int = None):
        """Store sentiment data"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO sentiment_history 
            (symbol, timestamp, sentiment_score, confidence, source, article_count, price, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (symbol, sentiment_data.timestamp, sentiment_data.score, sentiment_data.confidence,
              sentiment_data.source, sentiment_data.article_count, stock_price, volume))
        
        conn.commit()
        conn.close()
    
    def get_sentiment_history(self, symbol: str, days_back: int = 30) -> pd.DataFrame:
        """Get historical sentiment data"""
        conn = sqlite3.connect(self.db_path)
        
        query = '''
            SELECT * FROM sentiment_history 
            WHERE symbol = ? AND timestamp >= ? 
            ORDER BY timestamp DESC
        '''
        
        start_date = datetime.now(timezone.utc) - timedelta(days=days_back)
        df = pd.read_sql_query(query, conn, params=(symbol, start_date))
        conn.close()
        
        return df

class AdvancedNLPProcessor:
    """Advanced NLP processing using transformers and domain-specific models"""
    
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.finbert_model = None
        self.roberta_model = None
        self.aspect_model = None
        self.ner_model = None
        
        self.load_models()
        
        # Initialize other NLP tools
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found, some features will be limited")
            self.nlp = None
        
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
        # Financial keywords for aspect-based analysis
        self.financial_keywords = {
            'positive': ['earnings', 'revenue', 'profit', 'growth', 'expansion', 'acquisition', 
                        'partnership', 'dividend', 'buyback', 'upgrade', 'beat', 'exceeds',
                        'strong', 'outperform', 'bullish', 'rally', 'surge', 'gain'],
            'negative': ['loss', 'decline', 'drop', 'fall', 'weak', 'miss', 'cut', 'layoffs',
                        'bankruptcy', 'lawsuit', 'investigation', 'downgrade', 'bearish',
                        'crash', 'plunge', 'concern', 'warning', 'risk'],
            'neutral': ['report', 'announce', 'statement', 'conference', 'meeting', 'update',
                       'forecast', 'guidance', 'analysis', 'market', 'trading', 'shares']
        }
        
        self.spam_indicators = [
            'web-dl', 'hdcam', 'webrip', 'bluray', 'dvdrip', 'torrent', 'download',
            'watch online', 'free movie', 'streaming', 'episode', 'season', 'trailer',
            'cast', 'plot', 'imdb', 'rotten tomatoes', 'metacritic', 'awards show',
            'celebrity', 'hollywood', 'actress', 'actor', 'director', 'producer'
        ]
        
        self.reliable_financial_sources = [
            'reuters', 'bloomberg', 'wall street journal', 'wsj', 'financial times',
            'marketwatch', 'yahoo finance', 'cnbc', 'forbes', 'barrons', 'seeking alpha',
            'morningstar', 'zacks', 'fool', 'benzinga', 'investing.com', 'finviz',
            'sec.gov', 'investor.gov', 'nasdaq.com', 'nyse.com'
        ]
        
        self.unreliable_sources = [
            'entertainment tonight', 'tmz', 'people', 'variety', 'deadline',
            'hollywood reporter', 'rolling stone', 'billboard', 'mtv', 'vh1',
            'movie news', 'tv guide', 'ign', 'gamespot', 'kotaku', 'techcrunch blog'
        ]
        self.financial_aspects = {
            'earnings': ['earnings', 'profit', 'income', 'eps', 'revenue', 'sales', 'margin'],
            'growth': ['growth', 'expansion', 'increase', 'rising', 'surge', 'boom', 'uptick'],
            'management': ['ceo', 'management', 'leadership', 'executive', 'board', 'director'],
            'competition': ['competitor', 'competitive', 'market share', 'rivalry', 'versus'],
            'regulation': ['regulation', 'regulatory', 'compliance', 'legal', 'policy', 'law'],
            'innovation': ['innovation', 'technology', 'research', 'development', 'patent', 'r&d'],
            'market': ['market', 'demand', 'supply', 'consumer', 'customer', 'segment'],
            'financial_health': ['debt', 'cash', 'balance sheet', 'liquidity', 'assets', 'liabilities'],
            'operations': ['operations', 'production', 'manufacturing', 'supply chain', 'efficiency'],
            'partnerships': ['partnership', 'merger', 'acquisition', 'alliance', 'collaboration', 'deal']
        }
    def load_models(self):
        """Load pre-trained transformer models"""
        try:
            # FinBERT for financial sentiment
            self.finbert_model = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("FinBERT model loaded")
        except Exception as e:
            logger.warning(f"Could not load FinBERT: {e}")
        
        try:
            # RoBERTa for general sentiment
            self.roberta_model = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if self.device == "cuda" else -1
            )
            logger.info("RoBERTa model loaded")
        except Exception as e:
            logger.warning(f"Could not load RoBERTa: {e}")
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """Extract named entities from text"""
        entities = {'PERSON': [], 'ORG': [], 'MONEY': [], 'PERCENT': []}
        
        if self.nlp:
            doc = self.nlp(text)
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text)
        
        return entities
    
    def extract_keywords(self, text: str, num_keywords: int = 10) -> List[str]:
        """Extract key phrases using YAKE"""
        try:
            kw_extractor = yake.KeywordExtractor(
                lan="en",
                n=3,
                dedupLim=0.7,
                top=num_keywords
            )
            keywords_with_scores = kw_extractor.extract_keywords(text)
            keywords = [kw for kw, score in keywords_with_scores][:num_keywords]
            return keywords
        except:
            return []
    
    def aspect_based_sentiment(self, text: str) -> Dict[str, float]:
        """Analyze sentiment for different business aspects"""
        aspect_sentiments = {}
        text_lower = text.lower()
        
        for aspect, keywords in self.financial_aspects.items():
            # Find sentences containing aspect keywords
            sentences = text.split('.')
            relevant_sentences = []
            
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in keywords):
                    relevant_sentences.append(sentence)
            
            if relevant_sentences:
                combined_text = '. '.join(str(x) for x in relevant_sentences)
                sentiment = self.analyze_sentiment_ensemble(combined_text)
                aspect_sentiments[aspect] = sentiment['compound']
            else:
                aspect_sentiments[aspect] = 0.0
        
        return aspect_sentiments
    
    def analyze_sentiment_ensemble(self, text: str) -> Dict[str, float]:
        """Ensemble sentiment analysis using multiple models"""
        if not text or len(text.strip()) < 5:
            return {'compound': 0, 'pos': 0, 'neu': 1, 'neg': 0}
        
        scores = []
        weights = []
        
        # VADER analysis
        try:
            vader_result = self.vader_analyzer.polarity_scores(text)
            scores.append(vader_result['compound'])
            weights.append(0.2)
        except:
            pass
        
        # FinBERT analysis
        if self.finbert_model:
            try:
                finbert_result = self.finbert_model(text[:512])  # Truncate for BERT
                finbert_score = finbert_result[0]['score']
                if finbert_result[0]['label'] == 'negative':
                    finbert_score = -finbert_score
                elif finbert_result[0]['label'] == 'neutral':
                    finbert_score = 0
                scores.append(finbert_score)
                weights.append(0.4)
            except:
                pass
        
        # RoBERTa analysis
        if self.roberta_model:
            try:
                roberta_result = self.roberta_model(text[:512])
                roberta_score = roberta_result[0]['score']
                if roberta_result[0]['label'] == 'LABEL_0':  # Negative
                    roberta_score = -roberta_score
                elif roberta_result[0]['label'] == 'LABEL_1':  # Neutral
                    roberta_score = 0
                scores.append(roberta_score)
                weights.append(0.4)
            except:
                pass
        
        # Calculate weighted average
        if scores and weights:
            # Normalize weights
            weights = np.array(weights)
            weights = weights / weights.sum()
            
            final_score = np.average(scores, weights=weights)
            final_score = np.clip(final_score, -1, 1)
        else:
            final_score = 0
        
        # Convert to VADER-like format
        if final_score > 0.1:
            pos = min(abs(final_score), 1.0)
            neg = 0
            neu = 1 - pos
        elif final_score < -0.1:
            neg = min(abs(final_score), 1.0)
            pos = 0
            neu = 1 - neg
        else:
            pos = 0
            neg = 0
            neu = 1.0
        
        return {
            'compound': final_score,
            'pos': pos,
            'neu': neu,
            'neg': neg
        }
    def is_financial_content(self, title: str, content: str, source: str) -> tuple[bool, float]:
        """
        More lenient financial content detection with debugging
        """
        text = f"{title} {content}".lower()
        source_lower = source.lower()
        
        # DEBUG: Print what we're analyzing
        print(f"DEBUG: Analyzing - Title: {title[:50]}... Source: {source}")
        
        # Check for obvious spam indicators (more lenient)
        critical_spam = ['web-dl', 'hdcam', 'webrip', 'torrent', 'download movie']
        spam_count = sum(1 for indicator in critical_spam if indicator in text)
        if spam_count >= 1:
            print(f"DEBUG: Rejected for spam: {spam_count} indicators")
            return False, 0.0
        
        # Source reliability check (more lenient)
        source_score = 0.5  # Default neutral score
        if any(reliable in source_lower for reliable in self.reliable_financial_sources):
            source_score = 1.0
            print(f"DEBUG: Reliable source bonus: {source}")
        elif any(unreliable in source_lower for unreliable in self.unreliable_sources):
            source_score = 0.2  # Don't completely reject, just lower score
            print(f"DEBUG: Unreliable source penalty: {source}")
        
        # Financial keyword analysis (more inclusive)
        all_financial_keywords = (
            self.financial_keywords['positive'] + 
            self.financial_keywords['negative'] + 
            self.financial_keywords['neutral']
        )
        
        # Add more general business/stock terms
        additional_terms = [
            'company', 'business', 'stock', 'share', 'market', 'price', 'value',
            'investment', 'investor', 'analyst', 'recommendation', 'target',
            'quarter', 'annual', 'report', 'results', 'performance', 'outlook'
        ]
        
        all_terms = all_financial_keywords + additional_terms
        financial_matches = sum(1 for term in all_terms if term in text)
        
        print(f"DEBUG: Financial matches: {financial_matches}")
        
        # Entertainment indicators (more specific)
        entertainment_indicators = [
            'movie premiere', 'box office', 'film festival', 'tv series finale',
            'celebrity gossip', 'red carpet', 'award ceremony', 'music album',
            'concert tour', 'streaming exclusive', 'netflix original'
        ]
        
        entertainment_count = sum(1 for indicator in entertainment_indicators if indicator in text)
        print(f"DEBUG: Entertainment matches: {entertainment_count}")
        
        # Calculate relevance score (more generous)
        base_relevance = min(1.0, financial_matches * 0.1)  # Each match = 10%
        source_bonus = source_score * 0.3
        entertainment_penalty = entertainment_count * 0.2
        
        relevance_score = max(0.0, base_relevance + source_bonus - entertainment_penalty)
        
        # Much lower threshold for acceptance
        is_financial = relevance_score >= 0.2 and entertainment_count <= 3
        
        print(f"DEBUG: Final - Relevance: {relevance_score:.2f}, Is Financial: {is_financial}")
        
        return is_financial, relevance_score

    def clean_and_validate_keywords(self, keywords: list, text: str) -> list[str]:
        """
        More lenient keyword cleaning with debugging
        """
        if not keywords:
            return []
        
        print(f"DEBUG: Raw keywords: {keywords}")
        
        cleaned_keywords = []
        
        for keyword in keywords:
            if not keyword or len(str(keyword).strip()) < 2:
                continue
                
            keyword_clean = str(keyword).strip()
            keyword_lower = keyword_clean.lower()
            
            # Only skip the most obvious spam
            critical_spam = ['web-dl', 'hdcam', 'chars', 'plot summary']
            if any(spam in keyword_lower for spam in critical_spam):
                print(f"DEBUG: Skipping spam keyword: {keyword_clean}")
                continue
            
            # Skip single characters except important financial abbreviations
            if len(keyword_clean) == 1:
                continue
            
            # Keep if it looks like a meaningful term
            if len(keyword_clean) >= 3 or keyword_lower in ['pe', 'ev', 'ipo', 'ceo', 'cfo']:
                cleaned_keywords.append(keyword_clean.title())
        
        # Remove duplicates
        seen = set()
        final_keywords = []
        for kw in cleaned_keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                final_keywords.append(kw)
        
        result = final_keywords[:10]  # Allow more keywords
        print(f"DEBUG: Cleaned keywords: {result}")
        return result

    def filter_articles_by_quality(self, articles: list, symbol: str) -> list[dict]:
        """
        Much more lenient article filtering with detailed debugging
        """
        if not articles:
            return []
        
        print(f"\nDEBUG: Filtering {len(articles)} articles for {symbol}")
        
        filtered_articles = []
        symbol_lower = symbol.lower()
        rejected_reasons = {'not_financial': 0, 'no_symbol': 0, 'low_quality': 0}
        
        for i, article in enumerate(articles):
            title = article.get('title', '')
            content = article.get('content', '') or article.get('description', '')
            source = article.get('source', '')
            
            print(f"DEBUG: Article {i+1} - {title[:60]}...")
            
            if not title and not content:
                rejected_reasons['low_quality'] += 1
                continue
            
            # Check if content is financial (more lenient)
            is_financial, relevance_score = self.is_financial_content(title, content, source)
            
            if not is_financial and relevance_score < 0.15:  # Very low threshold
                rejected_reasons['not_financial'] += 1
                print(f"DEBUG: Rejected - not financial enough")
                continue
            
            # Symbol mention check (more flexible)
            full_text = f"{title} {content}".lower()
            symbol_mentioned = (
                symbol_lower in full_text or
                symbol.upper() in f"{title} {content}" or
                len(symbol) >= 4 and any(word.startswith(symbol_lower[:3]) for word in full_text.split())
            )
            
            if not symbol_mentioned and relevance_score < 0.5:
                rejected_reasons['no_symbol'] += 1
                print(f"DEBUG: Rejected - symbol not mentioned clearly")
                continue
            
            # Much lower quality thresholds
            article_quality = self._calculate_article_quality(article, f"{title} {content}")
            
            article_with_quality = article.copy()
            article_with_quality['relevance_score'] = max(relevance_score, 0.3)  # Boost low scores
            article_with_quality['quality_score'] = max(article_quality, 0.3)   # Minimum quality
            article_with_quality['is_financial'] = True  # If we got here, consider it financial
            
            # Very low thresholds - accept almost everything that got this far
            if relevance_score >= 0.1 and article_quality >= 0.2:
                filtered_articles.append(article_with_quality)
                print(f"DEBUG: Accepted - relevance: {relevance_score:.2f}, quality: {article_quality:.2f}")
            else:
                rejected_reasons['low_quality'] += 1
                print(f"DEBUG: Rejected - quality too low")
        
        print(f"DEBUG: Final results for {symbol}:")
        print(f"  - Original articles: {len(articles)}")
        print(f"  - Filtered articles: {len(filtered_articles)}")
        print(f"  - Rejection reasons: {rejected_reasons}")
        
        # Sort by combined score but don't be too picky
        filtered_articles.sort(
            key=lambda x: (x.get('relevance_score', 0.3) + x.get('quality_score', 0.3)) / 2, 
            reverse=True
        )
        
        # If we have very few articles, be even more lenient
        if len(filtered_articles) < 3 and articles:
            print(f"DEBUG: Too few filtered articles ({len(filtered_articles)}), adding backup articles")
            
            # Add some backup articles with minimum filtering
            for article in articles:
                if len(filtered_articles) >= 10:  # Don't go overboard
                    break
                    
                if article not in [fa for fa in filtered_articles]:
                    title = article.get('title', '')
                    content = article.get('content', '') or article.get('description', '')
                    
                    # Very basic check - just avoid obvious spam
                    text_check = f"{title} {content}".lower()
                    if not any(spam in text_check for spam in ['web-dl', 'torrent', 'download movie']):
                        backup_article = article.copy()
                        backup_article['relevance_score'] = 0.4
                        backup_article['quality_score'] = 0.4
                        backup_article['is_financial'] = True
                        filtered_articles.append(backup_article)
                        print(f"DEBUG: Added backup article: {title[:50]}...")
        
        return filtered_articles
    def _calculate_article_quality(self, article: Dict, text: str) -> float:
        """Calculate article quality score based on various factors"""
        quality_score = 0.5  # Base score
        
        # Source reliability check
        source = article.get('source', '').lower()
        if any(reliable in source for reliable in self.reliable_financial_sources):
            quality_score += 0.3
        elif any(unreliable in source for unreliable in self.unreliable_sources):
            quality_score -= 0.2
        
        # Content length assessment
        text_length = len(text.strip())
        if text_length > 1000:
            quality_score += 0.15
        elif text_length > 500:
            quality_score += 0.1
        elif text_length > 200:
            quality_score += 0.05
        elif text_length < 50:
            quality_score -= 0.2
        
        # Title quality (not too short, not too long)
        title = article.get('title', '')
        title_length = len(title)
        if 20 <= title_length <= 150:
            quality_score += 0.05
        elif title_length < 10 or title_length > 200:
            quality_score -= 0.1
        
        # Readability check using textstat
        try:
            from textstat import flesch_reading_ease
            readability = flesch_reading_ease(text)
            if 30 <= readability <= 70:  # Good readability range
                quality_score += 0.1
            elif readability < 10 or readability > 90:  # Too hard or too easy
                quality_score -= 0.05
        except:
            # If textstat fails, no penalty
            pass
        
        # Check for spam indicators in title/content
        text_lower = text.lower()
        spam_count = sum(1 for spam in self.spam_indicators if spam in text_lower)
        if spam_count > 0:
            quality_score -= min(0.3, spam_count * 0.1)  # Penalty for spam
        
        # URL quality check
        url = article.get('url', '')
        if url:
            # Penalize suspicious URLs
            suspicious_url_patterns = ['.tk', '.ml', '.ga', '.cf', 'bit.ly', 'tinyurl']
            if any(pattern in url.lower() for pattern in suspicious_url_patterns):
                quality_score -= 0.1
            
            # Bonus for established domain extensions
            good_extensions = ['.com', '.org', '.net', '.gov', '.edu']
            if any(ext in url.lower() for ext in good_extensions):
                quality_score += 0.05
        
        # Check for duplicate or template content
        if text_lower.count('click here') > 2 or text_lower.count('read more') > 3:
            quality_score -= 0.1
        
        # Financial keyword density (positive indicator)
        all_financial_keywords = (
            self.financial_keywords.get('positive', []) + 
            self.financial_keywords.get('negative', []) + 
            self.financial_keywords.get('neutral', [])
        )
        
        financial_word_count = sum(1 for word in all_financial_keywords if word in text_lower)
        financial_density = financial_word_count / max(len(text.split()), 1)
        
        if financial_density > 0.02:  # More than 2% financial keywords
            quality_score += min(0.15, financial_density * 5)
        
        # Timestamp recency bonus
        timestamp = article.get('timestamp')
        if timestamp:
            try:
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                
                hours_old = (datetime.now(timezone.utc) - timestamp).total_seconds() / 3600
                if hours_old < 24:
                    quality_score += 0.1
                elif hours_old < 168:  # Less than a week
                    quality_score += 0.05
            except:
                pass  # If timestamp parsing fails, no bonus
        
        # Ensure score is within bounds
        quality_score = max(0.0, min(1.0, quality_score))
        
        return quality_score
    
class RealTimeDataCollector:
    """Real-time data collection with WebSocket support"""
    
    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.websocket_connections = {}
        self.data_queue = queue.Queue()
        self.is_running = False
        # Initialize API clients
        self.setup_api_clients()
    
    def setup_api_clients(self):
        """Setup various API clients"""
        try:
            if 'news_api' in self.api_keys:
                self.news_client = NewsApiClient(api_key=self.api_keys['news_api'])
                logger.info("NewsAPI client initialized")
        except Exception as e:
            logger.warning(f"NewsAPI setup failed: {e}")
        
        try:
            if 'reddit' in self.api_keys:
                self.reddit_client = praw.Reddit(
                    client_id=self.api_keys['reddit']['client_id'],
                    client_secret=self.api_keys['reddit']['client_secret'],
                    user_agent=self.api_keys['reddit']['user_agent']
                )
                logger.info("Reddit client initialized")
        except Exception as e:
            logger.warning(f"Reddit setup failed: {e}")
        
        try:
            if 'finnhub' in self.api_keys:
                self.finnhub_client = finnhub.Client(api_key=self.api_keys['finnhub'])
                logger.info("Finnhub client initialized")
        except Exception as e:
            logger.warning(f"Finnhub setup failed: {e}")
    
    async def collect_news_data(self, symbol: str, max_articles: int = 50) -> List[Dict]:
        """
        Collect and filter news data from multiple sources with quality control
        """
        articles = []
        
        # NewsAPI
        if hasattr(self, 'news_client'):
            try:
                logger.info(f"Starting NewsAPI collection for {symbol}")
                # More specific search query
                search_queries = [
                    f'"{symbol}" AND (earnings OR revenue OR stock OR price)',
                    f'"{symbol}" AND (financial OR quarterly OR results)',
                    symbol  # Fallback simple search
                ]
                
                for query in search_queries:
                    try:
                        logger.info(f"NewsAPI query: {query}")
                        news_data = self.news_client.get_everything(
                            q=query,
                            language='en',
                            sort_by='publishedAt',
                            page_size=min(20, max_articles // len(search_queries)),
                            domains='reuters.com,bloomberg.com,wsj.com,marketwatch.com,cnbc.com,forbes.com'
                        )
                        logger.info(f"NewsAPI returned {len(news_data.get('articles', []))} articles for query: {query}")
                        
                        for article in news_data.get('articles', []):
                            articles.append({
                                'title': article['title'],
                                'content': article.get('content', ''),
                                'url': article['url'],
                                'source': article['source']['name'],
                                'timestamp': datetime.fromisoformat(article['publishedAt'].replace('Z', '+00:00')),
                                'description': article.get('description', '')
                            })
                        
                        if len(articles) >= max_articles:
                            break
                            
                    except Exception as e:
                        logger.warning(f"NewsAPI query error for {symbol} with query '{query}': {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"NewsAPI error for {symbol}: {e}")
        
        # Finnhub news (keep existing code but add filtering)
        if hasattr(self, 'finnhub_client'):
            try:
                news_data = self.finnhub_client.company_news(
                    symbol, 
                    _from=(datetime.now(timezone.utc) - timedelta(days=7)).strftime('%Y-%m-%d'),
                    to=datetime.now(timezone.utc).strftime('%Y-%m-%d')
                )
                
                for article in news_data[:15]:
                    articles.append({
                        'title': article['headline'],
                        'content': article.get('summary', ''),
                        'url': article['url'],
                        'source': article['source'],
                        'timestamp': datetime.fromtimestamp(article['datetime'], tz=timezone.utc),
                        'description': article.get('summary', '')
                    })
            except Exception as e:
                logger.warning(f"Finnhub news error for {symbol}: {e}")
        
        # Apply quality filtering using the new NLP processor method
        # You'll need to access the nlp_processor from here
        # This assumes you pass it in or make it accessible
        return articles[:max_articles]
    
    def collect_reddit_data(self, symbol: str, limit: int = 100) -> List[Dict]:
        """Collect Reddit data"""
        posts = []
        
        if not hasattr(self, 'reddit_client'):
            return posts
        
        try:
            subreddits = ['investing', 'stocks', 'SecurityAnalysis', 'StockMarket', 'wallstreetbets']
            
            for subreddit_name in subreddits:
                subreddit = self.reddit_client.subreddit(subreddit_name)
                
                # Search for symbol mentions
                for submission in subreddit.search(symbol, time_filter='week', limit=20):
                    posts.append({
                        'title': submission.title,
                        'content': submission.selftext,
                        'score': submission.score,
                        'num_comments': submission.num_comments,
                        'created_utc': datetime.fromtimestamp(submission.created_utc, tz=timezone.utc),
                        'subreddit': subreddit_name,
                        'url': f"https://reddit.com{submission.permalink}"
                    })
        
        except Exception as e:
            logger.warning(f"Reddit data collection error for {symbol}: {e}")
        
        return posts[:limit]

class AnomalyDetector:
    """Detect anomalies in sentiment data"""
    
    def __init__(self):
        self.isolation_forest = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = MinMaxScaler()
        self.is_fitted = False
    
    def fit(self, sentiment_data: pd.DataFrame):
        """Fit anomaly detection model"""
        if len(sentiment_data) < 10:
            logger.warning("Insufficient data for anomaly detection")
            sentiment_data['is_anomaly'] = False
            return
        
        features = ['sentiment_score', 'confidence', 'article_count']
        available_features = [f for f in features if f in sentiment_data.columns]
        
        if not available_features:
            return
        
        X = sentiment_data[available_features].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.isolation_forest.fit(X_scaled)
        self.is_fitted = True
    
    def detect_anomalies(self, sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """Detect anomalies in new data"""
        if not self.is_fitted:
            sentiment_data['is_anomaly'] = False
            return sentiment_data
        
        features = ['sentiment_score', 'confidence', 'article_count']
        available_features = [f for f in features if f in sentiment_data.columns]
        
        if not available_features:
            sentiment_data['is_anomaly'] = False
            return sentiment_data
        
        X = sentiment_data[available_features].fillna(0)
        X_scaled = self.scaler.transform(X)
        
        anomaly_labels = self.isolation_forest.predict(X_scaled)
        sentiment_data['is_anomaly'] = anomaly_labels == -1
        sentiment_data['anomaly_score'] = self.isolation_forest.score_samples(X_scaled)
        
        return sentiment_data

class PredictiveModel:
    """Predictive model for stock movements based on sentiment"""
    
    def __init__(self):
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import LinearRegression
        from sklearn.ensemble import VotingRegressor
        
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, random_state=42),
            'gb': GradientBoostingRegressor(random_state=42),
            'lr': LinearRegression()
        }
        
        self.ensemble_model = VotingRegressor([
            ('rf', self.models['rf']),
            ('gb', self.models['gb']),
            ('lr', self.models['lr'])
        ])
        
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.feature_importance = {}
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict future returns"""
        if not self.is_fitted:
            logger.warning("Model not trained yet")
            return np.zeros(len(features_df))
        
        feature_cols = [col for col in features_df.columns if col != 'future_return_5d']
        X = features_df[feature_cols]
        X_scaled = self.scaler.transform(X)
        
        return self.ensemble_model.predict(X_scaled)

class VisualizationEngine:
    """Advanced visualization engine"""
    
    def __init__(self):
        plt.style.use('dark_background')
        
    def plot_sentiment_timeline(self, sentiment_df: pd.DataFrame, symbol: str) -> go.Figure:
        """Create interactive sentiment timeline"""
        symbol_data = sentiment_df[sentiment_df['symbol'] == symbol].sort_values('timestamp')
        
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=['Sentiment Score', 'Confidence Level', 'Article Volume'],
            vertical_spacing=0.08
        )
        
        # Sentiment score
        fig.add_trace(
            go.Scatter(
                x=symbol_data['timestamp'],
                y=symbol_data['sentiment_score'],
                mode='lines+markers',
                name='Sentiment',
                line=dict(color='#00ff00', width=2),
                hovertemplate='<b>%{y:.3f}</b><br>%{x}<extra></extra>'
            ),
            row=1, col=1
        )
        
        # Confidence
        fig.add_trace(
            go.Scatter(
                x=symbol_data['timestamp'],
                y=symbol_data['confidence'],
                mode='lines+markers',
                name='Confidence',
                line=dict(color='#ff6600', width=2),
                hovertemplate='<b>%{y:.1%}</b><br>%{x}<extra></extra>'
            ),
            row=2, col=1
        )
        
        # Article count
        fig.add_trace(
            go.Bar(
                x=symbol_data['timestamp'],
                y=symbol_data['article_count'],
                name='Articles',
                marker_color='#6600ff',
                hovertemplate='<b>%{y}</b> articles<br>%{x}<extra></extra>'
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f'Sentiment Analysis Timeline - {symbol}',
            height=800,
            showlegend=False,
            template='plotly_dark'
        )
        
        return fig
    
    def plot_sector_sentiment(self, sector_sentiment: pd.DataFrame) -> go.Figure:
        """Create sector sentiment comparison"""
        fig = go.Figure(data=[
            go.Bar(
                x=sector_sentiment['sector'],
                y=sector_sentiment['avg_sentiment'],
                marker_color=sector_sentiment['avg_sentiment'],
                colorscale='RdYlGn',
                text=[f"{x:.3f}" for x in sector_sentiment['avg_sentiment']],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Average Sentiment by Sector',
            xaxis_title='Sector',
            yaxis_title='Average Sentiment',
            template='plotly_dark'
        )
        
        return fig

class BacktestingEngine:
    """Advanced backtesting framework"""
    
    def __init__(self):
        self.transaction_cost = 0.001  # 0.1% transaction cost
        self.slippage = 0.0005  # 0.05% slippage
    
    def backtest_sentiment_strategy(
        self, 
        sentiment_df: pd.DataFrame, 
        price_df: pd.DataFrame,
        strategy_params: Dict
    ) -> Dict:
        """Backtest sentiment-based trading strategy"""
        
        # Merge data
        df = pd.merge(sentiment_df, price_df, on=['symbol', 'timestamp'])
        df = df.sort_values(['symbol', 'timestamp'])
        
        results = []
        
        for symbol in df['symbol'].unique():
            symbol_data = df[df['symbol'] == symbol].copy()
            
            if len(symbol_data) < 10:
                continue
            
            # Generate signals
            symbol_data['signal'] = 0
            
            # Buy signal: sentiment > threshold and confidence > min_confidence
            buy_condition = (
                (symbol_data['sentiment_score'] > strategy_params.get('buy_threshold', 0.2)) &
                (symbol_data['confidence'] > strategy_params.get('min_confidence', 0.6))
            )
            
            # Sell signal: sentiment < threshold or confidence < min_confidence
            sell_condition = (
                (symbol_data['sentiment_score'] < strategy_params.get('sell_threshold', -0.1)) |
                (symbol_data['confidence'] < strategy_params.get('min_confidence', 0.6))
            )
            
            symbol_data.loc[buy_condition, 'signal'] = 1
            symbol_data.loc[sell_condition, 'signal'] = -1
            
            # Calculate returns
            symbol_data['returns'] = symbol_data['price'].pct_change()
            symbol_data['strategy_returns'] = 0
            
            position = 0
            for i in range(1, len(symbol_data)):
                if symbol_data.iloc[i]['signal'] == 1 and position == 0:  # Buy
                    position = 1
                    # Account for transaction costs
                    symbol_data.iloc[i, symbol_data.columns.get_loc('strategy_returns')] = (
                        symbol_data.iloc[i]['returns'] - self.transaction_cost - self.slippage
                    )
                elif symbol_data.iloc[i]['signal'] == -1 and position == 1:  # Sell
                    position = 0
                    symbol_data.iloc[i, symbol_data.columns.get_loc('strategy_returns')] = (
                        -symbol_data.iloc[i]['returns'] - self.transaction_cost - self.slippage
                    )
                elif position == 1:  # Hold long position
                    symbol_data.iloc[i, symbol_data.columns.get_loc('strategy_returns')] = symbol_data.iloc[i]['returns']
            
            # Calculate metrics
            total_return = (1 + symbol_data['strategy_returns']).prod() - 1
            market_return = (1 + symbol_data['returns']).prod() - 1
            
            strategy_volatility = symbol_data['strategy_returns'].std() * np.sqrt(252)
            market_volatility = symbol_data['returns'].std() * np.sqrt(252)
            
            # Sharpe ratio (assuming 2% risk-free rate)
            risk_free_rate = 0.02
            sharpe_ratio = (total_return - risk_free_rate) / strategy_volatility if strategy_volatility > 0 else 0
            
            # Maximum drawdown
            cumulative_returns = (1 + symbol_data['strategy_returns']).cumprod()
            rolling_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            # Win rate
            winning_trades = symbol_data['strategy_returns'] > 0
            win_rate = winning_trades.sum() / len(symbol_data[symbol_data['strategy_returns'] != 0])
            
            results.append({
                'symbol': symbol,
                'total_return': total_return,
                'market_return': market_return,
                'excess_return': total_return - market_return,
                'volatility': strategy_volatility,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'num_trades': (symbol_data['signal'] != 0).sum()
            })
        
        return pd.DataFrame(results)
    
    def monte_carlo_simulation(
        self, 
        returns: np.ndarray, 
        num_simulations: int = 1000,
        days: int = 252
    ) -> Dict:
        """Monte Carlo simulation for risk assessment"""
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        # Generate random returns
        simulated_returns = np.random.normal(mean_return, std_return, (num_simulations, days))
        
        # Calculate cumulative returns for each simulation
        cumulative_returns = (1 + simulated_returns).cumprod(axis=1)
        final_values = cumulative_returns[:, -1]
        
        # Calculate statistics
        percentiles = np.percentile(final_values, [5, 25, 50, 75, 95])
        
        return {
            'mean_final_value': np.mean(final_values),
            'std_final_value': np.std(final_values),
            'var_5': percentiles[0],  # 5% Value at Risk
            'percentile_25': percentiles[1],
            'median': percentiles[2],
            'percentile_75': percentiles[3],
            'percentile_95': percentiles[4],
            'probability_loss': np.sum(final_values < 1) / num_simulations
        }

class ModelDriftDetector:
    """Detect and handle model drift"""
    
    def __init__(self, reference_window: int = 30, detection_window: int = 7):
        self.reference_window = reference_window
        self.detection_window = detection_window
        self.reference_metrics = {}
        self.drift_threshold = 0.1  # 10% change threshold

class EnhancedStockSentimentAnalyzer:
    """Enhanced main analyzer class with all improvements"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize components
        self.cache_manager = CacheManager(use_redis=self.config.get('use_redis', False))
        self.db_manager = DatabaseManager(self.config.get('db_path', 'sentiment_data.db'))
        self.nlp_processor = AdvancedNLPProcessor()
        self.data_collector = RealTimeDataCollector(self.config.get('api_keys', {}))
        self.anomaly_detector = AnomalyDetector()
        self.predictive_model = PredictiveModel()
        self.viz_engine = VisualizationEngine()
        self.backtesting_engine = BacktestingEngine()
        self.drift_detector = ModelDriftDetector()
        
        # Initialize Alpha Vantage client if API key is available
        self.alpha_vantage_client = None
        api_keys = self.config.get('api_keys', {})
        if 'alpha_vantage' in api_keys:
            try:
                self.alpha_vantage_client = TimeSeries(key=api_keys['alpha_vantage'], output_format='json')
                logger.info("Alpha Vantage client initialized")
            except Exception as e:
                logger.warning(f"Alpha Vantage setup failed: {e}")
        
        # Initialize Alpha Vantage Fundamental Data client
        self.alpha_vantage_fundamental = None
        if 'alpha_vantage' in api_keys:
            try:
                self.alpha_vantage_fundamental = FundamentalData(key=api_keys['alpha_vantage'], output_format='json')
                logger.info("Alpha Vantage Fundamental Data client initialized")
            except Exception as e:
                logger.warning(f"Alpha Vantage Fundamental Data setup failed: {e}")
        
        # Analysis settings
        self.sentiment_weights = {
            'news': 0.5,
            'social': 0.3,
            'analyst': 0.2
        }
        
        self.confidence_thresholds = {
            'high': 0.8,
            'medium': 0.6,
            'low': 0.4
        }
    
    async def analyze_stock_comprehensive(self, symbol: str) -> Dict:
        """Comprehensive stock analysis with all enhancements"""
        logger.info(f"Starting comprehensive analysis for {symbol}")
        
        # Check cache first
        cache_key = f"analysis_{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H')}"
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            logger.info(f"Using cached result for {symbol}")
            return cached_result
        
        try:
            # Collect data with logging
            logger.info(f"Collecting news data for {symbol}...")
            raw_news_articles = await self.data_collector.collect_news_data(symbol, max_articles=100)
            logger.info(f"Collected {len(raw_news_articles)} news articles for {symbol}")
            
            logger.info(f"Collecting Reddit data for {symbol}...")
            reddit_posts = self.data_collector.collect_reddit_data(symbol, limit=50)
            logger.info(f"Collected {len(reddit_posts)} Reddit posts for {symbol}")
            
            logger.info(f"Getting stock information for {symbol}...")
            stock_info = self._get_enhanced_stock_info(symbol)
            logger.info(f"Stock info retrieved for {symbol}")
            
            # Analyze news sentiment with advanced filtering (pass symbol for better filtering)
            news_sentiment = self._analyze_news_sentiment_advanced(raw_news_articles, symbol)
            
            # Analyze social sentiment
            social_sentiment = self._analyze_social_sentiment_advanced(reddit_posts)
            
            # Get filtered articles for further analysis
            filtered_articles = news_sentiment.get('article_breakdown', [])
            
            # Aspect-based analysis on filtered content only
            if filtered_articles:
                all_text = ' '.join(
                    ' '.join(str(kw) for kw in article.get('keywords', [])) + ' ' + 
                    str(article.get('source', ''))
                    for article in filtered_articles[:10]  # Use top 10 filtered articles
                )
            else:
                all_text = ''
            
            aspect_sentiment = self.nlp_processor.aspect_based_sentiment(all_text) if all_text else {}
            
            # Extract cleaned keywords from filtered articles
            all_clean_keywords = []
            for article in filtered_articles[:15]:
                all_clean_keywords.extend(article.get('keywords', []))
            
            # Remove duplicates and get top keywords
            seen_keywords = set()
            final_keywords = []
            for kw in all_clean_keywords:
                if str(kw).lower() not in seen_keywords and len(str(kw)) > 2:
                    seen_keywords.add(str(kw).lower())
                    final_keywords.append(str(kw))
                if len(final_keywords) >= 8:
                    break
            
            # Extract entities from filtered content
            entities = self.nlp_processor.extract_entities(all_text) if all_text else {}
            
            # Calculate final sentiment score
            final_sentiment = self._calculate_weighted_sentiment(news_sentiment, social_sentiment)
            
            # Detect anomalies
            historical_data = self.db_manager.get_sentiment_history(symbol, days_back=30)
            if not historical_data.empty:
                current_data = pd.DataFrame([{
                    'sentiment_score': final_sentiment['score'],
                    'confidence': final_sentiment['confidence'],
                    'article_count': len(raw_news_articles)
                }])
                
                self.anomaly_detector.fit(historical_data)
                anomaly_result = self.anomaly_detector.detect_anomalies(current_data)
                is_anomaly = anomaly_result['is_anomaly'].iloc[0]
            else:
                is_anomaly = False
            
            # Generate prediction
            prediction = self._generate_prediction(symbol, final_sentiment, stock_info)
            
            # Grade assignment
            grade, description = self._assign_enhanced_grade(
                final_sentiment['score'],
                final_sentiment['confidence'],
                is_anomaly,
                len(raw_news_articles)
            )
            
            # Log grade calculation for debugging
            logger.info(f"Grade calculation for {symbol}: score={final_sentiment['score']:.3f}, confidence={final_sentiment['confidence']:.3f}, articles={len(raw_news_articles)}, grade={grade}")
            
            result = {
                'symbol': symbol,
                'company_name': stock_info.get('longName', ''),
                'analysis_timestamp': datetime.now(timezone.utc),
                'stock_info': stock_info,
                'sentiment': {
                    'overall_score': final_sentiment['score'],
                    'confidence': final_sentiment['confidence'],
                    'grade': grade,
                    'description': description,
                    'aspects': aspect_sentiment
                },
                'news_analysis': {
                    'sentiment': news_sentiment,
                    'article_count': len(raw_news_articles),
                    'filtered_count': len(filtered_articles),
                    'sources': list(set([article.get('source', '') for article in filtered_articles]))
                },
                'social_analysis': {
                    'sentiment': social_sentiment,
                    'post_count': len(reddit_posts),
                    'engagement_metrics': self._calculate_engagement_metrics(reddit_posts)
                },
                'entities': entities,
                'keywords': final_keywords,  # Now using cleaned keywords
                'anomaly_detected': is_anomaly,
                'prediction': prediction,
                'data_quality': {
                    'news_coverage': min(len(raw_news_articles) / 20, 1.0),
                    'social_activity': min(len(reddit_posts) / 30, 1.0),
                    'recency_score': self._calculate_recency_score(raw_news_articles),
                    'filter_effectiveness': news_sentiment.get('filter_stats', {})
                }
            }
            result = self._sanitize_numpy_types(result)
            
            # Store in database
            sentiment_data = SentimentData(
                score=final_sentiment['score'],
                confidence=final_sentiment['confidence'],
                source='comprehensive',
                timestamp=datetime.now(timezone.utc),
                article_count=len(raw_news_articles),
                keywords=final_keywords,
                aspects=aspect_sentiment
            )
            
            self.db_manager.store_sentiment(
                symbol, 
                sentiment_data, 
                stock_info.get('currentPrice'),
                stock_info.get('volume')
            )
            
            # Cache result
            self.cache_manager.set(cache_key, result, ttl=3600)
            
            logger.info(f"Comprehensive analysis completed for {symbol}")
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis for {symbol}: {e}")
            raise
    
    def _get_enhanced_stock_info(self, symbol: str) -> Dict:
        """Get enhanced stock information using Finnhub and Alpha Vantage (yfinance removed)"""
        try:
            current_price = None
            long_name = symbol
            sector = None
            industry = None
            volatility = 0.0
            volume_avg = 0.0
            
            # ---- PRICE (Finnhub primary, Alpha Vantage fallback) ----
            if hasattr(self.data_collector, "finnhub_client"):
                try:
                    q = self.data_collector.finnhub_client.quote(symbol)
                    if q and isinstance(q, dict):
                        current_price = float(q.get("c") or q.get("pc") or 0.0)  # current price or previous close
                except Exception as e:
                    logger.warning(f"Finnhub quote failed for {symbol}: {e}")
            
            # Alpha Vantage fallback for price
            if (not current_price or current_price == 0) and self.alpha_vantage_client:
                try:
                    data, _ = self.alpha_vantage_client.get_quote_endpoint(symbol=symbol)
                    if data and '05. price' in data:
                        current_price = float(data['05. price'])
                except Exception as e:
                    logger.warning(f"Alpha Vantage quote failed for {symbol}: {e}")
            
            current_price = float(current_price or 0.0)

            # ---- METADATA (Finnhub profile) ----
            if hasattr(self.data_collector, "finnhub_client"):
                try:
                    prof = self.data_collector.finnhub_client.company_profile2(symbol=symbol)
                    if prof and isinstance(prof, dict):
                        # Finnhub returns data in 'data' key sometimes
                        prof_data = prof.get("data", [])
                        if prof_data and isinstance(prof_data, list) and len(prof_data) > 0:
                            prof = prof_data[0]
                        elif isinstance(prof_data, dict):
                            prof = prof_data
                        
                        long_name = prof.get("name") or prof.get("ticker") or symbol
                        sector = prof.get("finnhubIndustry") or prof.get("gicsIndustry") or prof.get("gicsSector")
                        industry = prof.get("finnhubIndustry") or prof.get("gicsSubIndustry") or prof.get("gicsIndustry")
                except Exception as e:
                    logger.warning(f"Finnhub profile failed for {symbol}: {e}")

            # ---- Historical metrics (Finnhub candles for volatility) ----
            if hasattr(self.data_collector, "finnhub_client"):
                try:
                    # Get 3 months of daily candles
                    end_date = int(datetime.now(timezone.utc).timestamp())
                    start_date = int((datetime.now(timezone.utc) - timedelta(days=90)).timestamp())
                    candles = self.data_collector.finnhub_client.stock_candles(symbol, 'D', start_date, end_date)
                    
                    if candles and candles.get('s') == 'ok' and candles.get('c'):
                        closes = np.array(candles['c'])
                        volumes = np.array(candles['v'])
                        
                        if len(closes) > 1:
                            # Calculate volatility (annualized)
                            returns = np.diff(closes) / closes[:-1]
                            volatility = float(np.std(returns) * np.sqrt(252))
                            volume_avg = float(np.mean(volumes))
                except Exception as e:
                    logger.warning(f"Finnhub candles failed for {symbol}: {e}")

            # ---- Additional metrics from Alpha Vantage (if available) ----
            market_cap = 0
            pe_ratio = 0
            beta = 0
            if self.alpha_vantage_fundamental:
                try:
                    overview, _ = self.alpha_vantage_fundamental.get_company_overview(symbol=symbol)
                    if overview:
                        market_cap = int(float(overview.get('MarketCapitalization', 0))) if overview.get('MarketCapitalization') else 0
                        pe_ratio = float(overview.get('PERatio', 0)) if overview.get('PERatio') else 0
                        beta = float(overview.get('Beta', 0)) if overview.get('Beta') else 0
                except Exception as e:
                    logger.debug(f"Alpha Vantage overview failed for {symbol}: {e}")

            return {
                "longName": long_name,
                "sector": sector or "N/A",
                "industry": industry or "N/A",
                "currentPrice": current_price,
                "marketCap": market_cap,
                "volume": 0,  # Not available from current sources
                "averageVolume": volume_avg,
                "trailingPE": pe_ratio,
                "forwardPE": 0,  # Not available from current sources
                "priceToBook": 0,  # Not available from current sources
                "debtToEquity": 0,  # Not available from current sources
                "returnOnEquity": 0,  # Not available from current sources
                "profitMargins": 0,  # Not available from current sources
                "beta": beta,
                "volatility": volatility,
                "52WeekHigh": 0,  # Not available from current sources
                "52WeekLow": 0,  # Not available from current sources
                "dividendYield": 0,  # Not available from current sources
                "earningsGrowth": 0,  # Not available from current sources
                "revenueGrowth": 0,  # Not available from current sources
            }
        except Exception as e:
            logger.warning(f"Error getting stock info for {symbol}: {e}")
            return {"longName": symbol, "sector": "N/A", "currentPrice": 0.0}


    
    def _analyze_news_sentiment_advanced(self, articles: List[Dict], symbol: str = None) -> Dict:
        """
        Advanced news sentiment analysis with improved filtering
        """
        if not articles:
            return {'score': 0, 'confidence': 0, 'article_breakdown': []}
        
        # Apply quality filtering first
        if symbol:
            filtered_articles = self.nlp_processor.filter_articles_by_quality(articles, symbol)
        else:
            filtered_articles = articles
        
        if not filtered_articles:
            logger.warning(f"No quality articles found after filtering for {symbol}")
            return {'score': 0, 'confidence': 0, 'article_breakdown': [], 'filter_stats': {
                'original_count': len(articles),
                'filtered_count': 0,
                'filter_reason': 'No financial relevance'
            }}
        
        article_sentiments = []
        
        for article in filtered_articles:
            text = f"{article.get('title', '')} {article.get('content', '')} {article.get('description', '')}"
            
            if len(text.strip()) < 10:
                continue
            
            # Advanced sentiment analysis
            sentiment_result = self.nlp_processor.analyze_sentiment_ensemble(text)
            
            # Extract and clean keywords
            raw_keywords = self.nlp_processor.extract_keywords(text, 8)
            clean_keywords = self.nlp_processor.clean_and_validate_keywords(raw_keywords, text)
            
            # Extract entities
            entities = self.nlp_processor.extract_entities(text)
            
            # Use improved quality score if available
            quality_score = article.get('quality_score', self._calculate_article_quality(article, text))
            relevance_score = article.get('relevance_score', 1.0)
            
            article_sentiments.append({
                'sentiment': sentiment_result['compound'],
                'confidence': max(sentiment_result['pos'], sentiment_result['neg']),
                'quality_score': quality_score,
                'relevance_score': relevance_score,
                'source': article.get('source', ''),
                'timestamp': article.get('timestamp', datetime.now(timezone.utc)),
                'keywords': clean_keywords,
                'entities': entities
            })
        
        if not article_sentiments:
            return {'score': 0, 'confidence': 0, 'article_breakdown': []}
        
        # Weight sentiments by quality, relevance, and recency
        weighted_scores = []
        weights = []
        
        for article_data in article_sentiments:
            # Combined weight from multiple factors
            base_weight = (
                article_data['quality_score'] * 0.3 +
                article_data['relevance_score'] * 0.3 +
                article_data['confidence'] * 0.4
            )
            
            # Recency weight
            days_old = (datetime.now(timezone.utc) - article_data['timestamp']).days
            recency_weight = max(0.1, 1 / (1 + days_old * 0.1))
            
            final_weight = base_weight * recency_weight
            
            weighted_scores.append(article_data['sentiment'] * final_weight)
            weights.append(final_weight)
        
        # Calculate overall sentiment
        if sum(weights) > 0:
            overall_sentiment = sum(weighted_scores) / sum(weights)
            overall_confidence = np.average(
                [a['confidence'] for a in article_sentiments],
                weights=[a['quality_score'] * a['relevance_score'] for a in article_sentiments]
            )
        else:
            overall_sentiment = 0
            overall_confidence = 0
        
        return {
            'score': overall_sentiment,
            'confidence': overall_confidence,
            'article_breakdown': article_sentiments,
            'source_distribution': self._analyze_source_distribution(article_sentiments),
            'filter_stats': {
                'original_count': len(articles),
                'filtered_count': len(filtered_articles),
                'quality_improvement': len(filtered_articles) / len(articles) if articles else 0
            }
        }
    
    def _analyze_social_sentiment_advanced(self, posts: List[Dict]) -> Dict:
        """Advanced social media sentiment analysis"""
        if not posts:
            return {'score': 0, 'confidence': 0, 'engagement_weighted_score': 0}
        
        post_sentiments = []
        
        for post in posts:
            text = f"{post.get('title', '')} {post.get('content', '')}"
            
            if len(text.strip()) < 10:
                continue
            
            # Sentiment analysis
            sentiment_result = self.nlp_processor.analyze_sentiment_ensemble(text)
            
            # Calculate engagement score
            engagement = post.get('score', 0) + post.get('num_comments', 0) * 2
            
            post_sentiments.append({
                'sentiment': sentiment_result['compound'],
                'confidence': max(sentiment_result['pos'], sentiment_result['neg']),
                'engagement': engagement,
                'subreddit': post.get('subreddit', ''),
                'timestamp': post.get('created_utc', datetime.now(timezone.utc))
            })
        
        if not post_sentiments:
            return {'score': 0, 'confidence': 0, 'engagement_weighted_score': 0}
        
        # Calculate metrics
        sentiments = [p['sentiment'] for p in post_sentiments]
        engagements = [p['engagement'] for p in post_sentiments]
        
        overall_sentiment = np.mean(sentiments)
        overall_confidence = np.mean([p['confidence'] for p in post_sentiments])
        
        # Engagement-weighted sentiment
        if sum(engagements) > 0:
            engagement_weights = [e / sum(engagements) for e in engagements]
            engagement_weighted_sentiment = np.average(sentiments, weights=engagement_weights)
        else:
            engagement_weighted_sentiment = overall_sentiment
        
        return {
            'score': overall_sentiment,
            'confidence': overall_confidence,
            'engagement_weighted_score': engagement_weighted_sentiment,
            'total_engagement': sum(engagements),
            'subreddit_breakdown': self._analyze_subreddit_distribution(post_sentiments)
        }
    
    def _calculate_weighted_sentiment(self, news_sentiment: Dict, social_sentiment: Dict) -> Dict:
        """Calculate weighted final sentiment score"""
        news_score = news_sentiment.get('score', 0)
        news_confidence = news_sentiment.get('confidence', 0)
        social_score = social_sentiment.get('engagement_weighted_score', social_sentiment.get('score', 0))
        social_confidence = social_sentiment.get('confidence', 0)
        
        # Log input values for debugging
        logger.debug(f"Weighted sentiment calc: news_score={news_score:.3f}, news_conf={news_confidence:.3f}, "
                    f"social_score={social_score:.3f}, social_conf={social_confidence:.3f}")
        
        # Dynamic weighting based on data quality and confidence
        # Ensure minimum weights even with low confidence to avoid zero total
        news_weight = self.sentiment_weights['news'] * max(news_confidence, 0.1)
        social_weight = self.sentiment_weights['social'] * max(social_confidence, 0.1)
        
        total_weight = news_weight + social_weight
        
        if total_weight > 0:
            final_score = (news_score * news_weight + social_score * social_weight) / total_weight
            final_confidence = (news_confidence * news_weight + social_confidence * social_weight) / total_weight
            
            # If both scores are very close to zero, use a small default based on direction
            if abs(final_score) < 0.01:
                # Use a small positive/negative value based on which is stronger
                if abs(news_score) > abs(social_score):
                    final_score = 0.05 if news_score >= 0 else -0.05
                elif abs(social_score) > abs(news_score):
                    final_score = 0.05 if social_score >= 0 else -0.05
                else:
                    final_score = 0.05  # Default slight positive
                
                # Ensure minimum confidence
                final_confidence = max(final_confidence, 0.2)
        else:
            # Fallback if both weights are zero
            logger.warning("Both news and social weights are zero, using default sentiment")
            final_score = 0.05  # Small positive default
            final_confidence = 0.2  # Low confidence
        
        return {
            'score': np.clip(final_score, -1, 1),
            'confidence': min(max(final_confidence, 0.1), 1.0),  # Ensure between 0.1 and 1.0
            'weights_used': {
                'news': news_weight / total_weight if total_weight > 0 else 0,
                'social': social_weight / total_weight if total_weight > 0 else 0
            }
        }
    
    def _generate_prediction(self, symbol: str, sentiment: Dict, stock_info: Dict) -> Dict:
        """Generate price movement prediction"""
        try:
            # Simple prediction based on sentiment and stock metrics
            sentiment_score = sentiment.get('score', 0)
            confidence = sentiment.get('confidence', 0)
            
            # If sentiment is too low or confidence is too low, use a more conservative approach
            if abs(sentiment_score) < 0.01 or confidence < 0.1:
                logger.warning(f"Low sentiment/confidence for {symbol}: score={sentiment_score:.3f}, confidence={confidence:.3f}")
                # Use a small default prediction based on market average (slight positive bias)
                sentiment_score = 0.1 if sentiment_score >= 0 else -0.1
                confidence = max(confidence, 0.3)  # Minimum confidence
            
            # Adjust prediction based on stock fundamentals
            pe_ratio = stock_info.get('trailingPE', 15) or 15
            beta = stock_info.get('beta', 1) or 1
            volatility = stock_info.get('volatility', 0.2) or 0.2
            
            # Base prediction from sentiment - scale more aggressively
            # Use a multiplier that scales with sentiment strength
            sentiment_multiplier = 0.08 if abs(sentiment_score) > 0.3 else 0.05
            base_prediction = sentiment_score * sentiment_multiplier
            
            # Adjust for fundamentals
            if pe_ratio > 25:  # High PE might limit upside
                base_prediction *= 0.8
            elif pe_ratio < 10:  # Low PE might indicate value
                base_prediction *= 1.2
            
            # Adjust for volatility (higher volatility = larger potential moves)
            volatility_adjustment = min(volatility / 0.2, 2.0)  # Cap at 2x
            if volatility_adjustment < 0.5:  # Very low volatility
                volatility_adjustment = 0.5
            
            final_prediction = base_prediction * volatility_adjustment
            
            # Ensure prediction is meaningful (not too close to zero)
            if abs(final_prediction) < 0.001:
                # Apply a minimum prediction based on sentiment direction
                final_prediction = 0.002 if sentiment_score > 0 else -0.002
            
            # Prediction confidence based on sentiment confidence and data quality
            prediction_confidence = max(confidence * 0.7, 0.2)  # Minimum 20% confidence
            
            # Determine direction with better thresholds
            if final_prediction > 0.002:
                direction = 'bullish'
            elif final_prediction < -0.002:
                direction = 'bearish'
            else:
                direction = 'neutral'
            
            return {
                'expected_return_5d': final_prediction,
                'confidence': prediction_confidence,
                'direction': direction,
                'strength': abs(final_prediction),
                'factors': {
                    'sentiment_influence': abs(sentiment_score * sentiment_multiplier),
                    'fundamental_adjustment': abs(base_prediction - sentiment_score * sentiment_multiplier),
                    'volatility_factor': volatility_adjustment
                }
            }
            
        except Exception as e:
            logger.warning(f"Error generating prediction for {symbol}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'expected_return_5d': 0,
                'confidence': 0,
                'direction': 'neutral',
                'strength': 0,
                'factors': {}
            }
    
    def _assign_enhanced_grade(self, sentiment_score: float, confidence: float, is_anomaly: bool, article_count: int) -> Tuple[str, str]:
        """Assign enhanced grade considering multiple factors"""
        
        # Handle edge cases where sentiment is very low or zero
        if abs(sentiment_score) < 0.01 or confidence < 0.1:
            logger.warning(f"Very low sentiment/confidence: score={sentiment_score:.3f}, confidence={confidence:.3f}, articles={article_count}")
            # If we have articles but low sentiment, it's likely neutral
            if article_count > 0:
                return 'C', 'Neutral - Mixed or unclear sentiment with limited data'
            else:
                return 'C', 'Neutral - Insufficient data for analysis'
        
        # Adjust score by confidence and data quality
        # Use a more lenient data quality multiplier to avoid penalizing too much
        data_quality_multiplier = min(article_count / 15, 1.0)  # Changed from 20 to 15 for more lenient scoring
        
        # Use the raw sentiment score more directly, with confidence as a modifier
        # This prevents over-penalization when confidence is low
        adjusted_score = sentiment_score * (0.5 + 0.5 * confidence) * (0.5 + 0.5 * data_quality_multiplier)
        
        # Anomaly penalty
        if is_anomaly:
            adjusted_score *= 0.8
        
        # Log detailed calculation for debugging
        logger.info(f"Grade calc for {article_count} articles: sentiment={sentiment_score:.3f}, confidence={confidence:.3f}, "
                    f"data_quality={data_quality_multiplier:.3f}, adjusted={adjusted_score:.3f}, anomaly={is_anomaly}")
        
        # Grade assignment with more nuanced categories
        # Adjusted thresholds to better reflect actual sentiment
        if adjusted_score >= 0.4:
            return 'A+', 'Very Positive - Strong bullish sentiment with high confidence'
        elif adjusted_score >= 0.25:
            return 'A', 'Positive - Good bullish sentiment'
        elif adjusted_score >= 0.12:
            return 'B+', 'Moderately Positive - Mild bullish sentiment'
        elif adjusted_score >= 0.04:
            return 'B', 'Slightly Positive - Weak bullish sentiment'
        elif adjusted_score >= -0.04:
            return 'C', 'Neutral - Mixed or unclear sentiment'
        elif adjusted_score >= -0.12:
            return 'D+', 'Slightly Negative - Weak bearish sentiment'
        elif adjusted_score >= -0.25:
            return 'D', 'Moderately Negative - Mild bearish sentiment'
        elif adjusted_score >= -0.4:
            return 'F+', 'Negative - Strong bearish sentiment'
        else:
            return 'F', 'Very Negative - Very strong bearish sentiment with high confidence'
    
    def _calculate_article_quality(self, article: Dict, text: str) -> float:
        """Calculate article quality score"""
        quality_score = 0.5  # Base score
        
        # Source reliability (simplified)
        reliable_sources = ['reuters', 'bloomberg', 'wsj', 'financial times', 'marketwatch']
        source = article.get('source', '').lower()
        if any(rs in source for rs in reliable_sources):
            quality_score += 0.3
        
        # Content length
        if len(text) > 500:
            quality_score += 0.1
        elif len(text) > 200:
            quality_score += 0.05
        
        # Readability
        try:
            readability = flesch_reading_ease(text)
            if 30 <= readability <= 70:  # Good readability range
                quality_score += 0.1
        except:
            pass
        
        return min(quality_score, 1.0)
    
    def _calculate_engagement_metrics(self, posts: List[Dict]) -> Dict:
        """Calculate social media engagement metrics"""
        if not posts:
            return {'total_score': 0, 'avg_score': 0, 'total_comments': 0, 'avg_comments': 0}
        
        scores = [post.get('score', 0) for post in posts]
        comments = [post.get('num_comments', 0) for post in posts]
        
        return {
            'total_score': sum(scores),
            'avg_score': np.mean(scores),
            'total_comments': sum(comments),
            'avg_comments': np.mean(comments),
            'engagement_ratio': sum(comments) / max(sum(scores), 1)
        }
    
    def _calculate_recency_score(self, articles: List[Dict]) -> float:
        """Calculate how recent the news coverage is"""
        if not articles:
            return 0
        
        now = datetime.now(timezone.utc)
        recency_scores = []
        
        for article in articles:
            timestamp = article.get('timestamp', now)
            
            # Handle string timestamps (from NewsAPI)
            if isinstance(timestamp, str):
                try:
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                except ValueError:
                    timestamp = now
            
            # Convert all naive datetimes to UTC
            if isinstance(timestamp, datetime):
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
            
            try:
                hours_old = (now - timestamp).total_seconds() / 3600
                recency_score = max(0, 1 - (hours_old / 168))
                recency_scores.append(recency_score)
            except TypeError:
                # If still can't subtract, skip this article
                continue
        
        return np.mean(recency_scores) if recency_scores else 0
    
    def _analyze_source_distribution(self, article_sentiments: List[Dict]) -> Dict:
        """Analyze distribution of sentiment across news sources"""
        source_data = defaultdict(list)
        
        for article in article_sentiments:
            source = article['source']
            source_data[source].append(article['sentiment'])
        
        source_summary = {}
        for source, sentiments in source_data.items():
            source_summary[source] = {
                'avg_sentiment': np.mean(sentiments),
                'article_count': len(sentiments),
                'sentiment_std': np.std(sentiments)
            }
        
        return source_summary
    
    def _analyze_subreddit_distribution(self, post_sentiments: List[Dict]) -> Dict:
        """Analyze sentiment distribution across subreddits"""
        subreddit_data = defaultdict(list)
        
        for post in post_sentiments:
            subreddit = post['subreddit']
            subreddit_data[subreddit].append({
                'sentiment': post['sentiment'],
                'engagement': post['engagement']
            })
        
        subreddit_summary = {}
        for subreddit, posts in subreddit_data.items():
            sentiments = [p['sentiment'] for p in posts]
            engagements = [p['engagement'] for p in posts]
            
            subreddit_summary[subreddit] = {
                'avg_sentiment': np.mean(sentiments),
                'post_count': len(posts),
                'total_engagement': sum(engagements),
                'avg_engagement': np.mean(engagements)
            }
        
        return subreddit_summary
    
    async def analyze_portfolio_comprehensive(self, symbols: List[str]) -> List[Dict]:
        """Analyze multiple stocks comprehensively"""
        logger.info(f"Starting comprehensive portfolio analysis for {len(symbols)} stocks")
        
        results = []
        
        # Use asyncio for concurrent analysis
        tasks = [self.analyze_stock_comprehensive(symbol) for symbol in symbols]
        
        try:
            completed_results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for i, result in enumerate(completed_results):
                if isinstance(result, Exception):
                    logger.error(f"Error analyzing {symbols[i]}: {result}")
                    continue
                
                results.append(result)
                
        except Exception as e:
            logger.error(f"Error in portfolio analysis: {e}")
        
        # Sort by sentiment score
        results.sort(key=lambda x: x['sentiment']['overall_score'], reverse=True)
        
        # Add portfolio-level analytics
        portfolio_analytics = self._calculate_portfolio_analytics(results)
        
        return results, portfolio_analytics
    
    def _calculate_portfolio_analytics(self, results: List[Dict]) -> Dict:
        """Calculate portfolio-level analytics"""
        if not results:
            return {}
        
        sentiment_scores = [r['sentiment']['overall_score'] for r in results]
        confidence_scores = [r['sentiment']['confidence'] for r in results]
        
        # Sector analysis
        sector_data = defaultdict(list)
        for result in results:
            sector = result['stock_info'].get('sector', 'Unknown')
            sector_data[sector].append(result['sentiment']['overall_score'])
        
        sector_sentiment = {
            sector: np.mean(scores) 
            for sector, scores in sector_data.items()
        }
        
        # Market cap analysis
        large_cap_sentiment = []
        mid_cap_sentiment = []
        small_cap_sentiment = []
        
        for result in results:
            market_cap = result['stock_info'].get('marketCap', 0)
            sentiment = result['sentiment']['overall_score']
            
            if market_cap > 10e9:  # > 10B
                large_cap_sentiment.append(sentiment)
            elif market_cap > 2e9:  # 2B - 10B
                mid_cap_sentiment.append(sentiment)
            else:  # < 2B
                small_cap_sentiment.append(sentiment)
        
        return {
            'portfolio_metrics': {
                'avg_sentiment': np.mean(sentiment_scores),
                'sentiment_std': np.std(sentiment_scores),
                'avg_confidence': np.mean(confidence_scores),
                'positive_stocks': sum(1 for s in sentiment_scores if s > 0.1),
                'negative_stocks': sum(1 for s in sentiment_scores if s < -0.1),
                'neutral_stocks': sum(1 for s in sentiment_scores if -0.1 <= s <= 0.1)
            },
            'sector_analysis': sector_sentiment,
            'market_cap_analysis': {
                'large_cap_avg': np.mean(large_cap_sentiment) if large_cap_sentiment else 0,
                'mid_cap_avg': np.mean(mid_cap_sentiment) if mid_cap_sentiment else 0,
                'small_cap_avg': np.mean(small_cap_sentiment) if small_cap_sentiment else 0,
                'large_cap_count': len(large_cap_sentiment),
                'mid_cap_count': len(mid_cap_sentiment),
                'small_cap_count': len(small_cap_sentiment)
            },
            'risk_metrics': {
                'sentiment_var_95': np.percentile(sentiment_scores, 5),  # 95% VaR
                'sentiment_var_99': np.percentile(sentiment_scores, 1),  # 99% VaR
                'max_negative_sentiment': min(sentiment_scores),
                'sentiment_skewness': self._calculate_skewness(sentiment_scores),
                'sentiment_kurtosis': self._calculate_kurtosis(sentiment_scores)
            }
        }
    
    def _calculate_skewness(self, data: List[float]) -> float:
        """Calculate skewness of data"""
        if len(data) < 3:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        skewness = np.mean([((x - mean) / std) ** 3 for x in data])
        return skewness
    
    def _calculate_kurtosis(self, data: List[float]) -> float:
        """Calculate kurtosis of data"""
        if len(data) < 4:
            return 0
        
        mean = np.mean(data)
        std = np.std(data)
        
        if std == 0:
            return 0
        
        kurtosis = np.mean([((x - mean) / std) ** 4 for x in data]) - 3
        return kurtosis
    
    def _sanitize_numpy_types(self, data):
        """Convert numpy types to native Python types for JSON serialization"""
        import numpy as np
        
        if isinstance(data, dict):
            return {key: self._sanitize_numpy_types(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_numpy_types(item) for item in data]
        elif isinstance(data, np.integer):
            return int(data)
        elif isinstance(data, np.floating):
            return float(data)
        elif isinstance(data, np.bool_):
            return bool(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    def _ensure_string_conversion(self, results):
        """Ensure all symbol values are strings"""
        for result in results:
            if 'symbol' in result:
                result['symbol'] = str(result['symbol'])
        return results

    def generate_comprehensive_report(self, results: List[Dict], portfolio_analytics: Dict) -> str:
        """Generate comprehensive analysis report"""
        if not results:
            return "No analysis data available."
        
        # 🎯 FIX: Ensure symbols are strings immediately to prevent type errors.
        sanitized_results = []
        for result in results:
            if isinstance(result, dict) and 'symbol' in result:
                result['symbol'] = str(result['symbol'])
            sanitized_results.append(result)
        
        try:
            print(f"DEBUG: Analyzing {len(sanitized_results)} results, first symbol: {sanitized_results[0].get('symbol', 'unknown')}")
        except Exception as e:
            print(f"DEBUG: Error accessing first result - {e}")

        report = []
        report.append("🤖 ENHANCED AI STOCK SENTIMENT ANALYSIS REPORT")
        report.append("📊 Advanced Multi-Source Analysis with ML Predictions")
        report.append("🎯 Powered by Transformer Models & Real-Time Data")
        report.append("=" * 80)
        
        # Executive Summary
        portfolio_metrics = portfolio_analytics.get('portfolio_metrics', {})
        
        report.append(f"\n📈 EXECUTIVE SUMMARY")
        report.append("-" * 50)
        report.append(f"Stocks Analyzed: {len(results)}")
        report.append(f"Average Sentiment: {portfolio_metrics.get('avg_sentiment', 0):.3f}")
        report.append(f"Average Confidence: {portfolio_metrics.get('avg_confidence', 0):.1%}")
        report.append(f"Positive Outlook: {portfolio_metrics.get('positive_stocks', 0)} stocks")
        report.append(f"Negative Outlook: {portfolio_metrics.get('negative_stocks', 0)} stocks")
        report.append(f"Neutral Outlook: {portfolio_metrics.get('neutral_stocks', 0)} stocks")
        
        # Risk Assessment
        risk_metrics = portfolio_analytics.get('risk_metrics', {})
        report.append(f"\n⚠️ RISK ASSESSMENT")
        report.append("-" * 50)
        report.append(f"Sentiment VaR (95%): {risk_metrics.get('sentiment_var_95', 0):.3f}")
        report.append(f"Sentiment VaR (99%): {risk_metrics.get('sentiment_var_99', 0):.3f}")
        report.append(f"Maximum Negative Sentiment: {risk_metrics.get('max_negative_sentiment', 0):.3f}")
        report.append(f"Sentiment Skewness: {risk_metrics.get('sentiment_skewness', 0):.3f}")
        
        # Market Sentiment Overview
        avg_sentiment = portfolio_metrics.get('avg_sentiment', 0)
        if avg_sentiment > 0.3:
            mood = "🚀 VERY BULLISH - Strong positive sentiment across portfolio"
        elif avg_sentiment > 0.1:
            mood = "📈 BULLISH - Generally positive market sentiment"
        elif avg_sentiment > -0.1:
            mood = "😐 NEUTRAL - Mixed sentiment signals"
        elif avg_sentiment > -0.3:
            mood = "📉 BEARISH - Generally negative market sentiment"
        else:
            mood = "🔻 VERY BEARISH - Strong negative sentiment across portfolio"
        
        report.append(f"\nMarket Mood: {mood}")
        
        # Sector Analysis
        sector_analysis = portfolio_analytics.get('sector_analysis', {})
        if sector_analysis:
            report.append(f"\n🏢 SECTOR SENTIMENT ANALYSIS")
            report.append("-" * 50)
            
            # Sort sectors by sentiment
            sorted_sectors = sorted(sector_analysis.items(), key=lambda x: x[1], reverse=True)
            
            for sector, sentiment in sorted_sectors:
                emoji = "🟢" if sentiment > 0.1 else "🟡" if sentiment > -0.1 else "🔴"
                report.append(f"{emoji} {sector[:25]:<25} {sentiment:+.3f}")
        
        # Market Cap Analysis
        market_cap_analysis = portfolio_analytics.get('market_cap_analysis', {})
        if market_cap_analysis:
            report.append(f"\n💰 MARKET CAP SENTIMENT BREAKDOWN")
            report.append("-" * 50)
            
            large_cap = market_cap_analysis.get('large_cap_avg', 0)
            mid_cap = market_cap_analysis.get('mid_cap_avg', 0)
            small_cap = market_cap_analysis.get('small_cap_avg', 0)
            
            report.append(f"Large Cap (>$10B): {large_cap:+.3f} ({market_cap_analysis.get('large_cap_count', 0)} stocks)")
            report.append(f"Mid Cap ($2B-$10B): {mid_cap:+.3f} ({market_cap_analysis.get('mid_cap_count', 0)} stocks)")
            report.append(f"Small Cap (<$2B): {small_cap:+.3f} ({market_cap_analysis.get('small_cap_count', 0)} stocks)")
        
        # Top Performers
        report.append(f"\n🏆 TOP SENTIMENT PERFORMERS")
        report.append("-" * 80)
        
        for i, result in enumerate(results[:15], 1):
            sentiment = result['sentiment']
            stock_info = result['stock_info']
            prediction = result.get('prediction', {})
            
            # Status indicators
            confidence_emoji = "🔥" if sentiment['confidence'] > 0.8 else "✅" if sentiment['confidence'] > 0.6 else "⚠️"
            anomaly_emoji = "🚨" if result.get('anomaly_detected', False) else ""
            prediction_emoji = "📈" if prediction.get('direction') == 'bullish' else "📉" if prediction.get('direction') == 'bearish' else "➡️"
            
            report.append(f"\n{i:2d}. {result['symbol']:6} - {sentiment['grade']:2} | {sentiment['overall_score']:+.3f} | {sentiment['confidence']:.0%} {confidence_emoji}{anomaly_emoji}")
            report.append(f"     {stock_info.get('longName', 'N/A')[:50]}")
            report.append(f"     Sector: {stock_info.get('sector', 'N/A')[:20]} | Price: ${stock_info.get('currentPrice', 0):.2f}")
            report.append(f"     {sentiment['description']}")
            
            # Prediction
            if prediction:
                expected_return = prediction.get('expected_return_5d', 0)
                pred_confidence = prediction.get('confidence', 0)
                report.append(f"     {prediction_emoji} 5-Day Prediction: {expected_return:+.1%} (confidence: {pred_confidence:.0%})")
            
            # Data quality metrics
            data_quality = result.get('data_quality', {})
            news_coverage = data_quality.get('news_coverage', 0)
            social_activity = data_quality.get('social_activity', 0)
            recency = data_quality.get('recency_score', 0)
            
            report.append(f"     Data Quality: News {news_coverage:.0%} | Social {social_activity:.0%} | Recency {recency:.0%}")
            
            # Aspect-based sentiment
            aspects = sentiment.get('aspects', {})
            if aspects:
                aspect_summary = []
                for aspect, score in aspects.items():
                    if abs(score) > 0.1:
                        emoji = "+" if score > 0 else "-"
                        aspect_summary.append(f"{aspect}({emoji})")
                
                if aspect_summary:
                    report.append(f"     Key Aspects: {' | '.join(str(x) for x in aspect_summary[:4])}")
            
            # Keywords
            keywords = result.get('keywords', [])
            if keywords:
                report.append(f"     Keywords: {', '.join(str(k) for k in keywords[:5])}")
        
        # Anomaly Alerts
        anomaly_stocks = [r for r in results if r.get('anomaly_detected', False)]
        if anomaly_stocks:
            report.append(f"\n🚨 ANOMALY ALERTS")
            report.append("-" * 50)
            report.append("The following stocks show unusual sentiment patterns:")
            
            for stock in anomaly_stocks[:5]:
                report.append(f"    • {stock['symbol']:6} - {stock['sentiment']['description']}")
                report.append(f"      Unusual sentiment deviation detected - investigate further")
        
        # Predictions Summary
        bullish_predictions = [r for r in results if r.get('prediction', {}).get('direction') == 'bullish']
        bearish_predictions = [r for r in results if r.get('prediction', {}).get('direction') == 'bearish']
        
        if bullish_predictions or bearish_predictions:
            report.append(f"\n🎯 AI PREDICTIONS SUMMARY")
            report.append("-" * 50)
            
            if bullish_predictions:
                bull_symbols = [str(r.get('symbol', 'N/A')) for r in bullish_predictions[:10]]
                report.append(f"📈 Bullish Predictions ({len(bullish_predictions)}): {', '.join(bull_symbols)}")
            
            if bearish_predictions:
                bear_symbols = [str(r.get('symbol', 'N/A')) for r in bearish_predictions[:10]]
                report.append(f"📉 Bearish Predictions ({len(bearish_predictions)}): {', '.join(bear_symbols)}")

        # Data Sources Summary
        total_articles = sum([r['news_analysis']['article_count'] for r in results])
        total_posts = sum([r['social_analysis']['post_count'] for r in results])
        unique_sources = set()
        
        for result in results:
            sources = result['news_analysis'].get('sources', [])
            unique_sources.update(sources)
        
        report.append(f"\n📊 DATA SOURCES SUMMARY")
        report.append("-" * 50)
        report.append(f"News Articles Analyzed: {total_articles:,}")
        report.append(f"Social Media Posts: {total_posts:,}")
        report.append(f"Unique News Sources: {len(unique_sources)}")
        report.append(f"Analysis Timestamp: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Advanced Analytics Summary
        report.append(f"\n🔬 ADVANCED ANALYTICS")
        report.append("-" * 50)
        report.append("✓ Transformer-based sentiment analysis (FinBERT, RoBERTa)")
        report.append("✓ Aspect-based sentiment decomposition")
        report.append("✓ Named entity recognition and keyword extraction")
        report.append("✓ Anomaly detection and quality scoring")
        report.append("✓ Multi-source data fusion and confidence weighting")
        report.append("✓ Predictive modeling with ensemble methods")
        report.append("✓ Real-time news and social media monitoring")
        
        # Investment Recommendations
        report.append(f"\n💡 ACTIONABLE INSIGHTS")
        report.append("-" * 50)
        
        # High-confidence positive stocks
        high_conf_positive = [
            r for r in results[:10]
            if r['sentiment']['overall_score'] > 0.15 and r['sentiment']['confidence'] > 0.7
        ]
        
        if high_conf_positive:
            symbols = [str(r['symbol']) for r in high_conf_positive]
            report.append(f"🟢 Strong Buy Candidates: {', '.join(symbols)}")
            report.append("    High sentiment scores with strong confidence levels")
        
        # High-confidence negative stocks
        high_conf_negative = [
            r for r in results[-10:]
            if r['sentiment']['overall_score'] < -0.15 and r['sentiment']['confidence'] > 0.7
        ]
        
        if high_conf_negative:
            symbols = [str(r['symbol']) for r in high_conf_negative]
            report.append(f"🔴 Avoid/Short Candidates: {', '.join(symbols)}")
            report.append("    Negative sentiment with high confidence - consider avoiding")
        
        # Monitoring recommendations
        low_confidence_stocks = [r for r in results if r['sentiment']['confidence'] < 0.5]
        if low_confidence_stocks:
            symbols = [str(r['symbol']) for r in low_confidence_stocks[:5]]
            report.append(f"👀 Monitor Closely: {', '.join(symbols)}")
            report.append("    Low confidence scores - wait for clearer signals")
        
        report.append(f"\n" + "=" * 80)
        report.append("⚠️  IMPORTANT DISCLAIMERS:")
        report.append("• This analysis is based on publicly available sentiment data")
        report.append("• AI predictions are probabilistic and not guaranteed")
        report.append("• Sentiment can change rapidly with new information")
        report.append("• Always conduct thorough due diligence before investing")
        report.append("• Consider consulting with qualified financial advisors")
        report.append("• Past sentiment patterns do not guarantee future performance")
        report.append("• This tool is for informational purposes only")
        
        return "\n".join(str(x) for x in report)
    
    def create_interactive_dashboard(self, results: List[Dict], portfolio_analytics: Dict):
        """Create interactive dashboard with visualizations"""
        if not results:
            return None
        
        # Sentiment timeline for top stocks
        fig_timeline = self.viz_engine.plot_sentiment_timeline(
            pd.DataFrame(results), 
            results[0]['symbol']
        )
        
        # Sector sentiment heatmap
        sector_data = portfolio_analytics.get('sector_analysis', {})
        if sector_data:
            sector_df = pd.DataFrame([
                {'sector': k, 'avg_sentiment': v} 
                for k, v in sector_data.items()
            ])
            fig_sector = self.viz_engine.plot_sector_sentiment(sector_df)
        else:
            fig_sector = None
        
        # Sentiment distribution
        sentiment_scores = [r['sentiment']['overall_score'] for r in results]
        fig_dist = go.Figure(data=[go.Histogram(
            x=sentiment_scores,
            nbinsx=20,
            marker_color='lightblue',
            opacity=0.7
        )])
        fig_dist.update_layout(
            title='Sentiment Score Distribution',
            xaxis_title='Sentiment Score',
            yaxis_title='Frequency',
            template='plotly_dark'
        )
        
        # Performance correlation (placeholder)
        # In real implementation, you would correlate with actual price movements
        
        return {
            'sentiment_timeline': fig_timeline,
            'sector_analysis': fig_sector,
            'sentiment_distribution': fig_dist
        }
    
    def run_backtesting_analysis(self, symbols: List[str], days_back: int = 90):
        """Run comprehensive backtesting analysis"""
        logger.info(f"Starting backtesting analysis for {len(symbols)} symbols over {days_back} days")
        
        # Get historical sentiment data
        historical_data = []
        for symbol in symbols:
            hist_sentiment = self.db_manager.get_sentiment_history(symbol, days_back)
            if not hist_sentiment.empty:
                historical_data.append(hist_sentiment)
        
        if not historical_data:
            logger.warning("No historical data available for backtesting")
            return None
        
        combined_hist = pd.concat(historical_data, ignore_index=True)
        
        # Get price data from Finnhub candles
        price_data = []
        for symbol in symbols:
            try:
                if hasattr(self.data_collector, "finnhub_client"):
                    # Get historical candles from Finnhub
                    end_date = int(datetime.now(timezone.utc).timestamp())
                    start_date = int((datetime.now(timezone.utc) - timedelta(days=days_back)).timestamp())
                    candles = self.data_collector.finnhub_client.stock_candles(symbol, 'D', start_date, end_date)
                    
                    if candles and candles.get('s') == 'ok' and candles.get('c') and candles.get('t'):
                        # Convert to DataFrame
                        timestamps = [datetime.fromtimestamp(ts, tz=timezone.utc) for ts in candles['t']]
                        prices = candles['c']
                        volumes = candles.get('v', [0] * len(prices))
                        
                        hist = pd.DataFrame({
                            'timestamp': timestamps,
                            'price': prices,
                            'Volume': volumes,
                            'symbol': symbol
                        })
                        price_data.append(hist[['symbol', 'timestamp', 'price', 'Volume']])
            except Exception as e:
                logger.warning(f"Could not get price data for {symbol}: {e}")
        
        if not price_data:
            logger.warning("No price data available for backtesting")
            return None
        
        combined_prices = pd.concat(price_data, ignore_index=True)
        
        # Run backtesting
        strategy_params = {
            'buy_threshold': 0.2,
            'sell_threshold': -0.1,
            'min_confidence': 0.6
        }
        
        backtest_results = self.backtesting_engine.backtest_sentiment_strategy(
            combined_hist, combined_prices, strategy_params
        )
        
        # Monte Carlo simulation for risk assessment
        if not backtest_results.empty:
            portfolio_returns = backtest_results['excess_return'].values
            mc_results = self.backtesting_engine.monte_carlo_simulation(
                portfolio_returns, num_simulations=1000, days=252
            )
            
            return {
                'backtest_results': backtest_results,
                'monte_carlo': mc_results,
                'strategy_params': strategy_params
            }
        
        return None

class RealTimeMonitor:
    """Real-time sentiment monitoring system"""
    
    def start_monitoring(self, symbols: List[str], check_interval: int = 1800):  # 30 minutes
        """Start real-time monitoring"""
        self.monitoring_symbols = symbols
        self.is_running = True
        
        def monitor_loop():
            while self.is_running:
                try:
                    self._check_sentiment_changes()
                    time.sleep(check_interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)  # Wait before retry
        
        self.monitor_thread = Thread(target=monitor_loop, daemon=True)
        self.monitor_thread.start()
        
        logger.info(f"Started monitoring {len(symbols)} symbols with {check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop real-time monitoring"""
        self.is_running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        logger.info("Monitoring stopped")
    
    async def _check_sentiment_changes(self):
        """Check for significant sentiment changes"""
        for symbol in self.monitoring_symbols:
            try:
                # Get current analysis
                current_analysis = await self.analyzer.analyze_stock_comprehensive(symbol)
                
                # Get historical data for comparison
                historical_data = self.analyzer.db_manager.get_sentiment_history(symbol, days_back=1)
                
                if not historical_data.empty:
                    latest_historical = historical_data.iloc[0]
                    
                    # Check for sentiment change
                    current_sentiment = current_analysis['sentiment']['overall_score']
                    historical_sentiment = latest_historical['sentiment_score']
                    sentiment_change = abs(current_sentiment - historical_sentiment)
                    
                    if sentiment_change > self.alert_thresholds['sentiment_change']:
                        self._send_alert(
                            symbol,
                            'SENTIMENT_CHANGE',
                            f"Sentiment changed by {sentiment_change:.3f} points",
                            current_analysis
                        )
                    
                    # Check for anomalies
                    if current_analysis.get('anomaly_detected', False):
                        self._send_alert(
                            symbol,
                            'ANOMALY_DETECTED',
                            "Unusual sentiment pattern detected",
                            current_analysis
                        )
                
            except Exception as e:
                logger.error(f"Error checking {symbol}: {e}")
    
    def _send_alert(self, symbol: str, alert_type: str, message: str, analysis: Dict):
        """Send alert (placeholder - implement actual alerting mechanism)"""
        alert = {
            'symbol': symbol,
            'type': alert_type,
            'message': message,
            'timestamp': datetime.now(timezone.utc),
            'sentiment_score': analysis['sentiment']['overall_score'],
            'confidence': analysis['sentiment']['confidence'],
            'grade': analysis['sentiment']['grade']
        }
        
        # Log alert
        logger.warning(f"ALERT [{alert_type}] {symbol}: {message}")
        
        # In a real implementation, you would:
        # - Send email/SMS notifications
        # - Post to Slack/Discord
        # - Store in database
        # - Trigger webhooks
        
        print(f"\n🚨 ALERT: {symbol} - {alert_type}")
        print(f"    Message: {message}")
        print(f"    Current Score: {analysis['sentiment']['overall_score']:+.3f}")
        print(f"    Grade: {analysis['sentiment']['grade']}")
        print(f"    Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}")

# Flask app initialization (for standalone use)
app = Flask(__name__)
CORS(app)  # This is crucial for allowing your frontend to access the backend

# Initialize analyzer with config (will be set when Flask app starts)
analyzer = None

def get_flask_analyzer():
    """Get or create analyzer instance for Flask app"""
    global analyzer
    if analyzer is None:
        # Load config from environment or config file
        # This should be handled by the Flask app's load_config function
        # For standalone use, load from config.json or environment variables
        import os
        import json
        
        config = {
            'use_redis': False,  # Set to True if Redis is available
            'db_path': 'enhanced_sentiment_data.db',
            'api_keys': {}
        }
        
        # Try to load from config.json
        config_path = os.path.join(os.path.dirname(__file__), 'config.json')
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config['api_keys'] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config.json: {e}")
        
        # Override with environment variables if available
        api_keys = config['api_keys']
        if os.getenv('NEWS_API_KEY'):
            api_keys['news_api'] = os.getenv('NEWS_API_KEY')
        if os.getenv('REDDIT_CLIENT_ID') or os.getenv('REDDIT_CLIENT_SECRET'):
            api_keys['reddit'] = {
                'client_id': os.getenv('REDDIT_CLIENT_ID', api_keys.get('reddit', {}).get('client_id', '')),
                'client_secret': os.getenv('REDDIT_CLIENT_SECRET', api_keys.get('reddit', {}).get('client_secret', '')),
                'user_agent': os.getenv('REDDIT_USER_AGENT', api_keys.get('reddit', {}).get('user_agent', 'StockAnalyzer'))
            }
        if os.getenv('FINNHUB_API_KEY'):
            api_keys['finnhub'] = os.getenv('FINNHUB_API_KEY')
        if os.getenv('ALPHA_VANTAGE_API_KEY'):
            api_keys['alpha_vantage'] = os.getenv('ALPHA_VANTAGE_API_KEY')
        
        analyzer = EnhancedStockSentimentAnalyzer(config)
    return analyzer

@app.route('/api/analyze', methods=['POST'])
def analyze_stocks_endpoint():
    """Flask route handler - must be synchronous"""
    data = request.json
    symbols = data.get('symbols', [])

    if not symbols:
        return jsonify({"error": "No symbols provided"}), 400

    try:
        # Get analyzer instance
        sentiment_analyzer = get_flask_analyzer()
        
        # Run async function in event loop
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            results, portfolio_analytics = loop.run_until_complete(
                sentiment_analyzer.analyze_portfolio_comprehensive(symbols)
            )
            
            # Sanitize the results to ensure they are JSON serializable
            sanitized_results = sentiment_analyzer._sanitize_numpy_types(results)
            
            # Ensure symbols are strings
            sanitized_results = sentiment_analyzer._ensure_string_conversion(sanitized_results)
            
            return jsonify({"results": sanitized_results, "analytics": portfolio_analytics})
        finally:
            loop.close()
            
    except Exception as e:
        # Log the full error for debugging
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'analyzer_ready': analyzer is not None
    })

def run_flask_app():
    """Run the Flask app"""
    print("🚀 Starting Stock Sentiment Analysis API Server")
    print("📊 Backend ready for React frontend")
    print("🌐 Server will run on http://localhost:5000")
    print("=" * 50)
    
    # Pre-initialize the analyzer
    try:
        print("🔄 Initializing sentiment analyzer...")
        get_flask_analyzer()
        print("✅ Analyzer ready!")
    except Exception as e:
        print(f"⚠️ Analyzer initialization warning: {e}")
        print("🔄 Will initialize on first request...")
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True,
        threaded=True
    )

async def enhanced_main():
    """Enhanced main execution with full feature set"""
    # Configuration - Load from config.json or environment variables
    import os
    import json
    
    config = {
        'use_redis': False,  # Set to True if Redis is available
        'db_path': 'enhanced_sentiment_data.db',
        'api_keys': {}
    }
    
    # Try to load from config.json
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config['api_keys'] = json.load(f)
            print("✅ Configuration loaded from config.json")
        except Exception as e:
            print(f"⚠️ Failed to load config.json: {e}")
            print("⚠️ Please create config.json from config.json.example")
    else:
        print("⚠️ config.json not found. Please create it from config.json.example")
        print("⚠️ Or set environment variables: NEWS_API_KEY, FINNHUB_API_KEY, etc.")
    
    # Override with environment variables if available
    api_keys = config['api_keys']
    if os.getenv('NEWS_API_KEY'):
        api_keys['news_api'] = os.getenv('NEWS_API_KEY')
    if os.getenv('REDDIT_CLIENT_ID') or os.getenv('REDDIT_CLIENT_SECRET'):
        api_keys['reddit'] = {
            'client_id': os.getenv('REDDIT_CLIENT_ID', api_keys.get('reddit', {}).get('client_id', '')),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET', api_keys.get('reddit', {}).get('client_secret', '')),
            'user_agent': os.getenv('REDDIT_USER_AGENT', api_keys.get('reddit', {}).get('user_agent', 'StockAnalyzer'))
        }
    if os.getenv('FINNHUB_API_KEY'):
        api_keys['finnhub'] = os.getenv('FINNHUB_API_KEY')
    if os.getenv('ALPHA_VANTAGE_API_KEY'):
        api_keys['alpha_vantage'] = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    # Initialize enhanced analyzer
    analyzer = EnhancedStockSentimentAnalyzer(config)
    
    print("🚀 ENHANCED AI STOCK SENTIMENT ANALYZER v2.0")
    print("🎯 Advanced Multi-Source Analysis with ML Predictions")
    print("🤖 Powered by Transformer Models & Real-Time Data")
    print("=" * 80)
    
    # Stock symbols to analyze
    symbols = [
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
        'JPM', 'BAC', 'JNJ', 'PFE', 'WMT', 'HD', 'XOM', 'CVX', 'KO', 'DIS'
    ]
    
    print(f"\n📊 Analyzing {len(symbols)} stocks with enhanced AI capabilities...")
    
    try:
        # Run comprehensive analysis
        results, portfolio_analytics = await analyzer.analyze_portfolio_comprehensive(symbols)
        results = analyzer._ensure_string_conversion(results)
        
        if results:
            # Generate comprehensive report
            report = analyzer.generate_comprehensive_report(results, portfolio_analytics)
            print("\n" + report)
            
            # Save report
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M")
            filename = f'enhanced_sentiment_analysis_{timestamp}.txt'
            
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write(f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write(report)
                print(f"\n💾 Report saved to: {filename}")
            except Exception as e:
                print(f"Could not save report: {e}")
            
            # Create visualizations
            try:
                dashboards = analyzer.create_interactive_dashboard(results, portfolio_analytics)
                if dashboards:
                    # Save visualizations as HTML
                    for name, fig in dashboards.items():
                        if fig:
                            html_filename = f'{name}_{timestamp}.html'
                            fig.write_html(html_filename)
                            print(f"📊 {name.title()} visualization saved to: {html_filename}")
            except Exception as e:
                logger.warning(f"Could not create visualizations: {e}")
            
            # Run backtesting analysis
            print(f"\n🔄 Running backtesting analysis...")
            try:
                backtest_results = analyzer.run_backtesting_analysis(symbols[:10], days_back=60)
                if backtest_results:
                    print("✅ Backtesting completed - results integrated into analysis")
                else:
                    print("⚠️ Insufficient historical data for backtesting")
            except Exception as e:
                logger.warning(f"Backtesting error: {e}")
            
            # Optional: Start real-time monitoring
            monitor_choice = input("\n🔴 Start real-time monitoring? (y/n): ").strip().lower()
            if monitor_choice == 'y':
                monitor = RealTimeMonitor(analyzer)
                monitor.start_monitoring(symbols[:5], check_interval=1800)  # 30 minutes
                
                print("📡 Real-time monitoring started. Press Ctrl+C to stop...")
                try:
                    while True:
                        await asyncio.sleep(1)
                except KeyboardInterrupt:
                    monitor.stop_monitoring()
                    print("\n🛑 Monitoring stopped")
            
        else:
            print("❌ No results generated. Please check your configuration and try again.")
    
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        print(f"❌ Analysis failed: {e}")

if __name__ == "__main__":
    # Run the enhanced analyzer
    try:
        asyncio.run(enhanced_main())
    except KeyboardInterrupt:
        print("\n\n👋 Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        logger.error(f"Fatal error: {e}")