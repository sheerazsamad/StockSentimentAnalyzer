"""
Configuration template for Stock Sentiment Analyzer
Copy this file to config.py and add your actual API keys
"""

# Main configuration
CONFIG = {
    'use_redis': False,  # Set to True if you have Redis installed
    'db_path': 'enhanced_sentiment_data.db',
    
    'api_keys': {
        # Get from: https://newsapi.org/
        'news_api': 'YOUR_NEWS_API_KEY_HERE',
        
        # Reddit API - Completely free
        # Get from: https://www.reddit.com/prefs/apps
        'reddit': {
            'client_id': 'YOUR_REDDIT_CLIENT_ID_HERE',
            'client_secret': 'YOUR_REDDIT_CLIENT_SECRET_HERE', 
            'user_agent': 'StockSentimentAnalyzer/1.0'
        },
        
        # Finnhub API - Free tier: 60 calls/minute
        # Get from: https://finnhub.io/
        'finnhub': 'YOUR_FINNHUB_API_KEY_HERE',
        
        # Alpha Vantage API - Free tier: 5 calls/minute, 500/day
        # Get from: https://www.alphavantage.co/
        'alpha_vantage': 'YOUR_ALPHA_VANTAGE_API_KEY_HERE'
    },
    
    # Analysis settings
    'sentiment_weights': {
        'news': 0.5,      # Weight for news sources
        'social': 0.3,    # Weight for social media
        'analyst': 0.2    # Weight for analyst reports
    },
    
    # Confidence thresholds
    'confidence_thresholds': {
        'high': 0.8,
        'medium': 0.6,
        'low': 0.4
    },
    
    # Rate limiting (requests per minute)
    'rate_limits': {
        'news_api': 100,      # NewsAPI allows more on paid plans
        'finnhub': 60,        # Free tier limit
        'reddit': 60,         # Conservative limit
        'alpha_vantage': 5    # Free tier limit
    }
}