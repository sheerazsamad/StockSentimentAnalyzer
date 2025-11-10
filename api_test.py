import requests
import praw
import finnhub
from newsapi import NewsApiClient
from alpha_vantage.timeseries import TimeSeries
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_api_keys():
    """Tests API keys for NewsAPI, Reddit, Finnhub, and Alpha Vantage."""

    # Load config from config.json or use environment variables
    import os
    import json
    
    config = {
        'use_redis': False,
        'db_path': 'enhanced_sentiment_data.db',
        'api_keys': {}
    }
    
    # Try to load from config.json
    config_path = os.path.join('stock-sentiment-app', 'backend', 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config['api_keys'] = json.load(f)
        except Exception as e:
            logging.warning(f"Failed to load config.json: {e}")
    
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
    
    if not api_keys.get('news_api') and not api_keys.get('finnhub'):
        logging.error("❌ No API keys found! Please set up config.json or environment variables.")
        logging.error("See stock-sentiment-app/backend/config.json.example for template.")
        return

    print("🚀 Starting API Key Test...")

    # ------------------- NewsAPI Test -------------------
    logging.info("Testing NewsAPI...")
    try:
        newsapi = NewsApiClient(api_key=config['api_keys']['news_api'])
        top_headlines = newsapi.get_top_headlines(country='us', page_size=1)
        if top_headlines['status'] == 'ok':
            logging.info("✅ NewsAPI key is working.")
        else:
            logging.error(f"❌ NewsAPI key failed. Status: {top_headlines.get('status', 'N/A')}")
    except Exception as e:
        logging.error(f"❌ NewsAPI key failed. Error: {e}")

    # ------------------- Reddit API Test -------------------
    logging.info("Testing Reddit API...")
    try:
        reddit = praw.Reddit(
            client_id=config['api_keys']['reddit']['client_id'],
            client_secret=config['api_keys']['reddit']['client_secret'],
            user_agent=config['api_keys']['reddit']['user_agent']
        )
        # Attempt to access a public subreddit to test the connection
        _ = reddit.subreddit('popular').hot(limit=1)
        logging.info("✅ Reddit API key is working.")
    except praw.exceptions.PRAWException as e:
        logging.error(f"❌ Reddit API key failed. Check client_id and client_secret. Error: {e}")
    except Exception as e:
        logging.error(f"❌ Reddit API key failed. Check client_id, client_secret, or user_agent. Error: {e}")

    # ------------------- Finnhub API Test -------------------
    logging.info("Testing Finnhub API...")
    try:
        finnhub_client = finnhub.Client(api_key=config['api_keys']['finnhub'])
        _ = finnhub_client.symbol_lookup('AAPL')
        logging.info("✅ Finnhub key is working.")
    except finnhub.FinnhubAPIException as e:
        logging.error(f"❌ Finnhub key failed. Check the key. Error: {e}")
    except Exception as e:
        logging.error(f"❌ Finnhub key failed. Error: {e}")

    # ------------------- Alpha Vantage API Test -------------------
    logging.info("Testing Alpha Vantage API...")
    try:
        ts = TimeSeries(key=config['api_keys']['alpha_vantage'], output_format='json')
        _ = ts.get_intraday(symbol='AAPL', interval='5min')
        logging.info("✅ Alpha Vantage key is working.")
    except Exception as e:
        logging.error(f"❌ Alpha Vantage key failed. Error: {e}")

if __name__ == "__main__":
    test_api_keys()