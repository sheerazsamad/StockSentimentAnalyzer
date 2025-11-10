from flask import Flask, request, jsonify
from flask_cors import CORS
import asyncio
import logging
import os
import json
from datetime import datetime, timezone
import traceback

# Import your existing analyzer
from enhanced_sentiment_analyzer import EnhancedStockSentimentAnalyzer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the analyzer once when the server starts
analyzer = None

def load_config():
    """Load configuration from config.json or environment variables"""
    config = {
        'use_redis': False,
        'db_path': 'sentiment_data.db',
        'api_keys': {}
    }
    
    # Try to load from config.json first
    config_path = os.path.join(os.path.dirname(__file__), 'config.json')
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                file_config = json.load(f)
                config['api_keys'] = file_config
                logger.info("Configuration loaded from config.json")
        except Exception as e:
            logger.warning(f"Failed to load config.json: {e}")
    
    # Override with environment variables if available
    api_keys = config['api_keys']
    
    # News API
    if os.getenv('NEWS_API_KEY'):
        api_keys['news_api'] = os.getenv('NEWS_API_KEY')
    
    # Reddit API
    if os.getenv('REDDIT_CLIENT_ID') or os.getenv('REDDIT_CLIENT_SECRET'):
        api_keys['reddit'] = {
            'client_id': os.getenv('REDDIT_CLIENT_ID', api_keys.get('reddit', {}).get('client_id', '')),
            'client_secret': os.getenv('REDDIT_CLIENT_SECRET', api_keys.get('reddit', {}).get('client_secret', '')),
            'user_agent': os.getenv('REDDIT_USER_AGENT', api_keys.get('reddit', {}).get('user_agent', 'StockAnalyzer'))
        }
    
    # Finnhub API
    if os.getenv('FINNHUB_API_KEY'):
        api_keys['finnhub'] = os.getenv('FINNHUB_API_KEY')
    
    # Alpha Vantage API
    if os.getenv('ALPHA_VANTAGE_API_KEY'):
        api_keys['alpha_vantage'] = os.getenv('ALPHA_VANTAGE_API_KEY')
    
    # Validate that we have at least some API keys
    if not api_keys.get('news_api') and not api_keys.get('finnhub'):
        logger.warning("No API keys found! Please set up config.json or environment variables.")
        logger.warning("See config.json.example for template.")
    
    return config

def get_analyzer():
    """Get or create analyzer instance"""
    global analyzer
    if analyzer is None:
        config = load_config()
        analyzer = EnhancedStockSentimentAnalyzer(config)
    return analyzer

@app.route('/api/analyze', methods=['POST'])
def analyze_stocks():
    """API endpoint to analyze stocks"""
    try:
        # Get request data
        data = request.get_json()
        if not data or 'symbols' not in data:
            return jsonify({'error': 'No symbols provided'}), 400
        
        symbols = data['symbols']
        if not symbols or not isinstance(symbols, list):
            return jsonify({'error': 'Invalid symbols format'}), 400
        
        # Limit to 5 symbols to prevent timeout
        symbols = symbols[:5]
        
        logger.info(f"Analyzing symbols: {symbols}")
        
        # Get analyzer instance
        sentiment_analyzer = get_analyzer()
        
        # Run analysis (convert async to sync)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = []
        for symbol in symbols:
            try:
                # Analyze each symbol individually to handle errors gracefully
                result = loop.run_until_complete(
                    sentiment_analyzer.analyze_stock_comprehensive(symbol.upper())
                )
                
                # Transform the result to match frontend expectations
                transformed_result = transform_result(result)
                results.append(transformed_result)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                # Add a fallback result for failed analysis
                results.append(create_fallback_result(symbol, str(e)))
        
        loop.close()
        
        # Return results in expected format
        response = {
            'results': results,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_analyzed': len(results)
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

def transform_result(backend_result):
    """Transform backend result to frontend format"""
    try:
        return {
            'symbol': str(backend_result.get('symbol', '')),
            'sentiment': {
                'overall_score': float(backend_result['sentiment']['overall_score']),
                'confidence': float(backend_result['sentiment']['confidence']),
                'grade': str(backend_result['sentiment']['grade']),
                'description': str(backend_result['sentiment']['description'])
            },
            'stock_info': {
                'longName': str(backend_result['stock_info'].get('longName', '')),
                'currentPrice': float(backend_result['stock_info'].get('currentPrice', 0)),
                'sector': str(backend_result['stock_info'].get('sector', ''))
            },
            'news_analysis': {
                'article_count': int(backend_result['news_analysis'].get('article_count', 0))
            },
            'prediction': {
                'direction': str(backend_result['prediction'].get('direction', 'neutral')),
                'expected_return_5d': float(backend_result['prediction'].get('expected_return_5d', 0)),
                'confidence': float(backend_result['prediction'].get('confidence', 0))
            },
            'keywords': [str(k) for k in backend_result.get('keywords', [])][:10]
        }
    except Exception as e:
        logger.error(f"Error transforming result: {e}")
        return create_fallback_result(backend_result.get('symbol', 'UNKNOWN'), str(e))

def create_fallback_result(symbol, error_msg):
    """Create a fallback result when analysis fails"""
    return {
        'symbol': str(symbol),
        'sentiment': {
            'overall_score': 0.0,
            'confidence': 0.0,
            'grade': 'N/A',
            'description': f'Analysis failed: {error_msg}'
        },
        'stock_info': {
            'longName': f'{symbol} Corporation',
            'currentPrice': 0.0,
            'sector': 'Unknown'
        },
        'news_analysis': {
            'article_count': 0
        },
        'prediction': {
            'direction': 'neutral',
            'expected_return_5d': 0.0,
            'confidence': 0.0
        },
        'keywords': []
    }

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'analyzer_ready': analyzer is not None
    })

@app.route('/api/symbols/<symbol>', methods=['GET'])
def get_single_symbol(symbol):
    """Get analysis for a single symbol"""
    try:
        sentiment_analyzer = get_analyzer()
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        result = loop.run_until_complete(
            sentiment_analyzer.analyze_stock_comprehensive(symbol.upper())
        )
        
        loop.close()
        
        transformed_result = transform_result(result)
        return jsonify(transformed_result)
        
    except Exception as e:
        logger.error(f"Error analyzing {symbol}: {str(e)}")
        return jsonify(create_fallback_result(symbol, str(e)))

if __name__ == '__main__':
    print("🚀 Starting Stock Sentiment Analysis API Server")
    print("📊 Backend ready for React frontend")
    print("🌐 Server will run on http://localhost:5000")
    print("=" * 50)
    
    # Pre-initialize the analyzer
    try:
        print("🔄 Initializing sentiment analyzer...")
        get_analyzer()
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