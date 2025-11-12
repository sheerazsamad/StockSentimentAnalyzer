from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import asyncio
import logging
import os
import json
from datetime import datetime, timezone, timedelta
import traceback
from email_validator import validate_email, EmailNotValidError

# Import your existing analyzer
from enhanced_sentiment_analyzer import EnhancedStockSentimentAnalyzer
from models import db, bcrypt, User, FavoriteStock, Watchlist, WatchlistStock

app = Flask(__name__)

# Configure database
basedir = os.path.abspath(os.path.dirname(__file__))
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get('DATABASE_URL') or \
    'sqlite:///' + os.path.join(basedir, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure JWT
app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY') or 'your-secret-key-change-in-production'
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
# Allow OPTIONS requests to bypass JWT (needed for CORS preflight)
app.config['JWT_HEADER_NAME'] = 'Authorization'
app.config['JWT_HEADER_TYPE'] = 'Bearer'

# Initialize extensions
db.init_app(app)
bcrypt.init_app(app)
jwt = JWTManager(app)
# Configure CORS to allow preflight OPTIONS requests
# This must be configured before JWT to allow OPTIONS requests to bypass authentication
# We handle OPTIONS manually, so disable automatic_options
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"],
        "supports_credentials": False
    }
}, automatic_options=False)

# Skip JWT validation for OPTIONS requests (CORS preflight)
# This must run before JWT validation
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        # Create a response and set status explicitly - this stops further processing
        resp = Response(status=200)
        resp.headers['Access-Control-Allow-Origin'] = '*'
        resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
        resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
        resp.headers['Access-Control-Max-Age'] = '3600'
        return resp

# JWT error handlers
@jwt.expired_token_loader
def expired_token_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token has expired'}), 401

@jwt.invalid_token_loader
def invalid_token_callback(error):
    error_msg = str(error)
    logger.error(f"Invalid token error: {error_msg}")
    logger.error(f"Request headers: {dict(request.headers)}")
    # Provide helpful message for common errors
    if 'Subject must be a string' in error_msg:
        return jsonify({
            'error': 'Token format is invalid. Please log out and log back in to get a new token.',
            'msg': error_msg
        }), 422
    return jsonify({'error': f'Invalid token: {error_msg}', 'msg': error_msg}), 422

@jwt.unauthorized_loader
def missing_token_callback(error):
    return jsonify({'error': 'Authorization token is missing'}), 401

@jwt.needs_fresh_token_loader
def token_not_fresh_callback(jwt_header, jwt_payload):
    return jsonify({'error': 'Token is not fresh'}), 401

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

@app.route('/api/register', methods=['POST'])
def register():
    """Register a new user"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        # Validate email
        if not email:
            return jsonify({'error': 'Email is required'}), 400
        
        if not password:
            return jsonify({'error': 'Password is required'}), 400
        
        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400
        
        # Validate email format
        try:
            validate_email(email)
        except EmailNotValidError as e:
            return jsonify({'error': f'Invalid email format: {str(e)}'}), 400
        
        # Check if user already exists
        if User.query.filter_by(email=email).first():
            return jsonify({'error': 'Email already registered'}), 400
        
        # Create new user
        user = User(email=email)
        user.set_password(password)
        
        db.session.add(user)
        db.session.commit()
        
        # Create access token (identity must be a string)
        access_token = create_access_token(identity=str(user.id))
        
        logger.info(f"New user registered: {email}")
        
        return jsonify({
            'message': 'User registered successfully',
            'access_token': access_token,
            'user': user.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Registration error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500

@app.route('/api/login', methods=['POST'])
def login():
    """Login user and return JWT token"""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        email = data.get('email', '').strip().lower()
        password = data.get('password', '')
        
        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400
        
        # Find user
        user = User.query.filter_by(email=email).first()
        
        if not user or not user.check_password(password):
            return jsonify({'error': 'Invalid email or password'}), 401
        
        if not user.is_active:
            return jsonify({'error': 'Account is deactivated'}), 403
        
        # Create access token (identity must be a string)
        access_token = create_access_token(identity=str(user.id))
        
        logger.info(f"User logged in: {email}")
        
        return jsonify({
            'message': 'Login successful',
            'access_token': access_token,
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Login failed: {str(e)}'}), 500

@app.route('/api/me', methods=['GET'])
@jwt_required()
def get_current_user():
    """Get current authenticated user"""
    try:
        user_id = get_jwt_identity()
        # Handle both string and int formats (for backward compatibility)
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        user = User.query.get(user_id_int)
        
        if not user:
            return jsonify({'error': 'User not found'}), 404
        
        return jsonify({
            'user': user.to_dict()
        }), 200
        
    except Exception as e:
        logger.error(f"Get user error: {str(e)}")
        return jsonify({'error': f'Failed to get user: {str(e)}'}), 500

@app.route('/api/analyze', methods=['POST'])
@jwt_required()
def analyze_stocks():
    """API endpoint to analyze stocks"""
    try:
        # Log authorization header for debugging
        auth_header = request.headers.get('Authorization', 'Not provided')
        logger.info(f"Authorization header: {auth_header[:50] if auth_header != 'Not provided' else 'Not provided'}")
        
        # Get request data
        data = request.get_json()
        if not data or 'symbols' not in data:
            return jsonify({'error': 'No symbols provided'}), 400
        
        symbols = data['symbols']
        if not symbols or not isinstance(symbols, list):
            return jsonify({'error': 'Invalid symbols format'}), 400
        
        # Limit to 5 symbols to prevent timeout
        symbols = symbols[:5]
        
        user_id = get_jwt_identity()
        # Handle both string and int formats (for backward compatibility)
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        logger.info(f"User {user_id_int} analyzing symbols: {symbols}")
        
        # Get analyzer instance
        sentiment_analyzer = get_analyzer()
        
        # Run analysis (convert async to sync)
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        results = []
        for symbol in symbols:
            try:
                logger.info(f"About to call analyze_stock_comprehensive for {symbol}")
                # Add timeout to prevent hanging (60 seconds per symbol - optimized for free tier)
                async def analyze_with_timeout():
                    logger.info(f"Inside analyze_with_timeout for {symbol}")
                    return await sentiment_analyzer.analyze_stock_comprehensive(symbol.upper())
                
                try:
                    logger.info(f"Starting async execution for {symbol}")
                    result = loop.run_until_complete(
                        asyncio.wait_for(analyze_with_timeout(), timeout=60.0)
                    )
                    logger.info(f"Async execution completed for {symbol}")
                    
                    # Transform the result to match frontend expectations
                    transformed_result = transform_result(result)
                    results.append(transformed_result)
                    
                except asyncio.TimeoutError:
                    logger.error(f"Analysis timeout for {symbol} after 60 seconds")
                    results.append(create_fallback_result(symbol, "Analysis timed out - API calls may be slow"))
                    
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {str(e)}")
                logger.error(f"Traceback: {traceback.format_exc()}")
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
        # Extract sources with URLs from article breakdown
        sources_list = []
        article_breakdown = backend_result.get('news_analysis', {}).get('sentiment', {}).get('article_breakdown', [])
        
        # Create a map of source -> list of URLs (to deduplicate sources)
        source_url_map = {}
        for article in article_breakdown:
            source = article.get('source', '')
            url = article.get('url', '')
            if source and url:
                if source not in source_url_map:
                    source_url_map[source] = []
                if url not in source_url_map[source]:
                    source_url_map[source].append(url)
        
        # Convert to list of {name, url} objects (use first URL for each source)
        sources_list = [
            {'name': source, 'url': urls[0]} 
            for source, urls in source_url_map.items() 
            if urls
        ]
        
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
                'article_count': int(backend_result['news_analysis'].get('article_count', 0)),
                'sources': sources_list  # Add sources with URLs
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
            'article_count': 0,
            'sources': []
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
@jwt_required()
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

# Explicit OPTIONS handler for favorites endpoints - must be before other routes
@app.route('/api/favorites', methods=['OPTIONS'])
def favorites_options():
    resp = Response(status=200)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    resp.headers['Access-Control-Max-Age'] = '3600'
    return resp

@app.route('/api/favorites/<symbol>', methods=['OPTIONS'])
def favorites_symbol_options(symbol):
    resp = Response(status=200)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    resp.headers['Access-Control-Allow-Methods'] = 'DELETE, OPTIONS'
    resp.headers['Access-Control-Max-Age'] = '3600'
    return resp

@app.route('/api/favorites', methods=['GET'])
@jwt_required()
def get_favorites():
    """Get user's favorite stocks"""
    try:
        user_id = get_jwt_identity()
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        
        favorites = FavoriteStock.query.filter_by(user_id=user_id_int).order_by(FavoriteStock.created_at.desc()).all()
        
        return jsonify({
            'favorites': [fav.to_dict() for fav in favorites]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting favorites: {str(e)}")
        return jsonify({'error': f'Failed to get favorites: {str(e)}'}), 500

@app.route('/api/favorites', methods=['POST'])
@jwt_required()
def add_favorite():
    """Add a stock to user's favorites"""
    try:
        user_id = get_jwt_identity()
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({'error': 'Symbol is required'}), 400
        
        symbol = data['symbol'].upper().strip()
        if not symbol:
            return jsonify({'error': 'Invalid symbol'}), 400
        
        # Check if already favorited
        existing = FavoriteStock.query.filter_by(user_id=user_id_int, symbol=symbol).first()
        if existing:
            return jsonify({
                'message': 'Stock already in favorites',
                'favorite': existing.to_dict()
            }), 200
        
        # Create new favorite
        favorite = FavoriteStock(user_id=user_id_int, symbol=symbol)
        db.session.add(favorite)
        db.session.commit()
        
        logger.info(f"User {user_id_int} added {symbol} to favorites")
        
        return jsonify({
            'message': 'Stock added to favorites',
            'favorite': favorite.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error adding favorite: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Failed to add favorite: {str(e)}'}), 500

@app.route('/api/favorites/<symbol>', methods=['DELETE'])
@jwt_required()
def remove_favorite(symbol):
    """Remove a stock from user's favorites"""
    try:
        user_id = get_jwt_identity()
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        
        symbol = symbol.upper().strip()
        
        favorite = FavoriteStock.query.filter_by(user_id=user_id_int, symbol=symbol).first()
        
        if not favorite:
            return jsonify({'error': 'Stock not in favorites'}), 404
        
        db.session.delete(favorite)
        db.session.commit()
        
        logger.info(f"User {user_id_int} removed {symbol} from favorites")
        
        return jsonify({
            'message': 'Stock removed from favorites'
        }), 200
        
    except Exception as e:
        logger.error(f"Error removing favorite: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Failed to remove favorite: {str(e)}'}), 500

# Watchlist endpoints
@app.route('/api/watchlists', methods=['GET'])
@jwt_required()
def get_watchlists():
    """Get all watchlists for the current user"""
    try:
        user_id = get_jwt_identity()
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        
        watchlists = Watchlist.query.filter_by(user_id=user_id_int).order_by(Watchlist.created_at.desc()).all()
        
        return jsonify({
            'watchlists': [watchlist.to_dict() for watchlist in watchlists]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting watchlists: {str(e)}")
        return jsonify({'error': f'Failed to get watchlists: {str(e)}'}), 500

@app.route('/api/watchlists', methods=['POST'])
@jwt_required()
def create_watchlist():
    """Create a new watchlist"""
    try:
        user_id = get_jwt_identity()
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        
        data = request.get_json()
        if not data or 'name' not in data:
            return jsonify({'error': 'Watchlist name is required'}), 400
        
        name = data['name'].strip()
        if not name:
            return jsonify({'error': 'Watchlist name cannot be empty'}), 400
        
        # Check if watchlist with same name already exists for this user
        existing = Watchlist.query.filter_by(user_id=user_id_int, name=name).first()
        if existing:
            return jsonify({'error': 'Watchlist with this name already exists'}), 400
        
        watchlist = Watchlist(user_id=user_id_int, name=name)
        db.session.add(watchlist)
        db.session.commit()
        
        logger.info(f"User {user_id_int} created watchlist: {name}")
        return jsonify({
            'watchlist': watchlist.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error creating watchlist: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Failed to create watchlist: {str(e)}'}), 500

@app.route('/api/watchlists/<int:watchlist_id>', methods=['DELETE'])
@jwt_required()
def delete_watchlist(watchlist_id):
    """Delete a watchlist"""
    try:
        user_id = get_jwt_identity()
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        
        watchlist = Watchlist.query.filter_by(id=watchlist_id, user_id=user_id_int).first()
        
        if not watchlist:
            return jsonify({'error': 'Watchlist not found'}), 404
        
        db.session.delete(watchlist)
        db.session.commit()
        
        return jsonify({'message': 'Watchlist deleted'}), 200
        
    except Exception as e:
        logger.error(f"Error deleting watchlist: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Failed to delete watchlist: {str(e)}'}), 500

@app.route('/api/watchlists/<int:watchlist_id>/stocks', methods=['GET'])
@jwt_required()
def get_watchlist_stocks(watchlist_id):
    """Get all stocks in a watchlist"""
    try:
        user_id = get_jwt_identity()
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        
        watchlist = Watchlist.query.filter_by(id=watchlist_id, user_id=user_id_int).first()
        
        if not watchlist:
            return jsonify({'error': 'Watchlist not found'}), 404
        
        stocks = WatchlistStock.query.filter_by(watchlist_id=watchlist_id).order_by(WatchlistStock.created_at.desc()).all()
        
        return jsonify({
            'watchlist': watchlist.to_dict(),
            'stocks': [stock.to_dict() for stock in stocks]
        }), 200
        
    except Exception as e:
        logger.error(f"Error getting watchlist stocks: {str(e)}")
        return jsonify({'error': f'Failed to get watchlist stocks: {str(e)}'}), 500

@app.route('/api/watchlists/<int:watchlist_id>/stocks', methods=['POST'])
@jwt_required()
def add_stock_to_watchlist(watchlist_id):
    """Add a stock to a watchlist"""
    try:
        user_id = get_jwt_identity()
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        
        watchlist = Watchlist.query.filter_by(id=watchlist_id, user_id=user_id_int).first()
        
        if not watchlist:
            return jsonify({'error': 'Watchlist not found'}), 404
        
        data = request.get_json()
        if not data or 'symbol' not in data:
            return jsonify({'error': 'Symbol is required'}), 400
        
        symbol = data['symbol'].upper().strip()
        if not symbol:
            return jsonify({'error': 'Invalid symbol'}), 400
        
        # Check if stock already in watchlist
        existing = WatchlistStock.query.filter_by(watchlist_id=watchlist_id, symbol=symbol).first()
        if existing:
            return jsonify({'error': 'Stock already in watchlist'}), 400
        
        watchlist_stock = WatchlistStock(watchlist_id=watchlist_id, symbol=symbol)
        db.session.add(watchlist_stock)
        db.session.commit()
        
        logger.info(f"User {user_id_int} added {symbol} to watchlist {watchlist.name}")
        return jsonify({
            'stock': watchlist_stock.to_dict()
        }), 201
        
    except Exception as e:
        logger.error(f"Error adding stock to watchlist: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Failed to add stock to watchlist: {str(e)}'}), 500

@app.route('/api/watchlists/<int:watchlist_id>/stocks/<symbol>', methods=['DELETE'])
@jwt_required()
def remove_stock_from_watchlist(watchlist_id, symbol):
    """Remove a stock from a watchlist"""
    try:
        user_id = get_jwt_identity()
        user_id_int = int(user_id) if isinstance(user_id, str) else user_id
        
        watchlist = Watchlist.query.filter_by(id=watchlist_id, user_id=user_id_int).first()
        
        if not watchlist:
            return jsonify({'error': 'Watchlist not found'}), 404
        
        symbol = symbol.upper().strip()
        watchlist_stock = WatchlistStock.query.filter_by(watchlist_id=watchlist_id, symbol=symbol).first()
        
        if not watchlist_stock:
            return jsonify({'error': 'Stock not in watchlist'}), 404
        
        db.session.delete(watchlist_stock)
        db.session.commit()
        
        return jsonify({'message': 'Stock removed from watchlist'}), 200
        
    except Exception as e:
        logger.error(f"Error removing stock from watchlist: {str(e)}")
        db.session.rollback()
        return jsonify({'error': f'Failed to remove stock from watchlist: {str(e)}'}), 500

# Explicit OPTIONS handler for watchlist endpoints
@app.route('/api/watchlists', methods=['OPTIONS'])
@app.route('/api/watchlists/<int:watchlist_id>', methods=['OPTIONS'])
@app.route('/api/watchlists/<int:watchlist_id>/stocks', methods=['OPTIONS'])
@app.route('/api/watchlists/<int:watchlist_id>/stocks/<symbol>', methods=['OPTIONS'])
def watchlists_options(watchlist_id=None, symbol=None):
    resp = Response(status=200)
    resp.headers['Access-Control-Allow-Origin'] = '*'
    resp.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization'
    resp.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
    resp.headers['Access-Control-Max-Age'] = '3600'
    return resp

if __name__ == '__main__':
    print("🚀 Starting Stock Sentiment Analysis API Server")
    print("📊 Backend ready for React frontend")
    print("🔐 Authentication enabled")
    print("🌐 Server will run on http://localhost:5000")
    print("=" * 50)
    
    # Initialize database
    with app.app_context():
        db.create_all()
        print("✅ Database initialized")
    
    # Pre-initialize the analyzer
    try:
        print("🔄 Initializing sentiment analyzer...")
        get_analyzer()
        print("✅ Analyzer ready!")
    except Exception as e:
        print(f"⚠️ Analyzer initialization warning: {e}")
        print("🔄 Will initialize on first request...")
    
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)