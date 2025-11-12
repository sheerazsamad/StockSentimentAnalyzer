# Stock Sentiment Analyzer

A comprehensive AI-powered stock sentiment analysis application that analyzes stocks using news articles, social media (Reddit), and financial data from multiple sources. The application provides sentiment scores, grades, predictions, and detailed analytics.

## Features

- **Multi-Source Sentiment Analysis**: Analyzes sentiment from news articles and Reddit posts
- **AI-Powered Grading**: Assigns grades (A+ to F) based on sentiment analysis
- **Price Predictions**: Provides 5-day expected return predictions
- **Real-Time Data**: Uses Finnhub and Alpha Vantage APIs for stock data
- **Interactive Frontend**: Modern React-based UI with Tailwind CSS
- **RESTful API**: Flask backend with CORS support

## Tech Stack

### Backend
- Python 3.x
- Flask (REST API)
- Transformers (FinBERT, RoBERTa for sentiment analysis)
- VADER Sentiment Analyzer
- Finnhub API (stock data)
- Alpha Vantage API (financial data)
- NewsAPI (news articles)
- PRAW (Reddit API)

### Frontend
- React 18
- Tailwind CSS
- Lucide React (icons)

## Prerequisites

- Python 3.8+
- Node.js 14+
- npm or yarn
- API Keys for:
  - [NewsAPI](https://newsapi.org/)
  - [Reddit](https://www.reddit.com/prefs/apps) (free)
  - [Finnhub](https://finnhub.io/)
  - [Alpha Vantage](https://www.alphavantage.co/)

## Installation

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd StockPredictor
```

### 2. Backend Setup

```bash
cd stock-sentiment-app/backend

# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Frontend Setup

```bash
cd stock-sentiment-app/frontend

# Install dependencies
npm install
```

### 4. Configuration

#### Option 1: Using config.json (Recommended for development)

1. Copy the example config file:
```bash
cd stock-sentiment-app/backend
cp config.json.example config.json
```

2. Edit `config.json` and add your API keys:
```json
{
  "news_api": "YOUR_NEWS_API_KEY_HERE",
  "reddit": {
    "client_id": "YOUR_REDDIT_CLIENT_ID_HERE",
    "client_secret": "YOUR_REDDIT_CLIENT_SECRET_HERE",
    "user_agent": "StockAnalyzer"
  },
  "finnhub": "YOUR_FINNHUB_API_KEY_HERE",
  "alpha_vantage": "YOUR_ALPHA_VANTAGE_API_KEY_HERE"
}
```

#### Option 2: Using Environment Variables (Recommended for production)

Set the following environment variables:

```bash
export NEWS_API_KEY="your_news_api_key"
export REDDIT_CLIENT_ID="your_reddit_client_id"
export REDDIT_CLIENT_SECRET="your_reddit_client_secret"
export REDDIT_USER_AGENT="StockAnalyzer"
export FINNHUB_API_KEY="your_finnhub_key"
export ALPHA_VANTAGE_API_KEY="your_alpha_vantage_key"
```

## Getting API Keys

### NewsAPI
1. Visit https://newsapi.org/
2. Sign up for a free account
3. Get your API key from the dashboard

### Reddit API (Free)
1. Visit https://www.reddit.com/prefs/apps
2. Click "create another app..."
3. Choose "script" as the app type
4. Note your client_id and client_secret

### Finnhub
1. Visit https://finnhub.io/
2. Sign up for a free account (60 calls/minute)
3. Get your API key from the dashboard

### Alpha Vantage
1. Visit https://www.alphavantage.co/
2. Get a free API key (5 calls/minute, 500/day)
3. Copy your API key

## Running the Application

### Start the Backend

```bash
cd stock-sentiment-app/backend
python3 app.py
```

The backend will start on `http://localhost:5000`

### Start the Frontend

```bash
cd stock-sentiment-app/frontend
npm start
```

The frontend will start on `http://localhost:3000` and automatically open in your browser.

### Running with AI Models (Disable Lightweight Mode)

By default, the application runs in **lightweight mode** using only VADER (a rule-based sentiment analyzer) to optimize memory usage. To enable AI-powered sentiment analysis using FinBERT and RoBERTa transformer models:

1. **Edit your `config.json` file** in `stock-sentiment-app/backend/`:
   ```json
   {
     "news_api": "YOUR_NEWS_API_KEY_HERE",
     "reddit": {
       "client_id": "YOUR_REDDIT_CLIENT_ID_HERE",
       "client_secret": "YOUR_REDDIT_CLIENT_SECRET_HERE",
       "user_agent": "StockAnalyzer"
     },
     "finnhub": "YOUR_FINNHUB_API_KEY_HERE",
     "alpha_vantage": "YOUR_ALPHA_VANTAGE_API_KEY_HERE",
     "lightweight_mode": false
   }
   ```

2. **Restart the backend server**:
   ```bash
   cd stock-sentiment-app/backend
   python3 app.py
   ```

**Note**: 
- Enabling AI models requires **~1-1.5 GB of RAM** (FinBERT ~500-700MB, RoBERTa ~500-700MB)
- First startup will download the models (~1.5 GB total) - this may take a few minutes
- Analysis will be slower but more accurate
- Recommended for local development with sufficient memory
- **Not recommended for Render free tier** (512MB RAM limit)

## Usage

1. Open the application in your browser (http://localhost:3000)
2. Enter stock symbols (e.g., AAPL, MSFT, GOOGL) in the input fields
3. Click "Analyze Stocks"
4. View the sentiment analysis results including:
   - Sentiment grade (A+ to F)
   - Overall sentiment score
   - Price predictions
   - Company information
   - News article count

## API Endpoints

### POST /api/analyze
Analyze one or more stock symbols.

**Request:**
```json
{
  "symbols": ["AAPL", "MSFT", "GOOGL"]
}
```

**Response:**
```json
{
  "results": [
    {
      "symbol": "AAPL",
      "sentiment": {
        "overall_score": 0.407,
        "confidence": 0.441,
        "grade": "A",
        "description": "Positive - Good bullish sentiment"
      },
      "stock_info": {
        "longName": "Apple Inc.",
        "currentPrice": 175.50,
        "sector": "Technology"
      },
      "prediction": {
        "direction": "bullish",
        "expected_return_5d": 0.025,
        "confidence": 0.3
      }
    }
  ],
  "timestamp": "2025-11-10T22:00:00Z",
  "total_analyzed": 1
}
```

### GET /api/health
Health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-11-10T22:00:00Z",
  "analyzer_ready": true
}
```

### GET /api/symbols/<symbol>
Get analysis for a single symbol.

## Project Structure

```
StockPredictor/
├── stock-sentiment-app/
│   ├── backend/
│   │   ├── app.py                    # Flask API server
│   │   ├── enhanced_sentiment_analyzer.py  # Main analyzer class
│   │   ├── config.json.example      # Config template
│   │   ├── requirements.txt         # Python dependencies
│   │   └── .gitignore
│   └── frontend/
│       ├── src/
│       │   ├── App.js               # Main React component
│       │   └── index.js             # React entry point
│       ├── package.json
│       └── .gitignore
├── .gitignore
└── README.md
```

## Security Notes

⚠️ **Important**: Never commit your `config.json` file or API keys to version control. The `.gitignore` file is configured to exclude sensitive files.

- `config.json` is already in `.gitignore`
- Use `config.json.example` as a template
- For production, use environment variables instead of config files

## Troubleshooting

### Backend won't start
- Check that all dependencies are installed: `pip install -r requirements.txt`
- Verify your API keys are set correctly
- Check the logs in `sentiment_analyzer.log`

### Frontend can't connect to backend
- Ensure the backend is running on port 5000
- Check CORS settings in `app.py`
- Verify the frontend proxy settings in `package.json`

### API rate limits
- NewsAPI free tier: 100 requests/day
- Finnhub free tier: 60 calls/minute
- Alpha Vantage free tier: 5 calls/minute, 500/day
- Consider implementing caching to reduce API calls

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- NewsAPI for news data
- Finnhub for stock market data
- Alpha Vantage for financial data
- Reddit for social sentiment data
- Hugging Face Transformers for sentiment analysis models

