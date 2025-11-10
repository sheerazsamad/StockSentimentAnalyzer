# Quick Setup Guide

## First Time Setup

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd StockPredictor
   ```

2. **Set up Backend**
   ```bash
   cd stock-sentiment-app/backend
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up Frontend**
   ```bash
   cd ../frontend
   npm install
   ```

4. **Configure API Keys**
   
   Copy the example config:
   ```bash
   cd ../backend
   cp config.json.example config.json
   ```
   
   Edit `config.json` and add your API keys (see README.md for where to get them).

5. **Run the Application**
   
   Terminal 1 (Backend):
   ```bash
   cd stock-sentiment-app/backend
   python3 app.py
   ```
   
   Terminal 2 (Frontend):
   ```bash
   cd stock-sentiment-app/frontend
   npm start
   ```

6. **Open in Browser**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:5000

## Environment Variables (Alternative to config.json)

Instead of using `config.json`, you can set environment variables:

```bash
export NEWS_API_KEY="your_key"
export REDDIT_CLIENT_ID="your_id"
export REDDIT_CLIENT_SECRET="your_secret"
export REDDIT_USER_AGENT="StockAnalyzer"
export FINNHUB_API_KEY="your_key"
export ALPHA_VANTAGE_API_KEY="your_key"
```

## Troubleshooting

- **Backend won't start**: Check that all dependencies are installed and API keys are set
- **Frontend can't connect**: Ensure backend is running on port 5000
- **API errors**: Verify your API keys are correct and not rate-limited

