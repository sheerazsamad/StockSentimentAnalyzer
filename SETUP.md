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

7. **Create an Account**
   - When you first open the app, you'll see a login screen
   - Click "Sign up" to create a new account with your email and password
   - After registration, you'll be automatically logged in

## Authentication Setup

The application now requires user accounts. Users can register and login with email and password.

**First time setup:**
- The database (`users.db`) will be automatically created when you first run the backend
- No additional setup needed for development

**Production setup:**
- Set a secure JWT secret key as an environment variable:
  ```bash
  export JWT_SECRET_KEY="your-very-secure-random-secret-key-here"
  ```
- For production, use a proper database (PostgreSQL, MySQL) instead of SQLite:
  ```bash
  export DATABASE_URL="postgresql://user:password@localhost/dbname"
  ```

## Environment Variables (Alternative to config.json)

Instead of using `config.json`, you can set environment variables:

```bash
export NEWS_API_KEY="your_key"
export REDDIT_CLIENT_ID="your_id"
export REDDIT_CLIENT_SECRET="your_secret"
export REDDIT_USER_AGENT="StockAnalyzer"
export FINNHUB_API_KEY="your_key"
export ALPHA_VANTAGE_API_KEY="your_key"
export JWT_SECRET_KEY="your-secure-secret-key"  # Required for authentication
```

## Render Deployment - Database Persistence

**⚠️ IMPORTANT: Preventing Data Loss on Redeploy**

To ensure your database persists across deployments on Render:

1. **Create a PostgreSQL Database Service** (not just use DATABASE_URL):
   - Go to Render Dashboard → New → PostgreSQL
   - Create a new PostgreSQL database
   - Note: This creates a **persistent** database that won't be deleted on redeploy

2. **Link Database to Your Web Service**:
   - Go to your Web Service settings
   - Under "Environment", you should see `DATABASE_URL` automatically set
   - If not, manually add it: `DATABASE_URL` = (copy from your PostgreSQL service)

3. **Verify Database Connection**:
   - Check Render logs after deployment
   - You should see: `Database connection: [your-database-host]`
   - If you see errors about database reset, the DATABASE_URL might be wrong

4. **Common Issues**:
   - **Data disappears on redeploy**: You're likely using a temporary database or DATABASE_URL is changing
   - **Solution**: Create a dedicated PostgreSQL service and link it to your web service
   - **Check logs**: The app now logs warnings if it detects data loss

5. **Environment Variables on Render**:
   - `DATABASE_URL`: Automatically set when you link a PostgreSQL service
   - `JWT_SECRET_KEY`: **MUST be set manually** (see Authentication Setup above)
   - Other API keys: Set in Environment tab of your web service

## Troubleshooting

- **Backend won't start**: Check that all dependencies are installed and API keys are set
- **Frontend can't connect**: Ensure backend is running on port 5000
- **API errors**: Verify your API keys are correct and not rate-limited
- **Data lost on redeploy**: See "Render Deployment - Database Persistence" section above


