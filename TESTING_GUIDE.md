# Testing Guide for Render & Vercel

## 🧪 Testing Render Backend

### Step 1: Check Deployment Status
1. Go to [Render Dashboard](https://dashboard.render.com)
2. Find your backend service (`stocksentimentanalyzer`)
3. Check if it's "Live" (green status)
4. Note your backend URL: `https://stocksentimentanalyzer.onrender.com`

### Step 2: Test Health Endpoint
```bash
curl https://stocksentimentanalyzer.onrender.com/api/health
```

Expected response:
```json
{
  "analyzer_ready": true,
  "status": "healthy",
  "timestamp": "..."
}
```

### Step 3: Test Analysis Endpoint
```bash
curl -X POST https://stocksentimentanalyzer.onrender.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"]}'
```

### Step 4: Check Render Logs
1. In Render dashboard, click on your service
2. Go to "Logs" tab
3. Look for:
   - `Collected 30 news articles` (not 100) ✅
   - `Collected 20 Reddit posts` (not 50) ✅
   - Analysis completes in 30-60 seconds ✅

---

## 🚀 Deploying & Testing on Vercel

### Step 1: Install Vercel CLI (if not installed)
```bash
npm install -g vercel
```

### Step 2: Login to Vercel
```bash
vercel login
```

### Step 3: Deploy Frontend
```bash
cd stock-sentiment-app/frontend
vercel
```

Follow the prompts:
- Set up and deploy? **Yes**
- Which scope? **Your account**
- Link to existing project? **No** (first time) or **Yes** (if redeploying)
- Project name? **stock-sentiment-frontend** (or your choice)
- Directory? **./** (current directory)
- Override settings? **No**

### Step 4: Set Environment Variable in Vercel

**Option A: Via Vercel Dashboard**
1. Go to [Vercel Dashboard](https://vercel.com/dashboard)
2. Click on your project
3. Go to **Settings** → **Environment Variables**
4. Add:
   - **Name**: `REACT_APP_API_URL`
   - **Value**: `https://stocksentimentanalyzer.onrender.com`
   - **Environment**: Production, Preview, Development (select all)
5. Click **Save**

**Option B: Via Vercel CLI**
```bash
vercel env add REACT_APP_API_URL
# When prompted, enter: https://stocksentimentanalyzer.onrender.com
# Select: Production, Preview, Development
```

### Step 5: Redeploy After Setting Environment Variable
```bash
vercel --prod
```

Or trigger redeploy from Vercel dashboard:
1. Go to your project
2. Click **Deployments**
3. Click **...** on latest deployment
4. Click **Redeploy**

---

## ✅ Testing Checklist

### Render Backend Tests
- [ ] Health endpoint returns `{"status": "healthy"}`
- [ ] Analysis endpoint responds within 60 seconds
- [ ] Logs show "Collected 30 news articles" (optimized)
- [ ] Logs show "Collected 20 Reddit posts" (optimized)
- [ ] No timeout errors
- [ ] Cache works (second request is instant)

### Vercel Frontend Tests
- [ ] Frontend loads at your Vercel URL
- [ ] Can enter stock symbol
- [ ] Analysis completes successfully
- [ ] Results display correctly
- [ ] No CORS errors in browser console
- [ ] Environment variable is set correctly

### Integration Tests
- [ ] Frontend connects to Render backend
- [ ] Analysis request completes end-to-end
- [ ] Error handling works (try invalid symbol)
- [ ] Loading states display correctly
- [ ] Results are accurate

---

## 🔍 Quick Test Scripts

### Test Render Backend
```bash
# Health check
curl https://stocksentimentanalyzer.onrender.com/api/health | python3 -m json.tool

# Analysis test
curl -X POST https://stocksentimentanalyzer.onrender.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"]}' | python3 -m json.tool
```

### Test Vercel Frontend
1. Open your Vercel URL in browser
2. Open browser DevTools (F12)
3. Go to **Network** tab
4. Enter a stock symbol and analyze
5. Check:
   - Request goes to Render backend ✅
   - Response is received ✅
   - No CORS errors ✅

---

## 🐛 Troubleshooting

### Render Issues
- **Service not responding**: Check Render logs for errors
- **Timeout errors**: Verify optimizations are deployed (check logs for "30 articles")
- **API errors**: Check API keys in Render environment variables

### Vercel Issues
- **Build fails**: Check build logs in Vercel dashboard
- **Can't connect to backend**: Verify `REACT_APP_API_URL` is set
- **CORS errors**: Backend should have CORS enabled (already configured)

---

## 📝 Notes

- Render free tier may spin down after 15 min inactivity (first request may be slow)
- Vercel automatically redeploys on git push (if connected to GitHub)
- Environment variables in Vercel need redeploy to take effect
- Cache TTL is 2 hours (same symbol within 2 hours = instant response)


