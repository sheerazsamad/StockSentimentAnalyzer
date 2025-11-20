# Render Data Persistence Guide

## ⚠️ CRITICAL: Preventing Data Loss on Render

This guide explains how to ensure your database data persists across deployments on Render.

## The Problem

When you push updates to Render, if your database is not properly configured, all user data (accounts, favorites, watchlists, analysis history) gets wiped. This happens because:

1. **Ephemeral Database**: Using a temporary database that gets recreated on each deploy
2. **Changing DATABASE_URL**: The database connection string changes between deployments
3. **Unlinked Services**: Database service is not properly linked to your web service

## The Solution: Persistent PostgreSQL Service

### Step 1: Create a Dedicated PostgreSQL Database Service

1. **Go to Render Dashboard**: https://dashboard.render.com
2. **Click "New +"** → **"PostgreSQL"**
3. **Configure the database**:
   - **Name**: `stock-sentiment-db` (or any name you prefer)
   - **Database**: Leave default or set a custom name
   - **User**: Leave default or set a custom user
   - **Region**: Choose the same region as your web service
   - **PostgreSQL Version**: Use the latest stable version
   - **Plan**: Free tier (1 GB) is **more than enough** for this app
   - **Add to Project**: **YES, add it to a project!** (see note below)
4. **Click "Create Database"**

**Important Notes**:
- This creates a **persistent** database that will NOT be deleted when you redeploy your web service
- **1 GB is plenty**: Even with 10,000+ users and millions of analysis entries, you'll use less than 500 MB
- **Adding to a project** helps Render automatically link services and makes management easier

### Step 2: Link Database to Your Web Service

**If you added the database to a project:**
- Render may automatically link it - check your Web Service's **"Environment"** tab for `DATABASE_URL`
- If it's already there, you're done! ✅

**If `DATABASE_URL` is not automatically set:**

1. **Go to your Web Service** (the Flask backend service)
2. **Click on "Settings"** tab
3. Scroll to **"Connections"** section
4. Click **"Link Resource"**
5. Select your PostgreSQL database service
6. Render will automatically add `DATABASE_URL` environment variable

**Verify it worked:**
- Go to **"Environment"** tab
- You should see `DATABASE_URL` listed
- It should look like: `postgresql://user:password@host:port/dbname`

### Step 3: Verify Database Connection

After deployment, check your Render logs. You should see:

```
🔗 Connecting to database: [your-db-name] on [host]
📊 Found X existing users in database
✅ Data persistence verified: X users (was X)
```

If you see:
```
⚠️ CRITICAL: DATA LOSS DETECTED!
```

This means the DATABASE_URL changed or the database was recreated.

### Step 4: Set JWT_SECRET_KEY (Critical for Login Persistence)

Even if your database persists, users will be logged out if `JWT_SECRET_KEY` changes.

1. **Generate a secure key**:
   ```bash
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Set it in Render**:
   - Go to your Web Service → **Environment** tab
   - Click **"Add Environment Variable"**
   - **Key**: `JWT_SECRET_KEY`
   - **Value**: (paste the generated key)
   - **Click "Save Changes"**

**Important**: Never change this key after users start using the app, or all existing login tokens will become invalid.

## How to Check Your Current Setup

### Check 1: Is DATABASE_URL Set?

1. Go to your Web Service → **Environment** tab
2. Look for `DATABASE_URL`
3. It should look like: `postgresql://user:password@host:port/dbname`

### Check 2: Is Database Service Linked?

1. Go to your Web Service → **Settings** tab
2. Scroll to **"Connections"** section
3. You should see your PostgreSQL database listed
4. If not, link it using "Link Resource"

### Check 3: Check Render Logs

After deployment, look for these log messages:

**Good signs**:
- `🔗 Connecting to database: [name] on [host]`
- `📊 Found X existing users in database`
- `✅ Data persistence verified`

**Bad signs**:
- `⚠️ CRITICAL: DATA LOSS DETECTED!`
- `Database connection: local SQLite` (should be PostgreSQL on Render)
- `Database tables initialized: 0 users` (when you know you had users)

## Common Issues and Solutions

### Issue 1: Data Disappears on Every Deploy

**Cause**: Using a temporary database or DATABASE_URL is not set correctly.

**Solution**:
1. Create a dedicated PostgreSQL service (Step 1 above)
2. Link it to your web service (Step 2 above)
3. Verify DATABASE_URL is set in environment variables

### Issue 2: Users Can't Login After Deploy

**Cause**: `JWT_SECRET_KEY` is not set or changed.

**Solution**:
1. Set `JWT_SECRET_KEY` as an environment variable (Step 4 above)
2. Never change it after users start using the app
3. If you must change it, all users will need to log in again

### Issue 3: DATABASE_URL Keeps Changing

**Cause**: Database service is being recreated instead of reused.

**Solution**:
1. Make sure you're using a **dedicated PostgreSQL service** (not ephemeral)
2. Don't delete and recreate the database service
3. Link the same database service to your web service

### Issue 4: "No such table: users" Error

**Cause**: Tables not being created or database connection issue.

**Solution**:
1. Check that `DATABASE_URL` is correctly set
2. Verify database service is running
3. Check logs for database initialization errors
4. The app should automatically create tables on first run

## Testing Data Persistence

1. **Create a test account** on your deployed app
2. **Add some data**: favorite a stock, create a watchlist, analyze a stock
3. **Push a new update** to Render
4. **After deployment completes**, check:
   - Can you still log in with the same account?
   - Are your favorites still there?
   - Is your analysis history still there?

If yes → Data persistence is working! ✅
If no → Follow the steps above to fix it.

## Best Practices

1. **Always use a dedicated PostgreSQL service** (never ephemeral)
2. **Set JWT_SECRET_KEY** before deploying to production
3. **Never delete your database service** unless you want to lose all data
4. **Monitor logs** after each deployment to catch data loss early
5. **Backup your database** regularly (Render free tier doesn't include automatic backups)

## Render Free Tier Limitations

- **Database size**: 1 GB limit
- **No automatic backups**: You need to manually backup if needed
- **Instance hours**: 750 hours/month (enough for one service running 24/7)
- **Database persists**: As long as you don't delete the service

## Need Help?

If you're still experiencing data loss:

1. **Check Render logs** for the error messages mentioned above
2. **Verify DATABASE_URL** is set and points to your PostgreSQL service
3. **Check that database service is linked** to your web service
4. **Ensure JWT_SECRET_KEY** is set and hasn't changed

The app now includes enhanced logging to help diagnose these issues automatically.

