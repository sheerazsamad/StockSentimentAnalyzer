#!/bin/bash
# Quick test script for Render backend

echo "🧪 Testing Render Backend..."
echo ""

# Test health endpoint
echo "1. Testing health endpoint..."
curl -s https://stocksentimentanalyzer.onrender.com/api/health | python3 -m json.tool
echo ""

# Test analysis endpoint (single stock)
echo "2. Testing analysis endpoint with AAPL..."
curl -s -X POST https://stocksentimentanalyzer.onrender.com/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbols": ["AAPL"]}' \
  | python3 -m json.tool | head -30

echo ""
echo "✅ Test complete!"
