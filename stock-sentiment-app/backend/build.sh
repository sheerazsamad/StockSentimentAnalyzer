#!/usr/bin/env bash
# Build script for Render deployment
set -e

echo "🔧 Installing build tools first..."
# Install setuptools and wheel FIRST, before anything else
python -m pip install --upgrade pip
python -m pip install --no-cache-dir setuptools>=65.5.0 wheel

echo "📦 Installing requirements..."
# Now install everything else
python -m pip install --no-cache-dir -r requirements.txt

