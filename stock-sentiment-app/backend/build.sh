#!/usr/bin/env bash
# Build script for Render deployment
set -e

# Upgrade pip first
pip install --upgrade pip setuptools wheel

# Install requirements
pip install -r requirements.txt

