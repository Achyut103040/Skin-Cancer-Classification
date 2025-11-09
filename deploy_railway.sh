#!/bin/bash

echo "🚀 Quick Railway Deployment Script"
echo "===================================="
echo ""

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null
then
    echo "📦 Railway CLI not found. Installing..."
    npm i -g @railway/cli
else
    echo "✅ Railway CLI already installed"
fi

echo ""
echo "🔐 Logging into Railway..."
railway login

echo ""
echo "📂 Initializing Railway project..."
railway init

echo ""
echo "⚙️ Setting environment variables..."
railway variables set PYTHON_VERSION=3.11.0
railway variables set MODEL_PATH=./models

echo ""
echo "🚀 Deploying to Railway..."
railway up

echo ""
echo "✅ Deployment complete!"
echo "🌐 Your app will be available at the Railway-provided URL"
echo ""
echo "📊 To check status: railway status"
echo "📝 To view logs: railway logs"
