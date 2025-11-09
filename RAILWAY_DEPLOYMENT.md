# 🚀 Railway Deployment Guide - 5GB Storage!

## Why Railway?
- ✅ **5GB Persistent Storage** (vs Render's 512MB)
- ✅ $5/month free credits (enough for small ML apps)
- ✅ Automatic GitHub deployments
- ✅ Easy environment variables
- ✅ Better for ML models with Google Drive downloads

---

## Quick Start (3 Methods)

### Method 1: One-Click Deploy (Easiest) ⭐

1. **Visit Railway**: https://railway.app
2. **Sign up** with GitHub
3. **Click**: "New Project" → "Deploy from GitHub repo"
4. **Select**: `Achyut103040/Skin-Cancer-Classification`
5. **Add Variables**:
   - `PORT`: 8080
   - `PYTHON_VERSION`: 3.11.0
   - `MODEL_PATH`: ./models
6. **Set Start Command**:
   ```
   gunicorn web_interface.enhanced_app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
   ```
7. **Deploy!** 🚀

---

### Method 2: Railway CLI (Windows)

```cmd
# Run the deployment script
deploy_railway.bat
```

Or manually:
```cmd
# Install Railway CLI
npm i -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Set environment variables
railway variables set PYTHON_VERSION=3.11.0
railway variables set MODEL_PATH=./models

# Deploy
railway up
```

---

### Method 3: Dockerfile Deploy (Advanced)

Railway auto-detects and builds from your requirements.txt, but for full control:

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements_flask_deploy.txt .
RUN pip install --no-cache-dir -r requirements_flask_deploy.txt

# Copy application
COPY . .

# Create models directory
RUN mkdir -p models

# Expose port
EXPOSE 8080

# Start command
CMD gunicorn web_interface.enhanced_app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

---

## Railway Configuration

### Environment Variables
Set these in Railway dashboard:
```
PYTHON_VERSION=3.11.0
MODEL_PATH=./models
PORT=8080
```

### Build Command (automatic)
Railway auto-detects Python and runs:
```bash
pip install -r requirements_flask_deploy.txt
```

### Start Command
```bash
gunicorn web_interface.enhanced_app:app --bind 0.0.0.0:$PORT --workers 2 --timeout 120
```

---

## Storage Management

### Railway Volumes (Persistent Storage)
To persist models across deploys:

1. Go to your Railway project
2. Click "Variables" → "Add Volume"
3. Mount point: `/app/models`
4. Size: 5GB (free tier)

This way models stay cached and don't re-download!

---

## Alternative: Fly.io (3GB Storage)

If Railway doesn't work, try Fly.io:

```cmd
# Install Fly CLI (Windows)
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Login
fly auth login

# Create app
fly launch

# Deploy
fly deploy
```

**fly.toml** (create this):
```toml
app = "skin-cancer-detection"

[build]
  builder = "paketobuildpacks/builder:base"

[env]
  PORT = "8080"

[http_service]
  internal_port = 8080
  force_https = true
  auto_stop_machines = true
  auto_start_machines = true

[[vm]]
  cpu_kind = "shared"
  cpus = 1
  memory_mb = 512

[mounts]
  source = "models_volume"
  destination = "/app/models"
  initial_size = "3gb"
```

---

## Alternative: Koyeb (2.5GB Storage)

1. Visit https://koyeb.com
2. Sign up with GitHub
3. "Create Service" → GitHub
4. Select repo: `Achyut103040/Skin-Cancer-Classification`
5. Build command: `pip install -r requirements_flask_deploy.txt`
6. Run command: `gunicorn web_interface.enhanced_app:app --bind 0.0.0.0:$PORT`
7. Deploy!

---

## Troubleshooting

### Out of Memory?
Reduce model loading in `enhanced_app.py`:
```python
LAZY_LOAD_MODELS = True  # Load models on-demand
```

### Out of Storage?
Use Google Drive for models (already configured):
```python
USE_GDRIVE_MODELS = True  # Download from Drive, not local
```

### Slow Cold Starts?
Railway keeps your app warm better than Render. But if needed:
- Use smaller models
- Enable lazy loading
- Use Railway's "Always On" (requires paid plan)

---

## Cost Comparison

| Platform | Storage | RAM | Free Tier | Best For |
|----------|---------|-----|-----------|----------|
| **Railway** | 5GB | 512MB | $5/month credits | ML models ⭐ |
| Render | 512MB | 512MB | Free forever | Small apps |
| Fly.io | 3GB | 256MB | Free tier | Docker apps |
| Koyeb | 2.5GB | 512MB | Free tier | Simple deploys |
| Hugging Face | N/A | 16GB | Free forever | Gradio only |

---

## Recommended Setup

**For your Flask app with 6 models (~400MB):**

1. **Use Railway** (5GB storage)
2. **Enable persistent volume** for models
3. **Use Google Drive downloads** (already configured)
4. **Set lazy loading** to save RAM

This setup will:
- ✅ Handle all 6 models (400MB total)
- ✅ Cache models to avoid re-downloads
- ✅ Stay within free tier
- ✅ Fast response times

---

## Next Steps

1. **Choose platform**: Railway (recommended)
2. **Run deployment script**: `deploy_railway.bat`
3. **Wait for build**: ~5-10 minutes
4. **Test your app**: Railway provides URL
5. **Monitor usage**: Check Railway dashboard

Need help? Let me know which platform you want to try!
