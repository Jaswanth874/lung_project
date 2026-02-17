# Deployment Guide

This guide explains how to deploy the Lung Nodule Detection System to various cloud platforms.

## 🚀 Quick Deployment Options

### Option 1: Render (Recommended - Free Tier Available)

**Steps:**

1. **Sign up/Login**: Go to [render.com](https://render.com) and sign up/login

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your GitHub repository: `Jaswanth874/lung_project`
   - Select the repository

3. **Configure Settings**:
   - **Name**: `lung-nodule-detection` (or your choice)
   - **Region**: Choose closest to you
   - **Branch**: `main`
   - **Root Directory**: Leave empty (or `ld1` if needed)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements-deploy.txt`
   - **Start Command**: `gunicorn wsgi:app --bind 0.0.0.0:$PORT --workers 2 --threads 2`

4. **Environment Variables**:
   - `FLASK_SECRET`: Generate a random secret key
   - `FLASK_PORT`: `5000` (Render sets PORT automatically)
   - `FLASK_DEBUG`: `False`
   - `GOOGLE_API_KEY`: Your Gemini API key (optional, for RAG)

5. **Deploy**: Click "Create Web Service"

6. **Your URL**: Will be `https://lung-nodule-detection.onrender.com` (or your custom name)

---

### Option 2: Railway (Easy & Fast)

**Steps:**

1. **Sign up**: Go to [railway.app](https://railway.app) and sign up with GitHub

2. **New Project**:
   - Click "New Project"
   - Select "Deploy from GitHub repo"
   - Choose `Jaswanth874/lung_project`

3. **Configure**:
   - Railway auto-detects Python
   - Uses `railway.json` configuration
   - Sets up environment automatically

4. **Environment Variables**:
   - Add `FLASK_SECRET` (generate random key)
   - Add `GOOGLE_API_KEY` (optional)

5. **Deploy**: Railway auto-deploys on push

6. **Your URL**: Railway provides a URL like `https://lung-project-production.up.railway.app`

---

### Option 3: Heroku (Classic Platform)

**Steps:**

1. **Install Heroku CLI**: [heroku.com/cli](https://devcenter.heroku.com/articles/heroku-cli)

2. **Login**:
   ```bash
   heroku login
   ```

3. **Create App**:
   ```bash
   heroku create lung-nodule-detection
   ```

4. **Set Environment Variables**:
   ```bash
   heroku config:set FLASK_SECRET=your-secret-key
   heroku config:set FLASK_DEBUG=False
   heroku config:set GOOGLE_API_KEY=your-key
   ```

5. **Deploy**:
   ```bash
   git push heroku main
   ```

6. **Open**:
   ```bash
   heroku open
   ```

---

### Option 4: PythonAnywhere (Free Tier Available)

**Steps:**

1. **Sign up**: [pythonanywhere.com](https://www.pythonanywhere.com)

2. **Upload Files**:
   - Go to "Files" tab
   - Upload your project files

3. **Create Web App**:
   - Go to "Web" tab
   - Click "Add a new web app"
   - Choose Flask and Python 3.9

4. **Configure**:
   - Set source code directory
   - Set WSGI file: `/home/yourusername/lung_project/wsgi.py`

5. **Reload**: Click "Reload" button

---

## 📋 Pre-Deployment Checklist

- [ ] Update `requirements-deploy.txt` with all dependencies
- [ ] Set `FLASK_DEBUG=False` in production
- [ ] Generate a secure `FLASK_SECRET` key
- [ ] Ensure `.env` file is NOT committed (in .gitignore)
- [ ] Test locally with `gunicorn wsgi:app`
- [ ] Update database path if needed (SQLite works on most platforms)

## 🔧 Production Configuration

### Environment Variables

Set these in your hosting platform:

```bash
FLASK_SECRET=your-random-secret-key-here
FLASK_DEBUG=False
FLASK_PORT=5000
GOOGLE_API_KEY=your-gemini-api-key  # Optional
```

### Generate Secret Key

```python
import secrets
print(secrets.token_hex(32))
```

## 🐛 Troubleshooting

### Issue: App crashes on startup
- Check logs in your hosting platform
- Ensure all dependencies are in `requirements-deploy.txt`
- Verify Python version compatibility

### Issue: Database errors
- SQLite works on most platforms
- For production, consider PostgreSQL (Render/Railway provide free tiers)

### Issue: Static files not loading
- Ensure templates directory is included
- Check file paths are relative

### Issue: Port binding errors
- Use `$PORT` environment variable (set by platform)
- Gunicorn automatically uses it

## 📊 Recommended Platform Comparison

| Platform | Free Tier | Ease | Best For |
|----------|-----------|------|----------|
| **Render** | ✅ Yes | ⭐⭐⭐⭐⭐ | Best overall |
| **Railway** | ✅ Yes | ⭐⭐⭐⭐⭐ | Fastest setup |
| **Heroku** | ❌ No | ⭐⭐⭐⭐ | Established platform |
| **PythonAnywhere** | ✅ Yes | ⭐⭐⭐ | Learning/Simple apps |

## 🔗 Quick Links

- **Render**: https://render.com
- **Railway**: https://railway.app
- **Heroku**: https://heroku.com
- **PythonAnywhere**: https://pythonanywhere.com

---

**Recommended**: Start with **Render** - it's free, easy, and reliable!
