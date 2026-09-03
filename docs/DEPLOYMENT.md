# Deployment Guide

## Current Deployment Status

- **Streamlit Cloud URL**: https://civil-engineering-ppi.streamlit.app/
- **GitHub repository**: https://github.com/jinliangyue/civil-engineering-dashboard
- **Data**: 132 monthly PPI observations (2015-01 to 2025-12) retrieved through akshare from China's National Bureau of Statistics
- **Status**: Deployed, automatically loads data from `data/raw/`

> The legacy manually-estimated fallback datasets have been removed. The current application uses the official monthly PPI dataset retrieved through akshare.

---

## Option A: Deploy to Streamlit Cloud (recommended, ~5 minutes)

### Step 1: Create GitHub repository (~2 minutes)

1. Open https://github.com
2. Sign in (create a free account if needed)
3. Click "+" in the top right, then "New repository"
4. Fill in:
   - Repository name: `civil-engineering-dashboard`
   - Description: `China Industrial PPI Time-Series Analysis and Forecasting Platform`
   - Public, do not check "Add README"
5. Click "Create repository"

### Step 2: Initialize Git locally and push (~3 minutes)

Open a terminal and run:

```bash
# 1. Switch to the project directory
cd ~/Desktop/Claude\ code/civil-engineering-dashboard

# 2. Initialize Git
git init

# 3. Configure identity (if not configured)
git config user.name "Your GitHub username"
git config user.email "your email"

# 4. Add all files
git add .

# 5. First commit
git commit -m "feat: initialize China industrial PPI platform"

# 6. Add the remote (replace with your GitHub username)
git remote add origin https://github.com/your-username/civil-engineering-dashboard.git

# 7. Push
git branch -M main
git push -u origin main
```

If a GitHub login prompt appears, enter your username and Personal Access Token (not password).

### Step 3: Deploy on Streamlit Cloud (~5 minutes)

1. Open https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Fill in:
   - Repository: `your-username/civil-engineering-dashboard`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
5. Click "Deploy"
6. Wait 2-5 minutes for the deploy to complete
7. Copy the application URL (e.g., `https://xxx.streamlit.app`)

### Step 4: Test the deployment

Open the application URL: https://civil-engineering-ppi.streamlit.app/

You should see:
- Title: China Industrial PPI Time-Series Analysis and Forecasting Platform
- 132 monthly data points loaded
- Time-series forecast tabs

---

## Option B: Local deployment (for testing only)

```bash
cd ~/Desktop/Claude\ code/civil-engineering-dashboard
pip3 install -r requirements.txt
streamlit run app/streamlit_app.py
```

The browser will open automatically at http://localhost:8501. This is for local testing only.

---

## Option C: Temporary public demo (using ngrok, ~1 minute)

If you don't want to create a GitHub repository yet, use ngrok to expose your local port:

```bash
# 1. Install ngrok
brew install ngrok

# 2. Register an ngrok account (free)
# Visit ngrok.com to register and copy your authtoken

# 3. Configure ngrok
ngrok config add-authtoken your-token

# 4. Start Streamlit
streamlit run app/streamlit_app.py &

# 5. Expose to the public internet
ngrok http 8501
```

You will receive a public URL (free tier valid for 8 hours).

---

## Recommended Path

- **Resume portfolio**: Option A (deploy to Streamlit Cloud)
- **Local testing**: Option B
- **Temporary demo**: Option C (if GitHub setup fails)

---

## After Deployment: Resume Links

```
GitHub: https://github.com/your-username/civil-engineering-dashboard
Demo:   https://your-app.streamlit.app
```

---

## Deployment Checklist

- [ ] GitHub repository is public
- [ ] README renders correctly
- [ ] Streamlit application URL is accessible
- [ ] 132 monthly data points load successfully
- [ ] Chinese text renders correctly (report if garbled)
- [ ] All forecast tabs display normally

---

## Common Issues

**Q: GitHub push authentication fails?**
A: Use a Personal Access Token instead of a password. GitHub → Settings → Developer settings → Personal access tokens → Generate new token. Select `repo` scope.

**Q: Streamlit Cloud deployment fails?**
A: Check that `requirements.txt` is complete. Look at the Logs tab on the deploy page.

**Q: Chinese text renders as garbled characters?**
A: matplotlib PNG output may have font issues (Plotly defaults are fine). Report any garbling so we can add Chinese font configuration.

---

## After Deployment: Report Back

1. GitHub repository URL
2. Streamlit Cloud application URL
3. Whether the application renders normally
