# Quick Start Guide for Namit Loomba

## Your Personalized Deployment Commands

### Step 1: Test Locally (2 minutes)

```cmd
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```

If it works, proceed to Step 2. If not, fix errors first.

---

### Step 2: Create GitHub Repository (3 minutes)

1. Go to https://github.com/NamitLoomba
2. Click "New repository" (green button)
3. Repository name: `predeliquency-risk` (or any name you prefer)
4. Description: `AI-Powered Pre-Delinquency Risk Platform`
5. Choose: **Public** (for free Streamlit Cloud)
6. Don't initialize with README (we have one)
7. Click "Create repository"

---

### Step 3: Push Your Code (5 minutes)

Copy and paste these commands one by one:

```cmd
git init
```

```cmd
git add streamlit_app.py
git add requirements_streamlit.txt
git add .streamlit/config.toml
git add .gitignore
git add ml/model.pkl
git add ml/scaler.pkl
git add ml/model_lgb.pkl
git add ml/scaler_lgb.pkl
git add ml/sequence_model.h5
git add ml/sequence_scaler.pkl
git add ml/sequence_model.py
git add ml/__init__.py
git add data/raw/synthetic_transactions.csv
```

```cmd
git commit -m "Deploy to Streamlit Cloud"
```

**Replace `predeliquency-risk` with your actual repo name:**
```cmd
git remote add origin https://github.com/NamitLoomba/predeliquency-risk.git
```

```cmd
git branch -M main
```

```cmd
git push -u origin main
```

If it asks for credentials:
- Username: `NamitLoomba`
- Password: Use a Personal Access Token (not your GitHub password)
  - Get token at: https://github.com/settings/tokens

---

### Step 4: Deploy on Streamlit Cloud (5 minutes)

1. Go to https://share.streamlit.io/
2. Click "Sign in with GitHub"
3. Authorize Streamlit Cloud
4. Click "New app"
5. Fill in:
   - **Repository**: `NamitLoomba/predeliquency-risk`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
6. Click "Deploy"
7. Wait 3-5 minutes for build

---

### Step 5: Get Your App URL

Your app will be live at:
```
https://namitloomba-predeliquency-risk.streamlit.app
```

(Or whatever you named your repo)

---

## What to Show Judges

### Option 1: Cloud App (Recommended)
Share this URL with judges:
```
https://namitloomba-predeliquency-risk.streamlit.app
```

**What they'll see:**
- XGBoost model ✓
- LightGBM model ✓
- Model comparison ✓
- Interactive dashboard ✓

### Option 2: Local Demo (Backup)
If judges want to see all 3 models including TensorFlow:

```cmd
REM Terminal 1
python backend/main.py

REM Terminal 2 (new window)
python -m streamlit run frontend/dashboard.py
```

Open: http://localhost:8501

---

## Troubleshooting

### "Port 8000 already in use"
```cmd
netstat -ano | findstr :8000
taskkill /PID <PID_NUMBER> /F
```

### "Streamlit not recognized"
```cmd
python -m streamlit run streamlit_app.py
```

### "Git not recognized"
Download Git from: https://git-scm.com/download/win

### GitHub asks for password
Use Personal Access Token instead:
1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select scopes: `repo`
4. Copy token and use as password

---

## Files Checklist

Before pushing to GitHub, verify these files exist:

```cmd
dir streamlit_app.py
dir requirements_streamlit.txt
dir .streamlit\config.toml
dir ml\model.pkl
dir ml\scaler.pkl
dir ml\model_lgb.pkl
dir ml\scaler_lgb.pkl
```

All should show file sizes, not "File Not Found"

---

## Your Repository Structure

```
NamitLoomba/predeliquency-risk/
├── streamlit_app.py              (Main app)
├── requirements_streamlit.txt    (Dependencies)
├── .streamlit/
│   └── config.toml              (Config)
├── ml/
│   ├── model.pkl                (XGBoost)
│   ├── scaler.pkl               (XGBoost scaler)
│   ├── model_lgb.pkl            (LightGBM)
│   └── scaler_lgb.pkl           (LightGBM scaler)
└── .gitignore                   (Git ignore)
```

---

## Timeline

- **Now**: Test locally
- **+5 min**: Create GitHub repo
- **+10 min**: Push code
- **+15 min**: Deploy on Streamlit Cloud
- **+20 min**: App is live!

---

## Success Criteria

✅ You're done when:
- App loads at `https://namitloomba-predeliquency-risk.streamlit.app`
- No errors in browser
- Can select models
- Can input data
- Can get predictions
- Model comparison works

---

## Next Steps After Deployment

1. **Test your app**: Open the URL and try all features
2. **Add to presentation**: Include the URL in your slides
3. **Update README**: Add the live URL to README.md
4. **Share with team**: Send URL to teammates
5. **Prepare demo script**: Practice showing features to judges

---

## Demo Script for Judges

1. "Here's our live app: [YOUR_URL]"
2. "We have 2 ML models: XGBoost and LightGBM"
3. "Let me show you a high-risk customer..."
   - Set Salary Delay: 7 days
   - Set Savings Drop: 50%
   - Set Failed Auto-debits: 3
   - Click "Analyze Risk"
4. "See? 85% risk score, Critical level"
5. "Now let's compare both models..."
   - Select "Compare Both Models"
   - Click "Analyze Risk"
6. "Both models agree - this customer needs immediate intervention"
7. "The system recommends: [show recommendation]"

---

## Contact

If you get stuck:
- Check DEPLOYMENT_CHECKLIST.md
- Check STREAMLIT_DEPLOYMENT.md
- Ask on Streamlit Community: https://discuss.streamlit.io/

---

**Good luck with your hackathon! 🚀**
