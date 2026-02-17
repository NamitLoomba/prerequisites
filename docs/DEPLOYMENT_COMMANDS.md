# Quick Deployment Commands

## For Streamlit Cloud Deployment

### 1. Test Locally First
```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run the standalone app
streamlit run streamlit_app.py
```

### 2. Prepare Git Repository
```bash
# Initialize git (if needed)
git init

# Add essential files only
git add streamlit_app.py
git add requirements_streamlit.txt
git add .streamlit/config.toml
git add .gitignore
git add README.md

# Add model files
git add ml/model.pkl
git add ml/scaler.pkl
git add ml/model_lgb.pkl
git add ml/scaler_lgb.pkl

# Commit
git commit -m "Deploy to Streamlit Cloud"
```

### 3. Push to GitHub
```bash
# Add your GitHub repository (replace YOUR_REPO with your repository name)
git remote add origin https://github.com/NamitLoomba/YOUR_REPO.git

# Push
git branch -M main
git push -u origin main
```

### 4. Deploy on Streamlit Cloud
1. Visit: https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Main file: `streamlit_app.py`
6. Click "Deploy"

---

## For Local Development (Current Setup)

### Run Backend + Frontend Separately
```bash
# Terminal 1: Start backend API
python backend/main.py

# Terminal 2: Start frontend dashboard
python -m streamlit run frontend/dashboard.py
```

---

## Files Needed for Each Deployment Type

### Streamlit Cloud (Standalone)
```
✓ streamlit_app.py
✓ requirements_streamlit.txt
✓ .streamlit/config.toml
✓ ml/model.pkl
✓ ml/scaler.pkl
✓ ml/model_lgb.pkl
✓ ml/scaler_lgb.pkl
✗ backend/ (not needed)
✗ data/raw/ (too large)
```

### Local Development (Full Stack)
```
✓ backend/main.py
✓ frontend/dashboard.py
✓ requirements.txt
✓ ml/model.pkl
✓ ml/scaler.pkl
✓ ml/model_lgb.pkl
✓ ml/scaler_lgb.pkl
✓ ml/sequence_model.h5 (optional)
```

---

## Troubleshooting

### Port 8000 already in use
```bash
# Windows: Find and kill process
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### Streamlit not recognized
```bash
# Use Python module syntax
python -m streamlit run streamlit_app.py
```

### Models not found
```bash
# Check if model files exist
dir ml\*.pkl

# Should see:
# model.pkl
# scaler.pkl
# model_lgb.pkl
# scaler_lgb.pkl
```

---

## Quick Reference

| Task | Command |
|------|---------|
| Test standalone app | `streamlit run streamlit_app.py` |
| Test full stack (backend) | `python backend/main.py` |
| Test full stack (frontend) | `python -m streamlit run frontend/dashboard.py` |
| Check models | `dir ml\*.pkl` |
| Git status | `git status` |
| Git add all | `git add .` |
| Git commit | `git commit -m "message"` |
| Git push | `git push origin main` |

---

## What to Show Judges

### Option 1: Streamlit Cloud (Recommended)
- Share URL: `https://your-app.streamlit.app`
- No setup needed for judges
- Works on any device
- Professional deployment

### Option 2: Local Demo
- Run: `python backend/main.py` + `python -m streamlit run frontend/dashboard.py`
- Open: `http://localhost:8501`
- Shows full tech stack
- Requires your laptop

---

**For hackathon, use Streamlit Cloud for easy sharing!**
