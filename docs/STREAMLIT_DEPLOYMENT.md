# Streamlit Cloud Deployment Guide

## Files Required for Deployment

### Essential Files (MUST INCLUDE)

1. **streamlit_app.py** - Main application file (Streamlit Cloud looks for this)
2. **requirements_streamlit.txt** - Python dependencies (includes TensorFlow)
3. **.streamlit/config.toml** - Streamlit configuration
4. **ml/model.pkl** - XGBoost trained model
5. **ml/scaler.pkl** - XGBoost feature scaler
6. **ml/model_lgb.pkl** - LightGBM trained model
7. **ml/scaler_lgb.pkl** - LightGBM feature scaler
8. **ml/sequence_model.h5** - TensorFlow LSTM model
9. **ml/sequence_scaler.pkl** - LSTM feature scaler
10. **ml/sequence_model.py** - LSTM predictor class
11. **ml/__init__.py** - Python package marker

### Optional Files

- **README.md** - Project description (shown on GitHub)
- **.gitignore** - Files to exclude from Git
- **PROJECT_DOCUMENTATION.md** - Detailed documentation

### Files to EXCLUDE (Don't Upload)

- **backend/** folder - Not needed for standalone Streamlit
- **data/raw/** - Large CSV files (use .gitignore)
- **__pycache__/** - Python cache
- **test_*.py** - Test scripts
- **requirements.txt** - Use requirements_streamlit.txt instead

## Deployment Steps

### Step 1: Prepare Your Repository

1. Create a new GitHub repository (public or private)
2. Copy these files to your repository:
   ```
   your-repo/
   ├── streamlit_app.py          ← Main app
   ├── requirements_streamlit.txt ← Dependencies
   ├── .streamlit/
   │   └── config.toml            ← Config
   ├── ml/
   │   ├── model.pkl              ← XGBoost model
   │   ├── scaler.pkl             ← XGBoost scaler
   │   ├── model_lgb.pkl          ← LightGBM model
   │   └── scaler_lgb.pkl         ← LightGBM scaler
   ├── .gitignore                 ← Git ignore rules
   └── README.md                  ← Project description
   ```

### Step 2: Push to GitHub

```bash
# Initialize git (if not already done)
git init

# Add files
git add streamlit_app.py
git add requirements_streamlit.txt
git add .streamlit/config.toml
git add ml/model.pkl
git add ml/scaler.pkl
git add ml/model_lgb.pkl
git add ml/scaler_lgb.pkl
git add .gitignore
git add README.md

# Commit
git commit -m "Initial deployment for Streamlit Cloud"

# Add remote (replace YOUR_REPO with your repository name)
git remote add origin https://github.com/NamitLoomba/YOUR_REPO.git

# Push
git branch -M main
git push -u origin main
```

### Step 3: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io/
2. Sign in with GitHub
3. Click "New app"
4. Select your repository
5. Set:
   - **Main file path**: `streamlit_app.py`
   - **Python version**: 3.12 (or 3.11)
6. Click "Deploy"

### Step 4: Wait for Deployment

- Streamlit Cloud will install dependencies from `requirements_streamlit.txt`
- Build time: 2-5 minutes
- Your app will be live at: `https://YOUR_USERNAME-YOUR_REPO.streamlit.app`

## Troubleshooting

### Error: "No module named 'xgboost'"
- Check that `requirements_streamlit.txt` is in the root directory
- Verify file name is exactly `requirements_streamlit.txt` or `requirements.txt`

### Error: "Model file not found"
- Ensure `ml/model.pkl` and other model files are committed to Git
- Check file paths in `streamlit_app.py` match your directory structure
- Model files must be < 100MB each for GitHub

### Error: "App is too large"
- Remove TensorFlow dependencies if not using LSTM model
- Exclude large CSV files using `.gitignore`
- Consider using Git LFS for large model files

### App loads but shows errors
- Check Streamlit Cloud logs (click "Manage app" → "Logs")
- Verify all model files are present in the `ml/` folder
- Test locally first: `streamlit run streamlit_app.py`

## Testing Locally Before Deployment

```bash
# Install dependencies
pip install -r requirements_streamlit.txt

# Run app
streamlit run streamlit_app.py

# Open browser to http://localhost:8501
```

## File Size Limits

- GitHub: 100MB per file (use Git LFS for larger files)
- Streamlit Cloud: 1GB total repository size
- Your model files:
  - model.pkl: ~500KB ✓
  - scaler.pkl: ~10KB ✓
  - model_lgb.pkl: ~300KB ✓
  - scaler_lgb.pkl: ~10KB ✓
  - sequence_model.h5: ~465KB ✓
  - sequence_scaler.pkl: ~10KB ✓
  - sequence_model.py: ~15KB ✓
  - Total: ~1.3MB ✓ (well under limit)

## Custom Domain (Optional)

After deployment, you can:
1. Go to app settings
2. Add custom domain
3. Point your DNS to Streamlit Cloud

## Environment Variables (If Needed)

If you need API keys or secrets:
1. Go to app settings
2. Click "Secrets"
3. Add key-value pairs in TOML format

## Updating Your App

```bash
# Make changes to streamlit_app.py
git add streamlit_app.py
git commit -m "Update feature"
git push

# Streamlit Cloud auto-deploys on push
```

## Alternative: Deploy with Full Backend

If you want to keep the FastAPI backend:
1. Deploy backend on Render/Railway/Heroku
2. Update `API_BASE_URL` in dashboard to point to deployed backend
3. Deploy frontend on Streamlit Cloud

## Cost

- Streamlit Cloud: FREE for public repos
- GitHub: FREE for public repos
- Total cost: $0

## Support

- Streamlit Docs: https://docs.streamlit.io/
- Community Forum: https://discuss.streamlit.io/
- GitHub Issues: Your repository issues page

---

## Quick Checklist

Before deploying, verify:
- [ ] `streamlit_app.py` exists in root
- [ ] `requirements_streamlit.txt` exists in root
- [ ] `.streamlit/config.toml` exists
- [ ] All 4 model files in `ml/` folder
- [ ] Files pushed to GitHub
- [ ] Repository is public (or you have Streamlit Cloud Pro)
- [ ] Tested locally with `streamlit run streamlit_app.py`

## Expected Result

Your app will be live at a URL like:
```
https://your-username-predeliquency-risk.streamlit.app
```

Users can:
- Select XGBoost or LightGBM models
- Input customer risk indicators
- Get real-time risk predictions
- Compare both models side-by-side
- See intervention recommendations

---

**Ready to deploy? Follow Step 1-3 above!**
