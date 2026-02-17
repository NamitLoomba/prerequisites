# Streamlit Cloud Deployment Checklist

## Pre-Deployment Checklist

### ✅ Files Ready
- [ ] `streamlit_app.py` exists in root directory
- [ ] `requirements_streamlit.txt` exists in root directory
- [ ] `.streamlit/config.toml` exists
- [ ] `.gitignore` exists
- [ ] `README_STREAMLIT.md` ready (rename to README.md)

### ✅ Model Files Ready
- [ ] `ml/model.pkl` exists (~500KB)
- [ ] `ml/scaler.pkl` exists (~10KB)
- [ ] `ml/model_lgb.pkl` exists (~300KB)
- [ ] `ml/scaler_lgb.pkl` exists (~10KB)
- [ ] All model files are < 100MB (GitHub limit)

### ✅ Test Locally
```bash
pip install -r requirements_streamlit.txt
streamlit run streamlit_app.py
```
- [ ] App loads without errors
- [ ] XGBoost model works
- [ ] LightGBM model works
- [ ] Model comparison works
- [ ] All sliders and inputs work
- [ ] Gauges display correctly

### ✅ GitHub Setup
- [ ] GitHub account created
- [ ] New repository created (public recommended)
- [ ] Repository name chosen (e.g., "predeliquency-risk-platform")

## Deployment Steps

### Step 1: Prepare Files
```bash
# Copy/rename README
copy README_STREAMLIT.md README.md

# Verify files exist
dir streamlit_app.py
dir requirements_streamlit.txt
dir .streamlit\config.toml
dir ml\*.pkl
```
- [ ] All files verified

### Step 2: Initialize Git
```bash
git init
git add streamlit_app.py
git add requirements_streamlit.txt
git add .streamlit/config.toml
git add .gitignore
git add README.md
git add ml/model.pkl
git add ml/scaler.pkl
git add ml/model_lgb.pkl
git add ml/scaler_lgb.pkl
```
- [ ] Files added to Git

### Step 3: Commit
```bash
git commit -m "Initial deployment to Streamlit Cloud"
```
- [ ] Files committed

### Step 4: Push to GitHub
```bash
# Replace YOUR_REPO with your actual repository name
git remote add origin https://github.com/NamitLoomba/YOUR_REPO.git
git branch -M main
git push -u origin main
```
- [ ] Files pushed to GitHub
- [ ] Verify files visible on GitHub website

### Step 5: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io/
   - [ ] Website opened
2. Click "Sign in with GitHub"
   - [ ] Signed in successfully
3. Click "New app"
   - [ ] New app dialog opened
4. Fill in details:
   - Repository: `NamitLoomba/YOUR_REPO`
   - Branch: `main`
   - Main file path: `streamlit_app.py`
   - [ ] Details filled
5. Click "Deploy"
   - [ ] Deployment started

### Step 6: Wait for Build
- [ ] Build started (shows logs)
- [ ] Dependencies installing
- [ ] App building
- [ ] App deployed successfully
- [ ] App URL received

### Step 7: Test Deployed App
- [ ] App loads in browser
- [ ] No error messages
- [ ] Models load successfully
- [ ] Can input data
- [ ] Can get predictions
- [ ] Gauges display correctly
- [ ] Model comparison works

### Step 8: Update README
- [ ] Add app URL to README.md
- [ ] Update badge link
- [ ] Push changes to GitHub

## Post-Deployment

### Share with Judges
- [ ] Copy app URL
- [ ] Add to presentation slides
- [ ] Test URL on different device
- [ ] Prepare demo script

### Backup Plan
- [ ] Local demo ready (backend + frontend)
- [ ] Commands documented
- [ ] Tested on your laptop

### Documentation
- [ ] README.md updated with app URL
- [ ] Screenshots added (optional)
- [ ] Contact info updated

## Troubleshooting Checklist

### If Build Fails
- [ ] Check Streamlit Cloud logs
- [ ] Verify `requirements_streamlit.txt` syntax
- [ ] Ensure all dependencies have versions
- [ ] Check Python version compatibility

### If App Shows Errors
- [ ] Check model file paths in code
- [ ] Verify all model files uploaded to GitHub
- [ ] Check file sizes (< 100MB each)
- [ ] Review Streamlit Cloud logs

### If Models Don't Load
- [ ] Verify `ml/` folder structure
- [ ] Check model file names match code
- [ ] Ensure files are not in .gitignore
- [ ] Re-push model files if needed

## Final Verification

### Functionality Test
- [ ] Open app URL
- [ ] Select XGBoost model
- [ ] Input test data
- [ ] Click "Analyze Risk"
- [ ] Verify prediction appears
- [ ] Select LightGBM model
- [ ] Verify different prediction
- [ ] Select "Compare Both"
- [ ] Verify comparison view works

### Performance Test
- [ ] App loads in < 5 seconds
- [ ] Predictions return in < 2 seconds
- [ ] No lag when changing inputs
- [ ] Gauges animate smoothly

### Cross-Device Test
- [ ] Test on desktop browser
- [ ] Test on mobile browser
- [ ] Test on tablet (if available)
- [ ] Share URL with friend to test

## Success Criteria

✅ **Deployment Successful When:**
- App is live at public URL
- No errors in Streamlit Cloud logs
- All models load successfully
- Predictions work correctly
- Model comparison works
- App is accessible from any device
- URL can be shared with judges

## Your App URL

Once deployed, your app will be at:
```
https://namitloomba-YOUR_REPO.streamlit.app
```

Example (if you name your repo "predeliquency-risk"):
```
https://namitloomba-predeliquency-risk.streamlit.app
```

## Time Estimate

- File preparation: 5 minutes
- Git setup: 5 minutes
- GitHub push: 2 minutes
- Streamlit Cloud deployment: 3-5 minutes
- Testing: 5 minutes
- **Total: ~20 minutes**

## Support Resources

- Streamlit Docs: https://docs.streamlit.io/
- Streamlit Community: https://discuss.streamlit.io/
- GitHub Docs: https://docs.github.com/
- Your deployment guide: STREAMLIT_DEPLOYMENT.md

---

## Quick Commands Reference

```bash
# Test locally
streamlit run streamlit_app.py

# Git setup
git init
git add .
git commit -m "Deploy to Streamlit Cloud"
git remote add origin https://github.com/NamitLoomba/YOUR_REPO.git
git push -u origin main

# Update after changes
git add .
git commit -m "Update app"
git push
```

---

**Ready to deploy? Start with "Pre-Deployment Checklist" above!**
