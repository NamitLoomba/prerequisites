# Streamlit Cloud Deployment Guide

## Step 1: Push to GitHub

Run these commands in your terminal:

```cmd
git add .
git commit -m "Add complete fraud detection application for Streamlit deployment"
git push origin main
```

## Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with your GitHub account
3. Click "New app"
4. Select:
   - Repository: `NamitLoomba/prerequisites`
   - Branch: `main`
   - Main file path: `streamlit_app_lite.py` (recommended for faster deployment)
5. Click "Deploy"

## Step 3: Configure Dependencies

Streamlit Cloud will automatically install packages from `requirements_lite.txt`

## Recommended App File

Use `streamlit_app_lite.py` - it's optimized for deployment without heavy TensorFlow dependencies.

## Troubleshooting

If deployment fails:
- Check the logs in Streamlit Cloud dashboard
- Verify all required files are pushed to GitHub
- Ensure `requirements_lite.txt` has all necessary packages
- Make sure model files (*.pkl, *.h5) are included in the repo

## Repository Details

- GitHub Username: NamitLoomba
- Repository: prerequisites
- Branch: main
- App File: streamlit_app_lite.py
