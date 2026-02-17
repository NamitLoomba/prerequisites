# Split Deployment Instructions

## Part 1: Deploy Backend on PythonAnywhere

### Step 1: Sign Up
1. Go to https://www.pythonanywhere.com
2. Create free account (no credit card needed)
3. Verify email

### Step 2: Clone Repository
1. Go to "Consoles" tab → Click "Bash"
2. Run:
```bash
git clone https://github.com/NamitLoomba/prerequisites.git
cd prerequisites
```

### Step 3: Install Dependencies
```bash
pip3.10 install --user fastapi uvicorn xgboost lightgbm scikit-learn pandas numpy joblib pydantic
```

### Step 4: Create WSGI Configuration
1. Go to "Web" tab
2. Click "Add a new web app"
3. Choose "Manual configuration"
4. Select "Python 3.10"
5. Click on WSGI configuration file
6. Replace content with:

```python
import sys
import os

# Replace YOUR_USERNAME with your actual username
project_home = '/home/YOUR_USERNAME/prerequisites'
if project_home not in sys.path:
    sys.path.insert(0, project_home)

from backend.main import app as application
```

### Step 5: Configure Web App
1. In "Web" tab, set:
   - Source code: `/home/YOUR_USERNAME/prerequisites`
   - Working directory: `/home/YOUR_USERNAME/prerequisites`
2. Click "Reload" button

### Step 6: Test Backend
Your backend will be at: `https://YOUR_USERNAME.pythonanywhere.com`

Test it: `https://YOUR_USERNAME.pythonanywhere.com/api/v1/`

---

## Part 2: Deploy Frontend on Streamlit Cloud

### Step 1: Push Code to GitHub
```cmd
git add .
git commit -m "Add API-based frontend"
git push origin main
```

### Step 2: Deploy on Streamlit Cloud
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Configure:
   - Repository: `NamitLoomba/prerequisites`
   - Branch: `main`
   - Main file: `streamlit_app_api.py`
5. Click "Advanced settings"
6. Add to Secrets:
```toml
API_URL = "https://YOUR_USERNAME.pythonanywhere.com/api/v1"
```
7. Click "Deploy"

---

## Part 3: Test Everything

### Backend Test
Visit: `https://YOUR_USERNAME.pythonanywhere.com/docs`
- You should see FastAPI Swagger documentation
- Test the `/api/v1/predict-risk` endpoint

### Frontend Test
Visit your Streamlit app URL
- Should show "✅ Connected to Backend API"
- Try making a prediction
- Upload CSV for batch analysis

---

## Troubleshooting

### Backend Issues
- Check PythonAnywhere error logs in "Web" tab
- Ensure all model files are uploaded
- Verify WSGI file has correct username

### Frontend Issues
- Check Streamlit logs for API connection errors
- Verify API_URL in secrets matches your PythonAnywhere URL
- Ensure backend is running and accessible

### CORS Issues
If you get CORS errors, the backend already has CORS enabled for all origins.

---

## URLs After Deployment

- Backend API: `https://YOUR_USERNAME.pythonanywhere.com`
- API Docs: `https://YOUR_USERNAME.pythonanywhere.com/docs`
- Frontend App: `https://namitloomba-prerequisites-streamlit-app-api-xxxxx.streamlit.app`

---

## Cost
- PythonAnywhere: FREE (no credit card)
- Streamlit Cloud: FREE (no credit card)
- Total: $0/month
