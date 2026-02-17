# Test Before Deployment Checklist

## Your App is Running at: http://localhost:8502

Follow this checklist to verify everything works before deploying to Streamlit Cloud.

---

## ✅ Test 1: Model Loading (30 seconds)

1. Open http://localhost:8502
2. Look at the top green banner
3. Should say: "✅ Models Loaded | XGBoost: ✓ | LightGBM: ✓ | TensorFlow LSTM: ✓"

**Expected**: All 3 models show checkmarks
**If failed**: Check that all model files exist in ml/ folder

---

## ✅ Test 2: XGBoost Prediction (1 minute)

1. Stay on "Risk Prediction" page
2. Select "🌲 XGBoost (Traditional ML)"
3. Set sliders:
   - Salary Delay: 7 days
   - Savings Decline: 50%
   - Failed Auto-debits: 3
4. Click "Analyze Risk"

**Expected**: 
- Risk score around 70-90%
- Risk level: High or Critical
- Gauge displays correctly
- Recommendation shows

---

## ✅ Test 3: LightGBM Prediction (1 minute)

1. Select "⚡ LightGBM (Fast ML)"
2. Keep same slider values
3. Click "Analyze Risk"

**Expected**:
- Risk score similar to XGBoost (±5%)
- Risk level matches or close
- Gauge displays correctly

---

## ✅ Test 4: TensorFlow LSTM Prediction (1 minute)

1. Select "🧠 TensorFlow LSTM (Deep Learning)"
2. Keep same slider values
3. Click "Analyze Risk"

**Expected**:
- Risk score calculated
- Shows "30-day sequence" in title
- Blue info box about synthetic sequence
- Gauge displays correctly

---

## ✅ Test 5: Compare 2 Models (1 minute)

1. Select "🔄 Compare XGBoost vs LightGBM"
2. Click "Analyze Risk"

**Expected**:
- Two columns side-by-side
- XGBoost on left, LightGBM on right
- Both show gauges
- Agreement analysis at bottom
- Score difference calculated

---

## ✅ Test 6: Compare All 3 Models (1 minute)

1. Select "🔬 Compare All 3 Models"
2. Click "Analyze Risk"

**Expected**:
- Three columns side-by-side
- XGBoost, LightGBM, LSTM
- All show gauges
- Consensus analysis at bottom
- Average score, std dev, agreement level

---

## ✅ Test 7: Portfolio Analysis - Sample Data (2 minutes)

1. Click "📊 Portfolio Overview" in sidebar
2. Click "Load Sample Portfolio" button

**Expected**:
- Success message: "✅ Loaded 50 sample customers"
- Model selection dropdown appears

3. Select "XGBoost"
4. Click "Analyze Portfolio"

**Expected**:
- Portfolio summary with 4 metrics
- Pie chart showing risk distribution
- Histogram showing risk scores
- Table of high-risk customers
- Download button appears

---

## ✅ Test 8: Download Results (30 seconds)

1. After portfolio analysis completes
2. Click "📥 Download Portfolio Analysis (CSV)"

**Expected**:
- CSV file downloads
- File name: portfolio_analysis_xgboost.csv
- Contains customer_id, risk_score, risk_level columns

---

## ✅ Test 9: CSV Upload (2 minutes)

1. Still on Portfolio Overview page
2. Scroll to "Expected CSV Format" section
3. Copy the sample data shown
4. Create a new CSV file with that data
5. Click "Browse files" under "Upload CSV file"
6. Select your CSV
7. Select a model
8. Click "Analyze Portfolio"

**Expected**:
- File uploads successfully
- Shows number of customers loaded
- Analysis completes
- Results display

---

## ✅ Test 10: About Page (30 seconds)

1. Click "ℹ️ About" in sidebar

**Expected**:
- Technology stack section shows all 3 models
- Model details for XGBoost, LightGBM, LSTM
- Accuracy metrics shown
- Disclaimer at bottom

---

## ✅ Test 11: Low Risk Customer (1 minute)

1. Go back to "Risk Prediction"
2. Select any model
3. Set all sliders to 0 or minimum values
4. Click "Analyze Risk"

**Expected**:
- Risk score: 0-25%
- Risk level: Low
- Green gauge
- Success message (green box)
- Recommendation: "Continue standard monitoring"

---

## ✅ Test 12: Critical Risk Customer (1 minute)

1. Select any model
2. Set all sliders to maximum values:
   - Salary Delay: 30 days
   - Savings Decline: 100%
   - Utility Delay: 30 days
   - Discretionary Drop: 100%
   - ATM Increase: 20
   - UPI Lending: 10
   - Failed Auto-debits: 5
3. Click "Analyze Risk"

**Expected**:
- Risk score: 75-100%
- Risk level: Critical
- Red gauge
- Error message (red box)
- Recommendation: "Offer payment holiday or emergency loan restructuring"

---

## 🎯 All Tests Passed?

If all 12 tests passed, you're ready to deploy!

### Next Steps:
1. Open COPY_PASTE_COMMANDS.txt
2. Follow the git commands
3. Push to GitHub
4. Deploy on Streamlit Cloud

---

## ❌ If Any Test Failed

### Models not loading?
```cmd
dir ml\*.pkl
dir ml\*.h5
dir ml\sequence_model.py
```
All files should exist.

### TensorFlow errors?
```cmd
pip install tensorflow>=2.13.0
```

### CSV not found?
```cmd
dir data\raw\synthetic_transactions.csv
```
File should exist.

### Port already in use?
Stop the old frontend:
```cmd
netstat -ano | findstr :8501
taskkill /PID <NUMBER> /F
```

---

## 📊 Performance Benchmarks

Your app should be:
- **Load time**: < 5 seconds
- **Prediction time**: < 2 seconds per customer
- **Portfolio analysis**: < 10 seconds for 50 customers
- **Model comparison**: < 3 seconds

If slower, check:
- Model file sizes
- TensorFlow installation
- System resources

---

## 🎬 Record a Demo Video (Optional)

Before deploying, consider recording a quick demo:
1. Open OBS Studio or screen recorder
2. Go through Tests 2, 6, and 7
3. Show all features working
4. Save video for presentation backup

---

**Once all tests pass, you're ready to deploy! 🚀**

Open NAMIT_QUICK_START.md for deployment commands.
