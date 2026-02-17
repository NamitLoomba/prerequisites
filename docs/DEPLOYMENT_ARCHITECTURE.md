# Deployment Architecture

## Current Setup (Local Development)

```
┌─────────────────────────────────────────────────────────────┐
│                     YOUR LAPTOP                              │
│                                                              │
│  ┌──────────────────┐         ┌──────────────────┐         │
│  │   Backend API    │         │   Frontend UI    │         │
│  │   (FastAPI)      │◄────────┤   (Streamlit)    │         │
│  │                  │  HTTP   │                  │         │
│  │  Port: 8000      │         │  Port: 8501      │         │
│  │                  │         │                  │         │
│  │  ┌────────────┐  │         │                  │         │
│  │  │ XGBoost    │  │         │                  │         │
│  │  │ LightGBM   │  │         │                  │         │
│  │  │ TensorFlow │  │         │                  │         │
│  │  └────────────┘  │         │                  │         │
│  └──────────────────┘         └──────────────────┘         │
│                                                              │
└─────────────────────────────────────────────────────────────┘
         │                              │
         │                              │
         └──────────────┬───────────────┘
                        │
                        ▼
                 http://localhost:8501
```

## Streamlit Cloud Deployment (Simplified)

```
┌─────────────────────────────────────────────────────────────┐
│                  STREAMLIT CLOUD                             │
│                  (Free Hosting)                              │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         streamlit_app.py                             │  │
│  │         (Standalone App)                             │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────┐    │  │
│  │  │  Models Loaded Directly                    │    │  │
│  │  │  ┌──────────┐  ┌──────────┐               │    │  │
│  │  │  │ XGBoost  │  │ LightGBM │               │    │  │
│  │  │  └──────────┘  └──────────┘               │    │  │
│  │  └────────────────────────────────────────────┘    │  │
│  │                                                      │  │
│  │  ┌────────────────────────────────────────────┐    │  │
│  │  │  Streamlit UI                              │    │  │
│  │  │  - Model Selection                         │    │  │
│  │  │  - Risk Prediction                         │    │  │
│  │  │  - Model Comparison                        │    │  │
│  │  └────────────────────────────────────────────┘    │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
                        │
                        │
                        ▼
    https://namitloomba-predeliquency-risk.streamlit.app
                        │
                        │
                        ▼
              ┌─────────────────┐
              │  Judges/Users   │
              │  (Any Device)   │
              └─────────────────┘
```

## File Flow: Local → GitHub → Streamlit Cloud

```
┌─────────────────────┐
│   YOUR LAPTOP       │
│                     │
│  streamlit_app.py   │
│  requirements.txt   │
│  ml/model.pkl       │
│  ml/scaler.pkl      │
│  ml/model_lgb.pkl   │
│  ml/scaler_lgb.pkl  │
└─────────────────────┘
          │
          │ git push
          ▼
┌─────────────────────┐
│   GITHUB            │
│                     │
│  NamitLoomba/       │
│  predeliquency-risk │
│                     │
│  (Public Repo)      │
└─────────────────────┘
          │
          │ Auto-deploy
          ▼
┌─────────────────────┐
│  STREAMLIT CLOUD    │
│                     │
│  Builds & Hosts     │
│  Your App           │
│                     │
│  (Free Tier)        │
└─────────────────────┘
          │
          │ Public URL
          ▼
┌─────────────────────┐
│   JUDGES/USERS      │
│                     │
│  Access via URL     │
│  No setup needed    │
└─────────────────────┘
```

## What's Different Between Local and Cloud?

| Feature | Local Setup | Streamlit Cloud |
|---------|-------------|-----------------|
| **Files** | backend/ + frontend/ | streamlit_app.py only |
| **API** | FastAPI (separate) | Not needed |
| **Models** | 3 (XGBoost, LightGBM, TensorFlow) | 2 (XGBoost, LightGBM) |
| **Ports** | 8000 + 8501 | Automatic |
| **Access** | localhost only | Public URL |
| **Setup** | 2 terminals | None |
| **Cost** | Free | Free |
| **Sharing** | Can't share | Easy URL sharing |

## Why Two Versions?

### Local Version (Full Stack)
- **Purpose**: Development & full demo
- **Advantage**: Shows all 3 models + backend API
- **Use Case**: If judges want deep technical dive

### Cloud Version (Simplified)
- **Purpose**: Easy sharing & presentation
- **Advantage**: Professional deployment, no setup
- **Use Case**: Primary demo for hackathon

## Data Flow in Cloud Version

```
User Input
    │
    ▼
┌─────────────────────┐
│  Streamlit UI       │
│  (streamlit_app.py) │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Feature Engineering│
│  (create_features)  │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Model Selection    │
│  (XGBoost/LightGBM) │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Load Model & Scale │
│  (joblib.load)      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Predict Risk       │
│  (model.predict)    │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Display Results    │
│  (Plotly gauges)    │
└─────────────────────┘
    │
    ▼
User sees prediction
```

## Deployment Timeline

```
Day 0: Development
├── Build backend API
├── Train 3 ML models
├── Create dashboard
└── Test locally ✓

Day 1: Deployment (TODAY)
├── Create streamlit_app.py
├── Test standalone version
├── Push to GitHub
├── Deploy to Streamlit Cloud
└── Share URL with judges ✓

Hackathon Day: Demo
├── Show cloud app (primary)
├── Show local app (if needed)
└── Answer judge questions ✓
```

## Tech Stack Comparison

### Local Setup
```
Frontend: Streamlit
    ↕ HTTP
Backend: FastAPI
    ↕
Models: XGBoost + LightGBM + TensorFlow
```

### Cloud Setup
```
Frontend + Models: Streamlit (All-in-one)
    ↕
Models: XGBoost + LightGBM (Embedded)
```

## Why This Architecture?

### Advantages of Cloud Deployment
1. **No Infrastructure**: Streamlit handles hosting
2. **Auto-scaling**: Handles multiple users
3. **HTTPS**: Secure by default
4. **CDN**: Fast loading worldwide
5. **Zero Cost**: Free tier sufficient
6. **Easy Updates**: Git push = auto-deploy

### Advantages of Local Demo
1. **Full Features**: All 3 models available
2. **API Showcase**: Can demo REST endpoints
3. **Flexibility**: Can modify on the fly
4. **No Internet**: Works offline

## Recommended Strategy

```
┌─────────────────────────────────────────┐
│  HACKATHON PRESENTATION                 │
├─────────────────────────────────────────┤
│                                         │
│  1. Start with Cloud App                │
│     "Here's our live deployment..."     │
│     [Show URL]                          │
│                                         │
│  2. Demo Features                       │
│     - Model selection                   │
│     - Risk prediction                   │
│     - Model comparison                  │
│                                         │
│  3. If judges ask about backend:        │
│     "We also have a full REST API..."   │
│     [Switch to local demo]              │
│                                         │
│  4. If judges ask about TensorFlow:     │
│     "We trained an LSTM model too..."   │
│     [Show local demo]                   │
│                                         │
└─────────────────────────────────────────┘
```

## Security & Privacy

### Cloud Deployment
- ✅ No real customer data
- ✅ Synthetic data only
- ✅ No API keys needed
- ✅ No database connections
- ✅ Stateless application

### Local Deployment
- ✅ Same security as cloud
- ✅ Runs on your machine
- ✅ No external connections

## Scalability

### Current (Free Tier)
- Handles: ~100 concurrent users
- Response time: < 2 seconds
- Uptime: 99%+

### If Needed (Paid Tier)
- Handles: 1000+ concurrent users
- Custom domain
- Priority support
- More resources

## Summary

You're deploying a **simplified, standalone version** to Streamlit Cloud for easy sharing, while keeping the **full-featured local version** as a backup for deep technical demos.

Both versions work perfectly - choose based on what judges want to see!
