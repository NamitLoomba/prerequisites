# Pre-Delinquency Risk Detection - Prototype

A lightweight prototype version of the fraud detection system for quick demos and presentations.

## Features

- ✅ Simple, fast setup (no ML models needed)
- ✅ Mock predictions using rule-based logic
- ✅ Interactive UI with Streamlit
- ✅ Real-time risk visualization
- ✅ Minimal dependencies
- ✅ Perfect for demos and presentations

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements_prototype.txt
```

### 2. Run the Prototype

```bash
streamlit run app_prototype.py
```

The app will open in your browser at `http://localhost:8501`

## What's Different from Full Version?

| Feature | Prototype | Full Version |
|---------|-----------|--------------|
| ML Models | ❌ Mock predictions | ✅ XGBoost + LightGBM |
| Setup Time | 1 minute | 10-15 minutes |
| Dependencies | 3 packages | 15+ packages |
| File Size | <10 KB | 50+ MB |
| Accuracy | Demo only | 95% |
| Backend API | ❌ No | ✅ FastAPI |
| Batch Processing | ❌ No | ✅ Yes |

## Use Cases

- Quick demos for stakeholders
- Concept presentations
- UI/UX testing
- Rapid prototyping
- Educational purposes

## Deployment

### Streamlit Cloud (Free)
1. Push to GitHub
2. Go to share.streamlit.io
3. Deploy `prototype/app_prototype.py`
4. Done!

### Local Sharing
```bash
streamlit run app_prototype.py --server.port 8501
```

Then share via ngrok or localtunnel.

## Upgrading to Full Version

To use the production-ready version with real ML models:
1. Go to parent directory
2. Install full requirements: `pip install -r requirements.txt`
3. Run: `streamlit run streamlit_app.py`

## License

Demo/Prototype - For educational and presentation purposes
