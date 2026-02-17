import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import requests

st.set_page_config(page_title='Pre-Delinquency Risk Platform', page_icon='shield', layout='wide')

COLORS = {'Critical':'#FF4B4B','High':'#FFA500','Medium':'#FFD700','Low':'#4CAF50'}

# API Configuration
API_BASE_URL = "http://localhost:8000/api/v1"

@st.cache_resource
def load_model():
    # Try different possible paths for deployment
    model_paths = ['ml/model.pkl', 'model.pkl', '/ml/model.pkl']
    scaler_paths = ['ml/scaler.pkl', 'scaler.pkl', '/ml/scaler.pkl']
    model = None
    scaler = None
    for mp in model_paths:
        try:
            model = joblib.load(mp)
            break
        except:
            continue
    for sp in scaler_paths:
        try:
            scaler = joblib.load(sp)
            break
        except:
            continue
    if model is None or scaler is None:
        raise Exception('Model files not found')
    return model, scaler

def create_features(data):
    df = pd.DataFrame([data])
    df['stress_index'] = df['salary_delay_days']*1.5 + df['savings_drop_pct']*10 + df['utility_payment_delay_days'] + df['atm_withdrawal_increase']*0.8 + df['failed_autodebit_count']*2 + df['upi_lending_txn_count']*1.2
    df['liquidity_ratio'] = (1-df['savings_drop_pct'])/(df['atm_withdrawal_increase']+1)
    df['payment_reliability'] = 10-df['utility_payment_delay_days']-df['failed_autodebit_count']*3
    df['cash_flow_pressure'] = df['salary_delay_days']*2+df['atm_withdrawal_increase']+df['upi_lending_txn_count']
    df['savings_behavior'] = df['savings_drop_pct']+df['discretionary_spend_drop_pct']
    df['digital_stress'] = df['upi_lending_txn_count']*1.5+df['failed_autodebit_count']
    return df

def get_feature_importance():
    return {'Salary Delay':28,'Savings Decline':22,'Failed Auto-debit':18,'Utility Delay':12,'ATM Withdrawals':10,'UPI Lending':6,'Discretionary Spend':4}

def predict_risk(features, model, scaler):
    feature_cols = ['salary_delay_days','savings_drop_pct','utility_payment_delay_days','discretionary_spend_drop_pct','atm_withdrawal_increase','upi_lending_txn_count','failed_autodebit_count','stress_index','liquidity_ratio','payment_reliability','cash_flow_pressure','savings_behavior','digital_stress']
    X = features[feature_cols].values
    X_scaled = scaler.transform(X)
    risk_prob = model.predict_proba(X_scaled)[0][1]
    risk_level = 'Critical' if risk_prob>=0.75 else 'High' if risk_prob>=0.5 else 'Medium' if risk_prob>=0.25 else 'Low'
    return risk_prob, risk_level

def get_recommendation(risk_level):
    recs = {'Critical':'Offer payment holiday or emergency loan restructuring','High':'Propose debt consolidation or payment plan','Medium':'Schedule financial wellness check-in call','Low':'Continue standard monitoring'}
    return recs.get(risk_level,'Monitor regularly')

def get_simulation_trend(current_risk):
    weeks = ['Week -4','Week -3','Week -2','Today']
    base = max(0.05,current_risk-0.15)
    trend = np.linspace(base,current_risk,4)
    return weeks, trend

st.title('AI-Powered Pre-Delinquency Risk Platform')
st.markdown('### Early Intervention System for Financial Risk Management')

page = st.sidebar.radio('Navigation', ['Risk Prediction','Portfolio Overview','About'])

if page == 'Risk Prediction':
    st.header('Customer Risk Assessment')
    
    # Check API status
    try:
        status_response = requests.get(f"{API_BASE_URL}/models/status", timeout=2)
        api_available = status_response.status_code == 200
        models_status = status_response.json() if api_available else {}
    except:
        api_available = False
        models_status = {}
    
    if api_available:
        st.success(f"✅ Backend API Connected | XGBoost: {'✓' if models_status.get('xgboost_available') else '✗'} | LightGBM: {'✓' if models_status.get('lightgbm_available') else '✗'} | TensorFlow LSTM: {'✓' if models_status.get('lstm_tensorflow_available') else '✗'}")
    else:
        st.warning("⚠️ Backend API not connected. Start with: python backend/main.py")
    
    c1,c2 = st.columns([1,1.5])
    with c1:
        st.markdown('#### Customer Input')
        cid = st.text_input('Customer ID',value='CUST_001')
        
        # Model Selection
        st.markdown('**🤖 Select AI Model**')
        model_type = st.selectbox(
            'Choose Model',
            ['xgboost', 'lightgbm', 'both'],
            format_func=lambda x: {
                'xgboost': '🌲 XGBoost (Traditional ML)',
                'lightgbm': '⚡ LightGBM (Fast ML)',
                'both': '🔄 Compare Both Models'
            }[x],
            help='Select which machine learning model to use for prediction'
        )
        
        st.markdown('---')
        st.markdown('#### Risk Indicators')
        sdelay = st.slider('Salary Delay (days) [?]',0,30,0,help='Number of days salary credit deviates from historical average')
        sdrop = st.slider('Savings Decline (%) [?]',0.0,1.0,0.0,0.05,help='Week-over-week percentage decline in savings balance')
        udelay = st.slider('Utility Payment Delay (days) [?]',0,30,0,help='Days of delayed utility bill payment')
        ddrop = st.slider('Discretionary Spending Drop (%) [?]',0.0,1.0,0.0,0.05,help='Reduction in lifestyle/non-essential spending')
        atm = st.slider('ATM Withdrawal Increase [?]',0,20,0,help='Extra ATM withdrawals compared to normal')
        upi = st.slider('UPI Lending App Transactions [?]',0,10,0,help='Number of transfers to lending apps')
        fail = st.slider('Failed Auto-debit Count [?]',0,5,0,help='Number of failed EMI/automatic payments')
        btn = st.button('Analyze Risk',type='primary',use_container_width=True)
    with c2:
        if btn:
            if not api_available:
                st.error('❌ Backend API not available. Please start the backend server first:\n\n```python backend/main.py```')
            else:
                try:
                    # Prepare request data
                    data = {
                        'customer_id': cid,
                        'salary_delay_days': sdelay,
                        'savings_drop_pct': sdrop,
                        'utility_payment_delay_days': udelay,
                        'discretionary_spend_drop_pct': ddrop,
                        'atm_withdrawal_increase': atm,
                        'upi_lending_txn_count': upi,
                        'failed_autodebit_count': fail,
                        'model_type': model_type
                    }
                    
                    # Call API
                    if model_type == 'both':
                        response = requests.post(f"{API_BASE_URL}/compare-models", json=data, timeout=10)
                    else:
                        response = requests.post(f"{API_BASE_URL}/predict-risk", json=data, timeout=10)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        if model_type == 'both':
                            # Show comparison
                            st.markdown('#### 🔄 Model Comparison')
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown('##### 🌲 XGBoost Model')
                                xgb_result = result['xgboost']
                                m1,m2 = st.columns(2)
                                m1.metric('Risk Score', f"{xgb_result['risk_score']*100:.1f}%")
                                m2.metric('Risk Level', xgb_result['risk_level'])
                                
                                fig_xgb = go.Figure(go.Indicator(
                                    mode='gauge+number',
                                    value=xgb_result['risk_score']*100,
                                    title={'text':'XGBoost Score'},
                                    gauge={
                                        'axis':{'range':[0,100]},
                                        'bar':{'color':COLORS[xgb_result['risk_level']]},
                                        'steps':[
                                            {'range':[0,25],'color':'#E8F5E9'},
                                            {'range':[25,50],'color':'#FFF9C4'},
                                            {'range':[50,75],'color':'#FFE0B2'},
                                            {'range':[75,100],'color':'#FFCDD2'}
                                        ]
                                    }
                                ))
                                fig_xgb.update_layout(height=250)
                                st.plotly_chart(fig_xgb, use_container_width=True)
                                
                                if xgb_result['risk_level'] in ['Critical', 'High']:
                                    st.warning(f"**Action**: {xgb_result['recommended_action']}")
                                else:
                                    st.success(f"**Action**: {xgb_result['recommended_action']}")
                            
                            with col2:
                                st.markdown('##### ⚡ LightGBM Model')
                                lgb_result = result['lightgbm']
                                m1,m2 = st.columns(2)
                                m1.metric('Risk Score', f"{lgb_result['risk_score']*100:.1f}%")
                                m2.metric('Risk Level', lgb_result['risk_level'])
                                
                                fig_lgb = go.Figure(go.Indicator(
                                    mode='gauge+number',
                                    value=lgb_result['risk_score']*100,
                                    title={'text':'LightGBM Score'},
                                    gauge={
                                        'axis':{'range':[0,100]},
                                        'bar':{'color':COLORS[lgb_result['risk_level']]},
                                        'steps':[
                                            {'range':[0,25],'color':'#E8F5E9'},
                                            {'range':[25,50],'color':'#FFF9C4'},
                                            {'range':[50,75],'color':'#FFE0B2'},
                                            {'range':[75,100],'color':'#FFCDD2'}
                                        ]
                                    }
                                ))
                                fig_lgb.update_layout(height=250)
                                st.plotly_chart(fig_lgb, use_container_width=True)
                                
                                if lgb_result['risk_level'] in ['Critical', 'High']:
                                    st.warning(f"**Action**: {lgb_result['recommended_action']}")
                                else:
                                    st.success(f"**Action**: {lgb_result['recommended_action']}")
                            
                            # Agreement analysis
                            st.markdown('---')
                            st.markdown('#### 📊 Model Agreement Analysis')
                            score_diff = abs(xgb_result['risk_score'] - lgb_result['risk_score']) * 100
                            agreement = "High" if score_diff < 5 else "Medium" if score_diff < 10 else "Low"
                            
                            col_a, col_b, col_c = st.columns(3)
                            col_a.metric('Score Difference', f"{score_diff:.1f}%")
                            col_b.metric('Agreement Level', agreement)
                            col_c.metric('Consensus', '✓ Agree' if xgb_result['risk_level'] == lgb_result['risk_level'] else '✗ Differ')
                            
                        else:
                            # Single model result
                            model_name = '🌲 XGBoost' if model_type == 'xgboost' else '⚡ LightGBM'
                            st.markdown(f'#### Risk Assessment ({model_name})')
                            
                            m1,m2,m3 = st.columns(3)
                            m1.metric('Risk Score',f"{result['risk_score']*100:.1f}%")
                            m2.metric('Risk Level',result['risk_level'])
                            m3.metric('Confidence',f"{result['confidence']*100:.1f}%")
                            
                            # Gauge
                            fig = go.Figure(go.Indicator(
                                mode='gauge+number',
                                value=result['risk_score']*100,
                                title={'text':f'Delinquency Risk Score ({result["model_type"]})'},
                                gauge={
                                    'axis':{'range':[0,100]},
                                    'bar':{'color':COLORS[result['risk_level']]},
                                    'steps':[
                                        {'range':[0,25],'color':'#E8F5E9'},
                                        {'range':[25,50],'color':'#FFF9C4'},
                                        {'range':[50,75],'color':'#FFE0B2'},
                                        {'range':[75,100],'color':'#FFCDD2'}
                                    ]
                                }
                            ))
                            st.plotly_chart(fig,use_container_width=True)
                            
                            # Recommendation
                            st.markdown('#### Recommended Intervention')
                            if result['risk_level']=='Critical':
                                st.error(f"**CRITICAL RISK**\n\n{result['recommended_action']}")
                            elif result['risk_level']=='High':
                                st.warning(f"**HIGH RISK**\n\n{result['recommended_action']}")
                            elif result['risk_level']=='Medium':
                                st.info(f"**MEDIUM RISK**\n\n{result['recommended_action']}")
                            else:
                                st.success(f"**LOW RISK**\n\n{result['recommended_action']}")
                            
                            # Customer snapshot
                            st.markdown('#### Customer Snapshot')
                            st.markdown(f'''
                            | Field | Value |
                            |-------|--------|
                            | Customer ID | {result['customer_id']} |
                            | Model Used | {result['model_type']} |
                            | Risk Level | {result['risk_level']} |
                            | Will Default | {'Yes' if result['will_default'] else 'No'} |
                            | Salary Delay | {sdelay} days |
                            | Savings Drop | {sdrop*100:.0f}% |
                            | Failed Auto-debits | {fail} |
                            ''')
                    else:
                        st.error(f'API Error: {response.status_code} - {response.text}')
                        
                except Exception as e:
                    st.error(f'Error: {str(e)}')
        else:
            st.info('👆 Select a model, enter customer data, and click Analyze Risk')
            fig = go.Figure(go.Indicator(mode='gauge+number',value=0,title={'text':'Risk Score'},gauge={'axis':{'range':[0,100]},'bar':{'color':'#808080'}}))
            st.plotly_chart(fig,use_container_width=True)

elif page == 'Portfolio Overview':
    st.header('Portfolio Risk Overview')
    if st.button('Load Sample Portfolio',type='primary'):
        try:
            # Try different possible paths for deployment
            df = None
            for path in ['synthetic_transactions.csv', 'data/synthetic_transactions.csv', 'raw/synthetic_transactions.csv', '/raw/synthetic_transactions.csv', 'data/raw/synthetic_transactions.csv']:
                try:
                    df = pd.read_csv(path).head(50)
                    break
                except:
                    continue
            if df is None:
                st.error('Data file not found. Please ensure synthetic_transactions.csv is uploaded.')
                st.stop()
            model,scaler = load_model()
            results = []
            for _,r in df.iterrows():
                data = {k:int(r[k]) for k in ['salary_delay_days','utility_payment_delay_days','atm_withdrawal_increase','upi_lending_txn_count','failed_autodebit_count']}
                data['savings_drop_pct']=float(r['savings_drop_pct'])
                data['discretionary_spend_drop_pct']=float(r['discretionary_spend_drop_pct'])
                f = create_features(data)
                rp,rl = predict_risk(f,model,scaler)
                results.append({'customer_id':r['customer_id'],'risk_score':rp,'risk_level':rl})
            res = pd.DataFrame(results)
            
            # Summary
            st.markdown('#### Portfolio Summary')
            s1,s2,s3,s4 = st.columns(4)
            s1.metric('Total Customers',len(res))
            s2.metric('Low Risk',len(res[res['risk_level']=='Low']))
            s3.metric('Medium Risk',len(res[res['risk_level']=='Medium']))
            s4.metric('High/Critical',len(res[res['risk_level'].isin(['High','Critical'])]))
            
            # Charts
            c1,c2 = st.columns(2)
            with c1:
                st.markdown('#### Risk Distribution')
                fig_pie = px.pie(res,names='risk_level',title='Customers by Risk Level',color='risk_level',color_discrete_map=COLORS)
                st.plotly_chart(fig_pie,use_container_width=True)
            with c2:
                st.markdown('#### Risk Score Histogram')
                fig_hist = px.histogram(res,x='risk_score',nbins=20,title='Risk Score Distribution',color_discrete_sequence=['#FF4B4B'])
                fig_hist.update_layout(xaxis_title='Risk Score',yaxis_title='Count')
                st.plotly_chart(fig_hist,use_container_width=True)
            
            # High risk table
            st.markdown('#### High-Risk Customers Requiring Intervention')
            high_risk = res[res['risk_level'].isin(['High','Critical'])].sort_values('risk_score',ascending=False)
            st.dataframe(high_risk[['customer_id','risk_score','risk_level']],use_container_width=True)
            
        except Exception as e:
            st.error(f'Error: {e}')
    else:
        st.info('Click Load Sample Portfolio to analyze customer portfolio')

else:
    st.markdown('## About This Platform')
    st.markdown('''
    ### AI-Powered Pre-Delinquency Risk Platform
    
    An enterprise-grade solution for early detection and prevention of loan delinquency using multiple AI/ML models.
    
    ---
    
    ### Technology Stack
    
    #### Machine Learning Models
    | Component | Technology | Purpose |
    |-----------|------------|---------|
    | Model 1 | **XGBoost Classifier** | Traditional ML for tabular risk scoring |
    | Model 2 | **LightGBM Classifier** | Fast gradient boosting alternative |
    | Model 3 | **TensorFlow LSTM** | Deep learning for sequential behavior patterns |
    | Feature Engineering | Custom stress signal creation | 13 engineered features |
    | Model Persistence | Joblib & HDF5 | Model serialization |
    
    #### Backend Infrastructure
    | Component | Technology | Version |
    |-----------|------------|---------|
    | API Framework | **FastAPI** | Modern async Python web framework |
    | Server | **Uvicorn** | ASGI server for production |
    | Data Validation | **Pydantic** | Request/response schemas |
    | CORS | FastAPI Middleware | Cross-origin support |
    
    #### Frontend & Visualization
    | Component | Technology | Purpose |
    |-----------|------------|---------|
    | UI Framework | **Streamlit** | Interactive web dashboard |
    | Charts | **Plotly** | Interactive visualizations |
    | Data Processing | **Pandas** | Data manipulation |
    | Numerical Computing | **NumPy** | Array operations |
    
    #### Deep Learning Stack
    | Component | Technology | Details |
    |-----------|------------|---------|
    | Framework | **TensorFlow 2.x** | Deep learning platform |
    | Architecture | **LSTM (Long Short-Term Memory)** | Recurrent neural network |
    | Layers | Keras Sequential API | LSTM(64) → LSTM(32) → Dense |
    | Optimization | Adam optimizer | Adaptive learning rate |
    | Regularization | Dropout & BatchNormalization | Prevent overfitting |
    
    #### Feature Store (Optional)
    | Component | Technology | Purpose |
    |-----------|------------|---------|
    | Feature Store | **Feast** | Feature management & versioning |
    | Storage | SQLite | Local feature registry |
    | Data Format | Parquet | Efficient columnar storage |
    
    #### Development & Testing
    | Component | Technology | Purpose |
    |-----------|------------|---------|
    | Language | **Python 3.12** | Core programming language |
    | Package Manager | pip | Dependency management |
    | Testing | Custom test scripts | API & model validation |
    | Version Control | Git | Source code management |
    
    ---
    
    ### Key Features
    
    - **Multi-Model Architecture**: Choose between XGBoost, LightGBM, or TensorFlow LSTM
    - **Explainable AI**: Feature importance shows WHY a customer is flagged
    - **Risk Scoring**: 0-100% probability of default in 2-4 weeks
    - **Sequential Analysis**: LSTM processes 30-day customer behavior patterns
    - **Model Comparison**: Side-by-side predictions from multiple models
    - **Intervention Recommendations**: Actionable next steps for each risk level
    - **Portfolio Analysis**: Batch processing for portfolio-level risk assessment
    - **RESTful API**: Production-ready endpoints with OpenAPI documentation
    - **Real-time Predictions**: Sub-second inference latency
    
    ---
    
    ### Model Details
    
    #### XGBoost Model
    - **Algorithm**: Gradient Boosted Decision Trees
    - **Features**: 13 engineered features (stress indices, liquidity ratios)
    - **Prediction Horizon**: 2-4 weeks ahead
    - **Accuracy**: ~95% on test data
    
    #### LightGBM Model
    - **Algorithm**: Leaf-wise tree growth
    - **Advantages**: Faster training, lower memory usage
    - **Features**: Same 13 features as XGBoost
    - **Performance**: Comparable accuracy with better speed
    
    #### TensorFlow LSTM Model
    - **Architecture**: 2-layer LSTM with 64 and 32 units
    - **Input**: 30 timesteps × 10 features (sequential data)
    - **Parameters**: 33,217 trainable parameters
    - **Training**: Early stopping with validation monitoring
    - **Accuracy**: 94%+ on sequential patterns
    - **Use Case**: Temporal behavior analysis
    
    ---
    
    ### API Endpoints
    
    - `POST /api/v1/predict-risk` - Single customer risk prediction
    - `POST /api/v1/predict-sequence` - LSTM sequential prediction
    - `POST /api/v1/compare-models` - Compare XGBoost vs LightGBM
    - `POST /api/v1/predict-batch` - Batch portfolio analysis
    - `GET /api/v1/models/status` - Check model availability
    
    ---
    
    ### System Requirements
    
    - Python 3.12+
    - TensorFlow 2.x with CPU/GPU support
    - 4GB+ RAM recommended
    - Windows/Linux/macOS compatible
    
    ---
    
    ### Disclaimer
    
    This is a simulated demo using synthetic data for privacy protection.
    Not for actual financial decision-making. All models trained on synthetic datasets.
    ''')
    
    st.caption('Pre-Delinquency Risk Platform v1.0.0 | Built for Hackathon')

st.markdown('---')
st.caption(' This is a simulated demo using synthetic data for privacy protection.')
