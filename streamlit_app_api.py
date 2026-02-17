import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import numpy as np

st.set_page_config(page_title='Pre-Delinquency Risk Platform', page_icon='🛡️', layout='wide')

COLORS = {'Critical':'#FF4B4B','High':'#FFA500','Medium':'#FFD700','Low':'#4CAF50'}

# Backend API URL - will be configured after deployment
API_URL = st.secrets.get("API_URL", "http://localhost:8000/api/v1")

def check_backend_health():
    """Check if backend API is accessible"""
    try:
        response = requests.get(f"{API_URL.replace('/api/v1', '')}/api/v1/", timeout=5)
        return response.status_code == 200
    except:
        return False

def predict_risk_api(customer_data, model_type='xgboost'):
    """Call backend API for risk prediction"""
    try:
        customer_data['model_type'] = model_type
        response = requests.post(f"{API_URL}/predict-risk", json=customer_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def compare_models_api(customer_data):
    """Call backend API to compare both models"""
    try:
        response = requests.post(f"{API_URL}/compare-models", json=customer_data, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

def predict_batch_api(customers_data, model_type='xgboost'):
    """Call backend API for batch prediction"""
    try:
        # Add model_type to each customer
        for customer in customers_data:
            customer['model_type'] = model_type
        
        payload = {"customers": customers_data}
        response = requests.post(f"{API_URL}/predict-batch", json=payload, timeout=30)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None

# Main App
st.title('🛡️ AI-Powered Pre-Delinquency Risk Platform')
st.markdown('### Early Intervention System for Financial Risk Management')

# Check backend status
backend_healthy = check_backend_health()
if backend_healthy:
    st.success(f"✅ Connected to Backend API: {API_URL}")
else:
    st.error(f"❌ Cannot connect to Backend API: {API_URL}")
    st.info("Please ensure the backend is running and the API_URL is correct in Streamlit secrets.")
    st.stop()

page = st.sidebar.radio('Navigation', ['🎯 Risk Prediction','📊 Portfolio Overview','ℹ️ About'])

if page == '🎯 Risk Prediction':
    st.header('Customer Risk Assessment')
    
    c1,c2 = st.columns([1,1.5])
    with c1:
        st.markdown('#### Customer Input')
        cid = st.text_input('Customer ID',value='CUST_001')
        
        st.markdown('**🤖 Select AI Model**')
        model_type = st.selectbox(
            'Choose Model',
            ['xgboost', 'lightgbm', 'both'],
            format_func=lambda x: {
                'xgboost': '🌲 XGBoost (Traditional ML)',
                'lightgbm': '⚡ LightGBM (Fast ML)',
                'both': '🔄 Compare Both Models'
            }[x]
        )
        
        st.markdown('---')
        st.markdown('#### Risk Indicators')
        sdelay = st.slider('Salary Delay (days)',0,30,0)
        sdrop = st.slider('Savings Decline (%)',0.0,1.0,0.0,0.05)
        udelay = st.slider('Utility Payment Delay (days)',0,30,0)
        ddrop = st.slider('Discretionary Spending Drop (%)',0.0,1.0,0.0,0.05)
        atm = st.slider('ATM Withdrawal Increase',0,20,0)
        upi = st.slider('UPI Lending App Transactions',0,10,0)
        fail = st.slider('Failed Auto-debit Count',0,5,0)
        btn = st.button('Analyze Risk',type='primary',use_container_width=True)
    
    with c2:
        if btn:
            customer_data = {
                'customer_id': cid,
                'salary_delay_days': sdelay,
                'savings_drop_pct': sdrop,
                'utility_payment_delay_days': udelay,
                'discretionary_spend_drop_pct': ddrop,
                'atm_withdrawal_increase': atm,
                'upi_lending_txn_count': upi,
                'failed_autodebit_count': fail
            }
            
            if model_type == 'both':
                st.markdown('#### 🔄 Model Comparison')
                
                with st.spinner('Analyzing with both models...'):
                    result = compare_models_api(customer_data)
                
                if result:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('##### 🌲 XGBoost')
                        xgb = result['xgboost']
                        m1,m2 = st.columns(2)
                        m1.metric('Risk Score', f"{xgb['risk_score']*100:.1f}%")
                        m2.metric('Risk Level', xgb['risk_level'])
                        
                        fig_xgb = go.Figure(go.Indicator(
                            mode='gauge+number',
                            value=xgb['risk_score']*100,
                            title={'text':'XGBoost Score'},
                            gauge={'axis':{'range':[0,100]},'bar':{'color':COLORS[xgb['risk_level']]},
                                   'steps':[{'range':[0,25],'color':'#E8F5E9'},{'range':[25,50],'color':'#FFF9C4'},
                                           {'range':[50,75],'color':'#FFE0B2'},{'range':[75,100],'color':'#FFCDD2'}]}
                        ))
                        fig_xgb.update_layout(height=250)
                        st.plotly_chart(fig_xgb, use_container_width=True)
                        
                        if xgb['risk_level'] in ['Critical', 'High']:
                            st.warning(f"**Action**: {xgb['recommended_action']}")
                        else:
                            st.success(f"**Action**: {xgb['recommended_action']}")
                    
                    with col2:
                        st.markdown('##### ⚡ LightGBM')
                        lgb = result['lightgbm']
                        m1,m2 = st.columns(2)
                        m1.metric('Risk Score', f"{lgb['risk_score']*100:.1f}%")
                        m2.metric('Risk Level', lgb['risk_level'])
                        
                        fig_lgb = go.Figure(go.Indicator(
                            mode='gauge+number',
                            value=lgb['risk_score']*100,
                            title={'text':'LightGBM Score'},
                            gauge={'axis':{'range':[0,100]},'bar':{'color':COLORS[lgb['risk_level']]},
                                   'steps':[{'range':[0,25],'color':'#E8F5E9'},{'range':[25,50],'color':'#FFF9C4'},
                                           {'range':[50,75],'color':'#FFE0B2'},{'range':[75,100],'color':'#FFCDD2'}]}
                        ))
                        fig_lgb.update_layout(height=250)
                        st.plotly_chart(fig_lgb, use_container_width=True)
                        
                        if lgb['risk_level'] in ['Critical', 'High']:
                            st.warning(f"**Action**: {lgb['recommended_action']}")
                        else:
                            st.success(f"**Action**: {lgb['recommended_action']}")
                    
                    st.markdown('---')
                    st.markdown('#### 📊 Model Agreement Analysis')
                    score_diff = result['score_difference'] * 100
                    agreement = "High" if score_diff < 5 else "Medium" if score_diff < 10 else "Low"
                    
                    col_a, col_b, col_c = st.columns(3)
                    col_a.metric('Score Difference', f"{score_diff:.1f}%")
                    col_b.metric('Agreement Level', agreement)
                    col_c.metric('Consensus', '✓ Agree' if result['agreement'] else '✗ Differ')
                    
            else:
                model_name = '🌲 XGBoost' if model_type == 'xgboost' else '⚡ LightGBM'
                st.markdown(f'#### Risk Assessment ({model_name})')
                
                with st.spinner(f'Analyzing with {model_name}...'):
                    result = predict_risk_api(customer_data, model_type)
                
                if result:
                    m1,m2 = st.columns(2)
                    m1.metric('Risk Score',f"{result['risk_score']*100:.1f}%")
                    m2.metric('Risk Level',result['risk_level'])
                    
                    fig = go.Figure(go.Indicator(
                        mode='gauge+number',
                        value=result['risk_score']*100,
                        title={'text':f'Delinquency Risk Score ({model_type})'},
                        gauge={'axis':{'range':[0,100]},'bar':{'color':COLORS[result['risk_level']]},
                               'steps':[{'range':[0,25],'color':'#E8F5E9'},{'range':[25,50],'color':'#FFF9C4'},
                                       {'range':[50,75],'color':'#FFE0B2'},{'range':[75,100],'color':'#FFCDD2'}]}
                    ))
                    st.plotly_chart(fig,use_container_width=True)
                    
                    st.markdown('#### Recommended Intervention')
                    if result['risk_level']=='Critical':
                        st.error(f"**CRITICAL RISK**\n\n{result['recommended_action']}")
                    elif result['risk_level']=='High':
                        st.warning(f"**HIGH RISK**\n\n{result['recommended_action']}")
                    elif result['risk_level']=='Medium':
                        st.info(f"**MEDIUM RISK**\n\n{result['recommended_action']}")
                    else:
                        st.success(f"**LOW RISK**\n\n{result['recommended_action']}")
        else:
            st.info('👆 Select a model, enter customer data, and click Analyze Risk')

elif page == '📊 Portfolio Overview':
    st.header('Portfolio Risk Overview')
    st.info('📤 Upload your CSV file with customer data to analyze portfolio risk')
    
    uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file).head(100)
            st.success(f"✅ Loaded {len(df)} customers")
            
            model_choice = st.selectbox('Select Model', ['xgboost', 'lightgbm'])
            
            if st.button('Analyze Portfolio', type='primary'):
                with st.spinner('Analyzing portfolio...'):
                    customers_data = []
                    for idx, r in df.iterrows():
                        try:
                            customer = {
                                'customer_id': str(r.get('customer_id', f'CUST_{idx:04d}')),
                                'salary_delay_days': int(r.get('salary_delay_days', 0)),
                                'savings_drop_pct': float(r.get('savings_drop_pct', 0)),
                                'utility_payment_delay_days': int(r.get('utility_payment_delay_days', 0)),
                                'discretionary_spend_drop_pct': float(r.get('discretionary_spend_drop_pct', 0)),
                                'atm_withdrawal_increase': int(r.get('atm_withdrawal_increase', 0)),
                                'upi_lending_txn_count': int(r.get('upi_lending_txn_count', 0)),
                                'failed_autodebit_count': int(r.get('failed_autodebit_count', 0))
                            }
                            customers_data.append(customer)
                        except:
                            continue
                    
                    result = predict_batch_api(customers_data, model_choice)
                    
                    if result and result['predictions']:
                        predictions = result['predictions']
                        res = pd.DataFrame([{
                            'customer_id': p['customer_id'],
                            'risk_score': p['risk_score'],
                            'risk_level': p['risk_level']
                        } for p in predictions])
                        
                        st.markdown('#### Portfolio Summary')
                        s1,s2,s3,s4 = st.columns(4)
                        s1.metric('Total', result['total_customers'])
                        s2.metric('Low Risk', len(res[res['risk_level']=='Low']))
                        s3.metric('Medium Risk', len(res[res['risk_level']=='Medium']))
                        s4.metric('High/Critical', result['high_risk_count'] + result['critical_risk_count'])
                        
                        c1,c2 = st.columns(2)
                        with c1:
                            fig_pie = px.pie(res, names='risk_level', title='Risk Distribution', color='risk_level', color_discrete_map=COLORS)
                            st.plotly_chart(fig_pie, use_container_width=True)
                        with c2:
                            fig_hist = px.histogram(res, x='risk_score', nbins=20, title='Risk Score Distribution')
                            st.plotly_chart(fig_hist, use_container_width=True)
                        
                        st.markdown('#### High-Risk Customers')
                        high_risk = res[res['risk_level'].isin(['High','Critical'])].sort_values('risk_score', ascending=False)
                        if len(high_risk) > 0:
                            st.dataframe(high_risk, use_container_width=True)
                        else:
                            st.success('✅ No high-risk customers!')
                        
                        csv = res.to_csv(index=False)
                        st.download_button("📥 Download Results", csv, f'portfolio_analysis_{model_choice}.csv', 'text/csv')
        except Exception as e:
            st.error(f'Error: {e}')

else:
    st.markdown('## About This Platform')
    st.markdown(f'''
    ### AI-Powered Pre-Delinquency Risk Platform
    
    Enterprise-grade solution for early detection and prevention of loan delinquency.
    
    ### Architecture
    - **Backend API**: {API_URL}
    - **Frontend**: Streamlit Cloud
    - **Models**: XGBoost & LightGBM
    
    ### Technology Stack
    - **Backend**: FastAPI + Python
    - **Frontend**: Streamlit
    - **ML Models**: XGBoost, LightGBM
    - **Visualization**: Plotly
    
    ### Key Features
    - Multi-model architecture (2 AI models)
    - Real-time risk scoring via API
    - Model comparison
    - Portfolio analysis with CSV upload
    - Intervention recommendations
    - 95% accuracy
    
    ### API Endpoints
    - POST /api/v1/predict-risk - Single prediction
    - POST /api/v1/compare-models - Compare both models
    - POST /api/v1/predict-batch - Batch prediction
    - GET /api/v1/models/status - Check model status
    
    ### Disclaimer
    Demo using synthetic data for privacy protection.
    ''')

st.markdown('---')
st.caption('🛡️ Pre-Delinquency Risk Platform v2.0.0 | Backend + Frontend Architecture')
