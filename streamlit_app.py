import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np
import os

# TensorFlow imports
try:
    import tensorflow as tf
    from ml.sequence_model import load_sequence_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    st.warning("TensorFlow not available. LSTM model will be disabled.")

st.set_page_config(page_title='Pre-Delinquency Risk Platform', page_icon='🛡️', layout='wide')

COLORS = {'Critical':'#FF4B4B','High':'#FFA500','Medium':'#FFD700','Low':'#4CAF50'}

@st.cache_resource
def load_models():
    """Load all ML models"""
    try:
        models = {
            'xgboost': {
                'model': joblib.load('ml/model.pkl'),
                'scaler': joblib.load('ml/scaler.pkl')
            },
            'lightgbm': {
                'model': joblib.load('ml/model_lgb.pkl'),
                'scaler': joblib.load('ml/scaler_lgb.pkl')
            }
        }
        
        # Try to load TensorFlow LSTM model
        if TENSORFLOW_AVAILABLE:
            try:
                models['lstm'] = load_sequence_model('ml/sequence_model.h5', 'ml/sequence_scaler.pkl')
            except Exception as e:
                st.warning(f"LSTM model not available: {e}")
        
        return models
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

def create_features(data):
    """Engineer features from raw inputs"""
    df = pd.DataFrame([data])
    df['stress_index'] = (df['salary_delay_days']*1.5 + df['savings_drop_pct']*10 + 
                          df['utility_payment_delay_days'] + df['atm_withdrawal_increase']*0.8 + 
                          df['failed_autodebit_count']*2 + df['upi_lending_txn_count']*1.2)
    df['liquidity_ratio'] = (1-df['savings_drop_pct'])/(df['atm_withdrawal_increase']+1)
    df['payment_reliability'] = 10-df['utility_payment_delay_days']-df['failed_autodebit_count']*3
    df['cash_flow_pressure'] = df['salary_delay_days']*2+df['atm_withdrawal_increase']+df['upi_lending_txn_count']
    df['savings_behavior'] = df['savings_drop_pct']+df['discretionary_spend_drop_pct']
    df['digital_stress'] = df['upi_lending_txn_count']*1.5+df['failed_autodebit_count']
    return df

def predict_risk(features, model, scaler):
    """Make risk prediction"""
    feature_cols = ['salary_delay_days','savings_drop_pct','utility_payment_delay_days',
                   'discretionary_spend_drop_pct','atm_withdrawal_increase','upi_lending_txn_count',
                   'failed_autodebit_count','stress_index','liquidity_ratio','payment_reliability',
                   'cash_flow_pressure','savings_behavior','digital_stress']
    X = features[feature_cols].values
    X_scaled = scaler.transform(X)
    risk_prob = model.predict_proba(X_scaled)[0][1]
    risk_level = 'Critical' if risk_prob>=0.75 else 'High' if risk_prob>=0.5 else 'Medium' if risk_prob>=0.25 else 'Low'
    return risk_prob, risk_level

def get_recommendation(risk_level):
    """Get intervention recommendation"""
    recs = {
        'Critical':'Offer payment holiday or emergency loan restructuring',
        'High':'Propose debt consolidation or payment plan',
        'Medium':'Schedule financial wellness check-in call',
        'Low':'Continue standard monitoring'
    }
    return recs.get(risk_level,'Monitor regularly')

# Main App
st.title('🛡️ AI-Powered Pre-Delinquency Risk Platform')
st.markdown('### Early Intervention System for Financial Risk Management')

# Load models
models = load_models()
if models is None:
    st.error("Failed to load models. Please ensure model files are in the ml/ directory.")
    st.stop()

# Show model status
model_status = "✅ Models Loaded | XGBoost: ✓ | LightGBM: ✓"
if 'lstm' in models:
    model_status += " | TensorFlow LSTM: ✓"
st.success(model_status)

page = st.sidebar.radio('Navigation', ['🎯 Risk Prediction','📊 Portfolio Overview','ℹ️ About'])

if page == '🎯 Risk Prediction':
    st.header('Customer Risk Assessment')
    
    c1,c2 = st.columns([1,1.5])
    with c1:
        st.markdown('#### Customer Input')
        cid = st.text_input('Customer ID',value='CUST_001')
        
        # Model Selection
        st.markdown('**🤖 Select AI Model**')
        
        model_options = ['xgboost', 'lightgbm', 'both']
        model_labels = {
            'xgboost': '🌲 XGBoost (Traditional ML)',
            'lightgbm': '⚡ LightGBM (Fast ML)',
            'both': '🔄 Compare XGBoost vs LightGBM'
        }
        
        # Add LSTM option if available
        if 'lstm' in models:
            model_options.extend(['lstm', 'all'])
            model_labels['lstm'] = '🧠 TensorFlow LSTM (Deep Learning)'
            model_labels['all'] = '🔬 Compare All 3 Models'
        
        model_type = st.selectbox(
            'Choose Model',
            model_options,
            format_func=lambda x: model_labels[x]
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
            # Prepare data
            data = {
                'salary_delay_days': sdelay,
                'savings_drop_pct': sdrop,
                'utility_payment_delay_days': udelay,
                'discretionary_spend_drop_pct': ddrop,
                'atm_withdrawal_increase': atm,
                'upi_lending_txn_count': upi,
                'failed_autodebit_count': fail
            }
            
            features = create_features(data)
            
            if model_type == 'lstm':
                # LSTM requires sequence data (30 days)
                st.markdown('#### 🧠 TensorFlow LSTM Prediction')
                st.info('LSTM model analyzes 30-day behavioral sequences. Generating synthetic sequence from current indicators...')
                
                # Generate a 30-day sequence from current data point
                # In production, this would be actual historical data
                base_features = np.array([
                    sdelay, sdrop, udelay, ddrop, atm, upi, fail,
                    features['stress_index'].values[0],
                    features['liquidity_ratio'].values[0],
                    features['payment_reliability'].values[0]
                ])
                
                # Create sequence with some variation
                sequence = np.zeros((30, 10))
                for i in range(30):
                    # Add temporal variation
                    noise = np.random.normal(0, 0.1, 10)
                    trend = (i / 30) * 0.2  # Gradual increase
                    sequence[i] = base_features * (1 + trend + noise)
                
                # Predict
                result = models['lstm'].predict_sequence(sequence)
                
                m1, m2 = st.columns(2)
                m1.metric('Risk Score', f"{result['risk_score']*100:.1f}%")
                m2.metric('Risk Level', result['risk_level'])
                
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=result['risk_score']*100,
                    title={'text':'LSTM Risk Score (30-day sequence)'},
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
                st.plotly_chart(fig, use_container_width=True)
                
                # Recommendation
                rec = get_recommendation(result['risk_level'])
                st.markdown('#### Recommended Intervention')
                if result['risk_level']=='Critical':
                    st.error(f"**CRITICAL RISK**\n\n{rec}")
                elif result['risk_level']=='High':
                    st.warning(f"**HIGH RISK**\n\n{rec}")
                elif result['risk_level']=='Medium':
                    st.info(f"**MEDIUM RISK**\n\n{rec}")
                else:
                    st.success(f"**LOW RISK**\n\n{rec}")
                
                st.info('💡 **Note**: LSTM model analyzes temporal patterns over 30 days. For demo purposes, a synthetic sequence was generated from current indicators.')
            
            elif model_type == 'all':
                # Compare all 3 models
                st.markdown('#### 🔬 All Models Comparison')
                
                col1, col2, col3 = st.columns(3)
                
                # XGBoost
                with col1:
                    st.markdown('##### 🌲 XGBoost')
                    xgb_score, xgb_level = predict_risk(features, models['xgboost']['model'], models['xgboost']['scaler'])
                    st.metric('Risk Score', f"{xgb_score*100:.1f}%")
                    st.metric('Risk Level', xgb_level)
                    
                    fig_xgb = go.Figure(go.Indicator(
                        mode='gauge+number',
                        value=xgb_score*100,
                        title={'text':'XGBoost'},
                        gauge={
                            'axis':{'range':[0,100]},
                            'bar':{'color':COLORS[xgb_level]},
                            'steps':[
                                {'range':[0,25],'color':'#E8F5E9'},
                                {'range':[25,50],'color':'#FFF9C4'},
                                {'range':[50,75],'color':'#FFE0B2'},
                                {'range':[75,100],'color':'#FFCDD2'}
                            ]
                        }
                    ))
                    fig_xgb.update_layout(height=200)
                    st.plotly_chart(fig_xgb, use_container_width=True)
                
                # LightGBM
                with col2:
                    st.markdown('##### ⚡ LightGBM')
                    lgb_score, lgb_level = predict_risk(features, models['lightgbm']['model'], models['lightgbm']['scaler'])
                    st.metric('Risk Score', f"{lgb_score*100:.1f}%")
                    st.metric('Risk Level', lgb_level)
                    
                    fig_lgb = go.Figure(go.Indicator(
                        mode='gauge+number',
                        value=lgb_score*100,
                        title={'text':'LightGBM'},
                        gauge={
                            'axis':{'range':[0,100]},
                            'bar':{'color':COLORS[lgb_level]},
                            'steps':[
                                {'range':[0,25],'color':'#E8F5E9'},
                                {'range':[25,50],'color':'#FFF9C4'},
                                {'range':[50,75],'color':'#FFE0B2'},
                                {'range':[75,100],'color':'#FFCDD2'}
                            ]
                        }
                    ))
                    fig_lgb.update_layout(height=200)
                    st.plotly_chart(fig_lgb, use_container_width=True)
                
                # LSTM
                with col3:
                    st.markdown('##### 🧠 LSTM')
                    # Generate sequence for LSTM
                    base_features = np.array([
                        sdelay, sdrop, udelay, ddrop, atm, upi, fail,
                        features['stress_index'].values[0],
                        features['liquidity_ratio'].values[0],
                        features['payment_reliability'].values[0]
                    ])
                    sequence = np.zeros((30, 10))
                    for i in range(30):
                        noise = np.random.normal(0, 0.1, 10)
                        trend = (i / 30) * 0.2
                        sequence[i] = base_features * (1 + trend + noise)
                    
                    lstm_result = models['lstm'].predict_sequence(sequence)
                    st.metric('Risk Score', f"{lstm_result['risk_score']*100:.1f}%")
                    st.metric('Risk Level', lstm_result['risk_level'])
                    
                    fig_lstm = go.Figure(go.Indicator(
                        mode='gauge+number',
                        value=lstm_result['risk_score']*100,
                        title={'text':'LSTM'},
                        gauge={
                            'axis':{'range':[0,100]},
                            'bar':{'color':COLORS[lstm_result['risk_level']]},
                            'steps':[
                                {'range':[0,25],'color':'#E8F5E9'},
                                {'range':[25,50],'color':'#FFF9C4'},
                                {'range':[50,75],'color':'#FFE0B2'},
                                {'range':[75,100],'color':'#FFCDD2'}
                            ]
                        }
                    ))
                    fig_lstm.update_layout(height=200)
                    st.plotly_chart(fig_lstm, use_container_width=True)
                
                # Consensus analysis
                st.markdown('---')
                st.markdown('#### 📊 Model Consensus Analysis')
                scores = [xgb_score, lgb_score, lstm_result['risk_score']]
                levels = [xgb_level, lgb_level, lstm_result['risk_level']]
                
                avg_score = np.mean(scores)
                std_score = np.std(scores)
                consensus = "High" if std_score < 0.05 else "Medium" if std_score < 0.10 else "Low"
                
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric('Average Score', f"{avg_score*100:.1f}%")
                col_b.metric('Score Std Dev', f"{std_score*100:.1f}%")
                col_c.metric('Consensus', consensus)
                col_d.metric('Agreement', f"{len(set(levels))} levels")
                
            elif model_type == 'both':
                # Compare both models
                st.markdown('#### 🔄 Model Comparison')
                
                col1, col2 = st.columns(2)
                
                # XGBoost
                with col1:
                    st.markdown('##### 🌲 XGBoost Model')
                    xgb_score, xgb_level = predict_risk(features, models['xgboost']['model'], models['xgboost']['scaler'])
                    m1,m2 = st.columns(2)
                    m1.metric('Risk Score', f"{xgb_score*100:.1f}%")
                    m2.metric('Risk Level', xgb_level)
                    
                    fig_xgb = go.Figure(go.Indicator(
                        mode='gauge+number',
                        value=xgb_score*100,
                        title={'text':'XGBoost Score'},
                        gauge={
                            'axis':{'range':[0,100]},
                            'bar':{'color':COLORS[xgb_level]},
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
                    
                    rec = get_recommendation(xgb_level)
                    if xgb_level in ['Critical', 'High']:
                        st.warning(f"**Action**: {rec}")
                    else:
                        st.success(f"**Action**: {rec}")
                
                # LightGBM
                with col2:
                    st.markdown('##### ⚡ LightGBM Model')
                    lgb_score, lgb_level = predict_risk(features, models['lightgbm']['model'], models['lightgbm']['scaler'])
                    m1,m2 = st.columns(2)
                    m1.metric('Risk Score', f"{lgb_score*100:.1f}%")
                    m2.metric('Risk Level', lgb_level)
                    
                    fig_lgb = go.Figure(go.Indicator(
                        mode='gauge+number',
                        value=lgb_score*100,
                        title={'text':'LightGBM Score'},
                        gauge={
                            'axis':{'range':[0,100]},
                            'bar':{'color':COLORS[lgb_level]},
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
                    
                    rec = get_recommendation(lgb_level)
                    if lgb_level in ['Critical', 'High']:
                        st.warning(f"**Action**: {rec}")
                    else:
                        st.success(f"**Action**: {rec}")
                
                # Agreement analysis
                st.markdown('---')
                st.markdown('#### 📊 Model Agreement Analysis')
                score_diff = abs(xgb_score - lgb_score) * 100
                agreement = "High" if score_diff < 5 else "Medium" if score_diff < 10 else "Low"
                
                col_a, col_b, col_c = st.columns(3)
                col_a.metric('Score Difference', f"{score_diff:.1f}%")
                col_b.metric('Agreement Level', agreement)
                col_c.metric('Consensus', '✓ Agree' if xgb_level == lgb_level else '✗ Differ')
                
            else:
                # Single model
                model_name = '🌲 XGBoost' if model_type == 'xgboost' else '⚡ LightGBM'
                st.markdown(f'#### Risk Assessment ({model_name})')
                
                risk_score, risk_level = predict_risk(features, models[model_type]['model'], models[model_type]['scaler'])
                
                m1,m2 = st.columns(2)
                m1.metric('Risk Score',f"{risk_score*100:.1f}%")
                m2.metric('Risk Level',risk_level)
                
                # Gauge
                fig = go.Figure(go.Indicator(
                    mode='gauge+number',
                    value=risk_score*100,
                    title={'text':f'Delinquency Risk Score ({model_type})'},
                    gauge={
                        'axis':{'range':[0,100]},
                        'bar':{'color':COLORS[risk_level]},
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
                rec = get_recommendation(risk_level)
                st.markdown('#### Recommended Intervention')
                if risk_level=='Critical':
                    st.error(f"**CRITICAL RISK**\n\n{rec}")
                elif risk_level=='High':
                    st.warning(f"**HIGH RISK**\n\n{rec}")
                elif risk_level=='Medium':
                    st.info(f"**MEDIUM RISK**\n\n{rec}")
                else:
                    st.success(f"**LOW RISK**\n\n{rec}")
        else:
            st.info('👆 Select a model, enter customer data, and click Analyze Risk')

elif page == '📊 Portfolio Overview':
    st.header('Portfolio Risk Overview')
    
    # Option 1: Upload CSV
    st.markdown('### Option 1: Upload Customer Data')
    uploaded_file = st.file_uploader("Upload CSV file with customer data", type=['csv'])
    
    # Option 2: Load sample data
    st.markdown('### Option 2: Use Sample Data')
    use_sample = st.button('Load Sample Portfolio', type='primary')
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} customers from uploaded file")
            process_portfolio = True
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            process_portfolio = False
    elif use_sample:
        try:
            # Try different possible paths for the sample data
            df = None
            for path in ['data/raw/synthetic_transactions.csv', 'synthetic_transactions.csv']:
                try:
                    df = pd.read_csv(path).head(50)
                    st.success(f"✅ Loaded {len(df)} sample customers")
                    break
                except:
                    continue
            
            if df is None:
                st.error('Sample data file not found. Please upload your own CSV file.')
                process_portfolio = False
            else:
                process_portfolio = True
        except Exception as e:
            st.error(f'Error loading sample data: {e}')
            process_portfolio = False
    else:
        st.info('👆 Upload a CSV file or click "Load Sample Portfolio" to analyze customer portfolio')
        process_portfolio = False
    
    if process_portfolio and df is not None:
        # Select model for portfolio analysis
        st.markdown('### Select Model for Analysis')
        portfolio_model = st.selectbox(
            'Choose model',
            ['xgboost', 'lightgbm', 'lstm'] if 'lstm' in models else ['xgboost', 'lightgbm'],
            format_func=lambda x: {
                'xgboost': '🌲 XGBoost',
                'lightgbm': '⚡ LightGBM',
                'lstm': '🧠 TensorFlow LSTM'
            }[x]
        )
        
        if st.button('Analyze Portfolio', type='primary'):
            with st.spinner('Analyzing portfolio...'):
                results = []
                
                for idx, r in df.iterrows():
                    try:
                        # Extract features from row
                        data = {
                            'salary_delay_days': int(r.get('salary_delay_days', 0)),
                            'savings_drop_pct': float(r.get('savings_drop_pct', 0)),
                            'utility_payment_delay_days': int(r.get('utility_payment_delay_days', 0)),
                            'discretionary_spend_drop_pct': float(r.get('discretionary_spend_drop_pct', 0)),
                            'atm_withdrawal_increase': int(r.get('atm_withdrawal_increase', 0)),
                            'upi_lending_txn_count': int(r.get('upi_lending_txn_count', 0)),
                            'failed_autodebit_count': int(r.get('failed_autodebit_count', 0))
                        }
                        
                        if portfolio_model == 'lstm':
                            # Generate sequence for LSTM
                            features = create_features(data)
                            base_features = np.array([
                                data['salary_delay_days'],
                                data['savings_drop_pct'],
                                data['utility_payment_delay_days'],
                                data['discretionary_spend_drop_pct'],
                                data['atm_withdrawal_increase'],
                                data['upi_lending_txn_count'],
                                data['failed_autodebit_count'],
                                features['stress_index'].values[0],
                                features['liquidity_ratio'].values[0],
                                features['payment_reliability'].values[0]
                            ])
                            
                            sequence = np.zeros((30, 10))
                            for i in range(30):
                                noise = np.random.normal(0, 0.1, 10)
                                trend = (i / 30) * 0.2
                                sequence[i] = base_features * (1 + trend + noise)
                            
                            result = models['lstm'].predict_sequence(sequence)
                            rp = result['risk_score']
                            rl = result['risk_level']
                        else:
                            # Traditional ML models
                            f = create_features(data)
                            rp, rl = predict_risk(f, models[portfolio_model]['model'], models[portfolio_model]['scaler'])
                        
                        results.append({
                            'customer_id': r.get('customer_id', f'CUST_{idx:04d}'),
                            'risk_score': rp,
                            'risk_level': rl
                        })
                    except Exception as e:
                        st.warning(f"Skipped row {idx}: {e}")
                        continue
                
                if len(results) == 0:
                    st.error('No valid results. Please check your CSV format.')
                    st.stop()
                
                res = pd.DataFrame(results)
                
                # Summary
                st.markdown('#### Portfolio Summary')
                s1, s2, s3, s4 = st.columns(4)
                s1.metric('Total Customers', len(res))
                s2.metric('Low Risk', len(res[res['risk_level']=='Low']))
                s3.metric('Medium Risk', len(res[res['risk_level']=='Medium']))
                s4.metric('High/Critical', len(res[res['risk_level'].isin(['High','Critical'])]))
                
                # Charts
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown('#### Risk Distribution')
                    fig_pie = px.pie(res, names='risk_level', title='Customers by Risk Level', 
                                    color='risk_level', color_discrete_map=COLORS)
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with c2:
                    st.markdown('#### Risk Score Histogram')
                    fig_hist = px.histogram(res, x='risk_score', nbins=20, 
                                          title='Risk Score Distribution',
                                          color_discrete_sequence=['#FF4B4B'])
                    fig_hist.update_layout(xaxis_title='Risk Score', yaxis_title='Count')
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                # High risk table
                st.markdown('#### High-Risk Customers Requiring Intervention')
                high_risk = res[res['risk_level'].isin(['High','Critical'])].sort_values('risk_score', ascending=False)
                
                if len(high_risk) > 0:
                    st.dataframe(high_risk[['customer_id','risk_score','risk_level']], use_container_width=True)
                else:
                    st.success('✅ No high-risk customers found!')
                
                # Download results
                st.markdown('#### Download Results')
                csv = res.to_csv(index=False)
                st.download_button(
                    label="📥 Download Portfolio Analysis (CSV)",
                    data=csv,
                    file_name=f'portfolio_analysis_{portfolio_model}.csv',
                    mime='text/csv'
                )
                
        # Show expected CSV format
        st.markdown('---')
        st.markdown('### Expected CSV Format')
        st.markdown('''
        Your CSV should have these columns:
        - `customer_id` (optional)
        - `salary_delay_days`
        - `savings_drop_pct`
        - `utility_payment_delay_days`
        - `discretionary_spend_drop_pct`
        - `atm_withdrawal_increase`
        - `upi_lending_txn_count`
        - `failed_autodebit_count`
        ''')
        
        # Show sample format
        sample_df = pd.DataFrame({
            'customer_id': ['CUST_001', 'CUST_002'],
            'salary_delay_days': [3, 7],
            'savings_drop_pct': [0.2, 0.5],
            'utility_payment_delay_days': [2, 5],
            'discretionary_spend_drop_pct': [0.15, 0.4],
            'atm_withdrawal_increase': [2, 5],
            'upi_lending_txn_count': [1, 3],
            'failed_autodebit_count': [0, 2]
        })
        st.dataframe(sample_df, use_container_width=True)
    
elif page == 'ℹ️ About':
    st.markdown('## About This Platform')
    st.markdown('''
    ### AI-Powered Pre-Delinquency Risk Platform
    
    An enterprise-grade solution for early detection and prevention of loan delinquency.
    
    ### Technology Stack
    - **XGBoost**: Traditional gradient boosting (95% accuracy)
    - **LightGBM**: Fast gradient boosting alternative (95% accuracy)
    - **TensorFlow LSTM**: Deep learning for sequential patterns (94% accuracy)
    - **Streamlit**: Interactive web interface
    - **Plotly**: Data visualization
    - **scikit-learn**: Feature engineering
    
    ### Key Features
    - Multi-model architecture (3 AI models)
    - Real-time risk scoring
    - Model comparison (side-by-side)
    - Sequential behavior analysis (LSTM)
    - Intervention recommendations
    - 94-95% accuracy across all models
    
    ### Model Details
    
    #### XGBoost Classifier
    - Algorithm: Gradient Boosted Decision Trees
    - Features: 13 engineered features
    - Use case: General purpose risk scoring
    
    #### LightGBM Classifier
    - Algorithm: Leaf-wise tree growth
    - Features: Same 13 features
    - Use case: Fast inference, production deployment
    
    #### TensorFlow LSTM
    - Architecture: 2-layer LSTM (64→32 units)
    - Input: 30-day behavioral sequences
    - Parameters: 33,217 trainable
    - Use case: Temporal pattern detection
    
    ### Disclaimer
    This is a demo using synthetic data for privacy protection.
    ''')

st.markdown('---')
st.caption('🛡️ Pre-Delinquency Risk Platform v1.0.0 | Built for Hackathon')
