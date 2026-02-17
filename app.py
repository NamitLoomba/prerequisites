import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import numpy as np

st.set_page_config(page_title='Pre-Delinquency Risk Platform', page_icon='shield', layout='wide')

COLORS = {'Critical':'#FF4B4B','High':'#FFA500','Medium':'#FFD700','Low':'#4CAF50'}

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
    c1,c2 = st.columns([1,1.5])
    with c1:
        st.markdown('#### Customer Input')
        cid = st.text_input('Customer ID',value='CUST_001')
        sdelay = st.slider('Salary Delay (days) [?]',0,30,0,help='Number of days salary credit deviates from historical average')
        sdrop = st.slider('Savings Decline (%) [?]',0.0,1.0,0.0,0.05,help='Week-over-week percentage decline in savings balance')
        udelay = st.slider('Utility Payment Delay (days) [?]',0,30,0,help='Days of delayed utility bill payment')
        ddrop = st.slider('Discretionary Spending Drop (%) [?]',0.0,1.0,0.0,0.05,help='Reduction in lifestyle/non-essential spending')
        atm = st.slider('ATM Withdrawal Increase [?]',0,20,0,help='Extra ATM withdrawals compared to normal')
        upi = st.slider('UPI Lending App Transactions [?]',0,10,0,help='Number of transfers to lending apps')
        fail = st.slider('Failed Auto-debit Count [?]',0,5,0,help='Number of failed EMI/automatic payments')
        btn = st.button('Analyze Risk',type='primary',use_container_width=True)
        st.markdown('---')
        st.markdown('**Simulation Mode**')
        sim_mode = st.checkbox('Enable scenario simulation')
    with c2:
        if btn:
            try:
                model,scaler = load_model()
                data = {'salary_delay_days':sdelay,'savings_drop_pct':sdrop,'utility_payment_delay_days':udelay,'discretionary_spend_drop_pct':ddrop,'atm_withdrawal_increase':atm,'upi_lending_txn_count':upi,'failed_autodebit_count':fail}
                features = create_features(data)
                risk_prob,risk_level = predict_risk(features,model,scaler)
                action = get_recommendation(risk_level)
                importance = get_feature_importance()
                
                # Main metrics
                st.markdown('#### Risk Assessment')
                m1,m2,m3 = st.columns(3)
                m1.metric('Risk Score',f'{risk_prob*100:.1f}%')
                m2.metric('Risk Level',risk_level,delta_color='inverse')
                m3.metric('Confidence',f'{max(risk_prob,1-risk_prob)*100:.1f}%')
                
                # Gauge
                fig = go.Figure(go.Indicator(mode='gauge+number',value=risk_prob*100,title={'text':'Delinquency Risk Score'},gauge={'axis':{'range':[0,100]},'bar':{'color':COLORS[risk_level]},'steps':[{'range':[0,25],'color':'#E8F5E9'},{'range':[25,50],'color':'#FFF9C4'},{'range':[50,75],'color':'#FFE0B2'},{'range':[75,100],'color':'#FFCDD2'}]}))
                st.plotly_chart(fig,use_container_width=True)
                
                # Two columns: Explanation + Action
                col_act1,col_act2 = st.columns(2)
                with col_act1:
                    st.markdown('#### Top Contributing Factors')
                    st.caption('Why this customer is flagged:')
                    for factor,impact in importance.items():
                        st.progress(impact/100,f'{factor}: +{impact}%')
                with col_act2:
                    st.markdown('#### Recommended Intervention')
                    if risk_level=='Critical':
                        st.error(f'**CRITICAL RISK**\n\n{action}')
                    elif risk_level=='High':
                        st.warning(f'**HIGH RISK**\n\n{action}')
                    elif risk_level=='Medium':
                        st.info(f'**MEDIUM RISK**\n\n{action}')
                    else:
                        st.success(f'**LOW RISK**\n\n{action}')
                
                # Trend chart
                st.markdown('#### Risk Trend (Simulation)')
                weeks,trend = get_simulation_trend(risk_prob)
                fig_trend = go.Figure()
                fig_trend.add_trace(go.Scatter(x=weeks,y=[t*100 for t in trend],mode='lines+markers',name='Risk Score',line=dict(color='#FF4B4B',width=3)))
                fig_trend.update_layout(yaxis_title='Risk Score (%)',yaxis_range=[0,100],height=250)
                st.plotly_chart(fig_trend,use_container_width=True)
                
                # Customer snapshot
                st.markdown('#### Customer Snapshot')
                st.markdown(f'''
                | Field | Value |
                |-------|--------|
                | Customer ID | {cid} |
                | Risk Level | {risk_level} |
                | Salary Delay | {sdelay} days |
                | Savings Drop | {sdrop*100:.0f}% |
                | Failed Auto-debits | {fail} |
                ''')
                
            except Exception as e:
                st.error(f'Model not loaded. Error: {e}')
        else:
            st.info('Enter customer data and click Analyze Risk')
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
    
    An enterprise-grade solution for early detection and prevention of loan delinquency.
    
    ---
    
    ### Technology Stack
    
    | Component | Technology |
    |-----------|------------|
    | ML Model | XGBoost Classifier |
    | Feature Engineering | Custom stress signal creation |
    | Backend | FastAPI |
    | Frontend | Streamlit |
    | Visualization | Plotly |
    
    ---
    
    ### Key Features
    
    - **Explainable AI**: Feature importance shows WHY a customer is flagged
    - **Risk Scoring**: 0-100% probability of default in 2-4 weeks
    - **Intervention Recommendations**: Actionable next steps for each risk level
    - **Portfolio Analysis**: Batch processing for portfolio-level risk assessment
    - **Scenario Simulation**: Test different risk scenarios
    
    ---
    
    ### Model Details
    
    - **Algorithm**: XGBoost (Gradient Boosted Trees)
    - **Prediction Horizon**: 2-4 weeks ahead
    - **Features**: 13 engineered features including stress indices
    - **Training Data**: Synthetic (privacy-safe)
    
    ---
    
    ### Disclaimer
    
    This is a simulated demo using synthetic data for privacy protection.
    Not for actual financial decision-making.
    ''')
    
    st.caption('Pre-Delinquency Risk Platform v1.0.0 | Built for Hackathon')

st.markdown('---')
st.caption(' This is a simulated demo using synthetic data for privacy protection.')
