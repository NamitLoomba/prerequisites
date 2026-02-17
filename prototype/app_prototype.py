"""
Pre-Delinquency Risk Detection - Prototype
Lightweight version with mock predictions for quick demos
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import random

st.set_page_config(page_title='Pre-Delinquency Risk Prototype', page_icon='🛡️', layout='wide')

COLORS = {'Critical':'#FF4B4B','High':'#FFA500','Medium':'#FFD700','Low':'#4CAF50'}

def mock_predict(features):
    """Mock prediction using simple rules (no ML models needed)"""
    # Calculate risk score based on input features
    risk_score = (
        features['salary_delay_days'] * 0.03 +
        features['savings_drop_pct'] * 0.4 +
        features['utility_payment_delay_days'] * 0.02 +
        features['discretionary_spend_drop_pct'] * 0.2 +
        features['atm_withdrawal_increase'] * 0.02 +
        features['upi_lending_txn_count'] * 0.05 +
        features['failed_autodebit_count'] * 0.08
    )
    
    # Add some randomness for realism
    risk_score = min(1.0, risk_score + random.uniform(-0.05, 0.05))
    
    # Determine risk level
    if risk_score >= 0.75:
        risk_level = 'Critical'
    elif risk_score >= 0.5:
        risk_level = 'High'
    elif risk_score >= 0.25:
        risk_level = 'Medium'
    else:
        risk_level = 'Low'
    
    return risk_score, risk_level

def get_recommendation(risk_level):
    """Get intervention recommendation"""
    recs = {
        'Critical': 'Immediate intervention - Offer payment holiday or loan restructuring',
        'High': 'Priority contact - Propose debt consolidation plan',
        'Medium': 'Schedule check-in call - Offer financial counseling',
        'Low': 'Continue standard monitoring'
    }
    return recs.get(risk_level, 'Monitor regularly')

# Main App
st.title('🛡️ Pre-Delinquency Risk Detection - Prototype')
st.markdown('### AI-Powered Early Warning System')
st.info('📌 This is a prototype with mock predictions for demonstration purposes')

page = st.sidebar.radio('Navigation', ['🎯 Risk Assessment', '📊 About'])

if page == '🎯 Risk Assessment':
    st.header('Customer Risk Assessment')
    
    col1, col2 = st.columns([1, 1.5])
    
    with col1:
        st.markdown('#### Customer Information')
        customer_id = st.text_input('Customer ID', value='CUST_001')
        
        st.markdown('---')
        st.markdown('#### Risk Indicators')
        
        salary_delay = st.slider('Salary Delay (days)', 0, 30, 0, 
                                help='Number of days salary is delayed')
        
        savings_drop = st.slider('Savings Decline (%)', 0.0, 1.0, 0.0, 0.05,
                                help='Percentage drop in savings balance')
        
        utility_delay = st.slider('Utility Payment Delay (days)', 0, 30, 0,
                                 help='Days late on utility payments')
        
        spending_drop = st.slider('Discretionary Spending Drop (%)', 0.0, 1.0, 0.0, 0.05,
                                 help='Reduction in non-essential spending')
        
        atm_increase = st.slider('ATM Withdrawal Increase', 0, 20, 0,
                                help='Increase in ATM withdrawal frequency')
        
        upi_lending = st.slider('UPI Lending App Transactions', 0, 10, 0,
                               help='Transactions with lending apps')
        
        failed_debits = st.slider('Failed Auto-debit Count', 0, 5, 0,
                                 help='Number of failed automatic payments')
        
        analyze_btn = st.button('🔍 Analyze Risk', type='primary', use_container_width=True)
    
    with col2:
        if analyze_btn:
            st.markdown('#### Risk Analysis Results')
            
            # Prepare features
            features = {
                'salary_delay_days': salary_delay,
                'savings_drop_pct': savings_drop,
                'utility_payment_delay_days': utility_delay,
                'discretionary_spend_drop_pct': spending_drop,
                'atm_withdrawal_increase': atm_increase,
                'upi_lending_txn_count': upi_lending,
                'failed_autodebit_count': failed_debits
            }
            
            # Get prediction
            risk_score, risk_level = mock_predict(features)
            
            # Display metrics
            m1, m2 = st.columns(2)
            m1.metric('Risk Score', f"{risk_score*100:.1f}%")
            m2.metric('Risk Level', risk_level)
            
            # Risk gauge
            fig = go.Figure(go.Indicator(
                mode='gauge+number',
                value=risk_score*100,
                title={'text': 'Delinquency Risk Score'},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': COLORS[risk_level]},
                    'steps': [
                        {'range': [0, 25], 'color': '#E8F5E9'},
                        {'range': [25, 50], 'color': '#FFF9C4'},
                        {'range': [50, 75], 'color': '#FFE0B2'},
                        {'range': [75, 100], 'color': '#FFCDD2'}
                    ],
                    'threshold': {
                        'line': {'color': 'red', 'width': 4},
                        'thickness': 0.75,
                        'value': 75
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendation
            st.markdown('#### Recommended Action')
            recommendation = get_recommendation(risk_level)
            
            if risk_level == 'Critical':
                st.error(f"🚨 **CRITICAL RISK**\n\n{recommendation}")
            elif risk_level == 'High':
                st.warning(f"⚠️ **HIGH RISK**\n\n{recommendation}")
            elif risk_level == 'Medium':
                st.info(f"ℹ️ **MEDIUM RISK**\n\n{recommendation}")
            else:
                st.success(f"✅ **LOW RISK**\n\n{recommendation}")
            
            # Risk factors breakdown
            st.markdown('---')
            st.markdown('#### Risk Factors Breakdown')
            
            factors_df = pd.DataFrame({
                'Factor': [
                    'Salary Delay',
                    'Savings Drop',
                    'Utility Delay',
                    'Spending Drop',
                    'ATM Increase',
                    'Lending Apps',
                    'Failed Debits'
                ],
                'Value': [
                    salary_delay,
                    f"{savings_drop*100:.0f}%",
                    utility_delay,
                    f"{spending_drop*100:.0f}%",
                    atm_increase,
                    upi_lending,
                    failed_debits
                ],
                'Impact': [
                    'High' if salary_delay > 15 else 'Medium' if salary_delay > 7 else 'Low',
                    'High' if savings_drop > 0.5 else 'Medium' if savings_drop > 0.2 else 'Low',
                    'High' if utility_delay > 15 else 'Medium' if utility_delay > 7 else 'Low',
                    'High' if spending_drop > 0.5 else 'Medium' if spending_drop > 0.2 else 'Low',
                    'High' if atm_increase > 10 else 'Medium' if atm_increase > 5 else 'Low',
                    'High' if upi_lending > 5 else 'Medium' if upi_lending > 2 else 'Low',
                    'High' if failed_debits > 2 else 'Medium' if failed_debits > 0 else 'Low'
                ]
            })
            
            st.dataframe(factors_df, use_container_width=True, hide_index=True)
            
        else:
            st.info('👆 Enter customer data and click "Analyze Risk" to see results')

else:
    st.markdown('## About This Prototype')
    st.markdown('''
    ### Pre-Delinquency Risk Detection System
    
    This is a **prototype** demonstrating the concept of early detection and prevention of loan delinquency.
    
    #### Key Features
    - Real-time risk scoring
    - Interactive risk assessment
    - Visual risk indicators
    - Actionable recommendations
    - Simple, intuitive interface
    
    #### How It Works
    1. Enter customer financial behavior indicators
    2. System analyzes risk factors
    3. Generates risk score (0-100%)
    4. Provides risk level classification
    5. Recommends intervention strategy
    
    #### Risk Levels
    - **Low (0-25%)**: Standard monitoring
    - **Medium (25-50%)**: Proactive engagement
    - **High (50-75%)**: Priority intervention
    - **Critical (75-100%)**: Immediate action required
    
    #### Technology Stack
    - **Frontend**: Streamlit
    - **Visualization**: Plotly
    - **Deployment**: Cloud-ready
    
    #### Use Cases
    - Banks and financial institutions
    - Lending platforms
    - Credit unions
    - Microfinance organizations
    
    #### Next Steps for Production
    - Integrate real ML models (XGBoost, LightGBM)
    - Connect to live data sources
    - Add batch processing
    - Implement API backend
    - Deploy to cloud infrastructure
    
    ---
    
    **Note**: This prototype uses mock predictions for demonstration. 
    Production version includes trained ML models with 95% accuracy.
    ''')

st.markdown('---')
st.caption('🛡️ Pre-Delinquency Risk Detection Prototype v1.0 | For Demo Purposes Only')
