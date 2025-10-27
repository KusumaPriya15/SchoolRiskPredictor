import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from data_preprocessing import DataPreprocessor
from model_training import DropoutRiskModel
from forecasting import DropoutForecaster, generate_national_forecast
import folium
from streamlit_folium import st_folium
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="AI-Driven Early Warning System - School Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎓 AI-Driven Early Warning System for School Dropout Prediction")
st.markdown("**UDISE+ India School Education Dataset | Predicting Dropout Risk: High, Medium, Low**")

@st.cache_data
def load_data(filepath='data/udise_data.csv'):
    """Load the UDISE+ dataset"""
    df = pd.read_csv(filepath)
    return df

@st.cache_resource
def load_model_and_preprocessor():
    """Load trained model and preprocessor"""
    try:
        if os.path.exists('models/dropout_model_best.pkl'):
            model_data = joblib.load('models/dropout_model_best.pkl')
            model = DropoutRiskModel()
            model.model = model_data['model']
            model.model_type = model_data['model_type']
            model.feature_names = model_data['feature_names']
            model.class_names = model_data['class_names']
        else:
            model = None
        
        if os.path.exists('models/preprocessor.pkl'):
            preprocessor = DataPreprocessor()
            preprocessor.load_preprocessor('models/preprocessor.pkl')
        else:
            preprocessor = None
        
        return model, preprocessor
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

def get_predictions(df, model, preprocessor):
    """Generate predictions for the dataset"""
    df_pred = df.copy()
    
    df_pred = preprocessor.engineer_features(df_pred)
    df_pred, feature_cols = preprocessor.prepare_features(df_pred, fit=False)
    
    X = df_pred[feature_cols]
    predictions, probabilities = model.predict(X)
    
    df['predicted_risk'] = predictions
    df['predicted_risk_label'] = df['predicted_risk'].map({0: 'Low', 1: 'Medium', 2: 'High'})
    df['risk_probability_low'] = probabilities[:, 0]
    df['risk_probability_medium'] = probabilities[:, 1]
    df['risk_probability_high'] = probabilities[:, 2]
    
    return df

def process_uploaded_data(uploaded_file, model, preprocessor):
    """Process uploaded CSV/Excel file"""
    try:
        if uploaded_file.name.endswith('.csv'):
            new_df = pd.read_csv(uploaded_file)
        else:
            new_df = pd.read_excel(uploaded_file)
        
        required_cols = ['school_id', 'year', 'state', 'district', 'total_enrollment', 
                        'num_teachers', 'pupil_teacher_ratio']
        
        missing_cols = [col for col in required_cols if col not in new_df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            return None
        
        new_df = get_predictions(new_df, model, preprocessor)
        
        return new_df
        
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

st.sidebar.markdown("### 📂 Data Upload")
uploaded_file = st.sidebar.file_uploader(
    "Upload Latest UDISE+ Data (CSV/Excel)",
    type=['csv', 'xlsx', 'xls'],
    help="Upload new yearly data to update predictions"
)

sidebar_option = st.sidebar.selectbox(
    "Select Dashboard Section",
    ["Overview", "School-wise Risk Scores", "Geographic Visualization", 
     "Temporal Trends", "Dropout Risk Forecast", "Model Performance", "Feature Importance"]
)

model, preprocessor = load_model_and_preprocessor()

if model is None or preprocessor is None:
    st.warning("⚠️ Models not trained yet. Please run model training first.")
    st.info("Run the following command to train models: `python model_training.py`")
    
    if st.button("Train Models Now"):
        with st.spinner("Training models... This may take a few minutes."):
            import subprocess
            result = subprocess.run(['python', 'model_training.py'], capture_output=True, text=True)
            if result.returncode == 0:
                st.success("Models trained successfully! Please refresh the page.")
                st.rerun()
            else:
                st.error("Error training models. Check console for details.")
    st.stop()

if uploaded_file is not None:
    st.sidebar.success(f"✅ File uploaded: {uploaded_file.name}")
    with st.spinner("Processing uploaded data..."):
        df = process_uploaded_data(uploaded_file, model, preprocessor)
        if df is None:
            st.stop()
        st.sidebar.info(f"Loaded {len(df)} records from uploaded file")
else:
    df = load_data()

if 'predicted_risk_label' not in df.columns:
    df = get_predictions(df, model, preprocessor)

latest_year = df['year'].max()
df_latest = df[df['year'] == latest_year].copy()

if sidebar_option == "Overview":
    st.header("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Schools", f"{df['school_id'].nunique():,}")
    with col2:
        high_risk_count = len(df_latest[df_latest['predicted_risk_label'] == 'High'])
        st.metric("High Risk Schools", f"{high_risk_count:,}", delta=f"{high_risk_count/len(df_latest)*100:.1f}%")
    with col3:
        st.metric("States Covered", f"{df['state'].nunique()}")
    with col4:
        st.metric("Latest Year", latest_year)
    
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Dropout Risk Distribution (Latest Year)")
        risk_dist = df_latest['predicted_risk_label'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
        
        colors = {'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'}
        fig = px.pie(
            values=risk_dist.values,
            names=risk_dist.index,
            color=risk_dist.index,
            color_discrete_map=colors,
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Risk Distribution by State (Top 10)")
        state_risk = df_latest.groupby('state')['predicted_risk_label'].apply(
            lambda x: (x == 'High').sum()
        ).sort_values(ascending=False).head(10)
        
        fig = px.bar(
            x=state_risk.values,
            y=state_risk.index,
            orientation='h',
            labels={'x': 'Number of High-Risk Schools', 'y': 'State'},
            color=state_risk.values,
            color_continuous_scale='Reds'
        )
        fig.update_layout(height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Key Insights")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_ptr = df_latest[df_latest['predicted_risk_label'] == 'High']['pupil_teacher_ratio'].mean()
        st.info(f"**Average PTR in High-Risk Schools:** {avg_ptr:.1f}")
    
    with col2:
        infra_score = df_latest[df_latest['predicted_risk_label'] == 'High']['infrastructure_score'].mean()
        st.info(f"**Avg Infrastructure Score (High-Risk):** {infra_score:.2f}")
    
    with col3:
        gender_parity = df_latest[df_latest['predicted_risk_label'] == 'High']['gender_parity_index'].mean()
        st.info(f"**Avg Gender Parity Index (High-Risk):** {gender_parity:.2f}")

elif sidebar_option == "School-wise Risk Scores":
    st.header("🏫 School-wise Risk Scores")
    
    st.sidebar.subheader("Filters")
    selected_states = st.sidebar.multiselect(
        "Select States",
        options=sorted(df_latest['state'].unique()),
        default=[]
    )
    
    selected_risk = st.sidebar.multiselect(
        "Select Risk Levels",
        options=['Low', 'Medium', 'High'],
        default=['High', 'Medium', 'Low']
    )
    
    filtered_df = df_latest.copy()
    if selected_states:
        filtered_df = filtered_df[filtered_df['state'].isin(selected_states)]
    if selected_risk:
        filtered_df = filtered_df[filtered_df['predicted_risk_label'].isin(selected_risk)]
    
    st.subheader(f"Displaying {len(filtered_df):,} schools")
    
    display_cols = [
        'school_id', 'state', 'district', 'school_type', 'location',
        'total_enrollment', 'pupil_teacher_ratio', 'infrastructure_score',
        'gender_parity_index', 'predicted_risk_label', 'risk_probability_high'
    ]
    
    display_df = filtered_df[display_cols].copy()
    display_df = display_df.sort_values('risk_probability_high', ascending=False)
    
    def color_risk(val):
        if val == 'High':
            return 'background-color: #dc3545; color: white'
        elif val == 'Medium':
            return 'background-color: #ffc107; color: black'
        else:
            return 'background-color: #28a745; color: white'
    
    styled_df = display_df.style.map(
        color_risk,
        subset=['predicted_risk_label']
    )
    
    st.dataframe(styled_df, height=500, use_container_width=True)
    
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Risk Scores (CSV)",
        data=csv,
        file_name=f"school_risk_scores_{latest_year}.csv",
        mime="text/csv"
    )

elif sidebar_option == "Geographic Visualization":
    st.header("🗺️ Geographic Visualization - Dropout Risk Hotspots")
    
    st.subheader("Risk Distribution by State (All Categories)")
    
    state_risk_all = df_latest.groupby(['state', 'predicted_risk_label']).size().reset_index(name='count')
    state_totals = df_latest.groupby('state').size().reset_index(name='total')
    state_risk_all = state_risk_all.merge(state_totals, on='state')
    state_risk_all['percentage'] = (state_risk_all['count'] / state_risk_all['total'] * 100).round(1)
    
    fig = px.bar(
        state_risk_all,
        x='state',
        y='percentage',
        color='predicted_risk_label',
        color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
        labels={'percentage': 'Percentage of Schools (%)', 'state': 'State', 'predicted_risk_label': 'Risk Category'},
        barmode='stack',
        title='Stacked Risk Distribution by State (All Categories)'
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("High Risk Percentage by State")
    
    state_risk_summary = df_latest.groupby('state').agg({
        'predicted_risk_label': lambda x: (x == 'High').sum() / len(x) * 100,
        'school_id': 'count',
        'pupil_teacher_ratio': 'mean',
        'infrastructure_score': 'mean'
    }).reset_index()
    
    state_risk_summary.columns = ['state', 'high_risk_percentage', 'total_schools', 'avg_ptr', 'avg_infra_score']
    
    state_risk_summary['risk_category'] = pd.cut(
        state_risk_summary['high_risk_percentage'],
        bins=[0, 30, 50, 100],
        labels=['Low', 'Medium', 'High']
    )
    
    fig = px.bar(
        state_risk_summary.sort_values('high_risk_percentage', ascending=False),
        x='state',
        y='high_risk_percentage',
        color='risk_category',
        color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
        labels={'high_risk_percentage': 'High Risk Schools (%)', 'state': 'State'},
        hover_data=['total_schools', 'avg_ptr', 'avg_infra_score']
    )
    fig.update_layout(height=500, xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Interactive State Heatmap")
    
    state_coords = {
        'Uttar Pradesh': [27.0, 80.0], 'Maharashtra': [19.0, 76.0], 'Bihar': [25.5, 85.0],
        'West Bengal': [23.0, 87.5], 'Madhya Pradesh': [23.0, 77.0], 'Tamil Nadu': [11.0, 78.0],
        'Rajasthan': [27.0, 74.0], 'Karnataka': [15.0, 76.0], 'Gujarat': [22.0, 72.0],
        'Andhra Pradesh': [16.0, 80.0], 'Odisha': [20.5, 84.0], 'Telangana': [18.0, 79.0],
        'Kerala': [10.5, 76.0], 'Jharkhand': [23.5, 85.5], 'Assam': [26.0, 92.5],
        'Punjab': [31.0, 75.5], 'Chhattisgarh': [21.0, 82.0], 'Haryana': [29.0, 76.0],
        'Delhi': [28.7, 77.2], 'Jammu and Kashmir': [34.0, 75.0], 'Uttarakhand': [30.0, 79.0],
        'Himachal Pradesh': [32.0, 77.0], 'Tripura': [23.8, 91.3], 'Meghalaya': [25.5, 91.5],
        'Manipur': [24.8, 93.9], 'Nagaland': [26.2, 94.5], 'Goa': [15.3, 74.0],
        'Arunachal Pradesh': [28.2, 94.7], 'Mizoram': [23.7, 92.7], 'Sikkim': [27.6, 88.5]
    }
    
    m = folium.Map(location=[22.0, 78.0], zoom_start=5, tiles='OpenStreetMap')
    
    for _, row in state_risk_summary.iterrows():
        state = row['state']
        if state in state_coords:
            coords = state_coords[state]
            risk_pct = row['high_risk_percentage']
            category = row['risk_category']
            
            if category == 'High':
                color = 'red'
            elif category == 'Medium':
                color = 'orange'
            else:
                color = 'green'
            
            folium.CircleMarker(
                location=coords,
                radius=max(5, risk_pct / 3),
                popup=f"""
                <b>{state}</b><br>
                Risk Category: <b>{category}</b><br>
                High Risk: {risk_pct:.1f}%<br>
                Total Schools: {int(row['total_schools'])}<br>
                Avg PTR: {row['avg_ptr']:.1f}<br>
                Avg Infrastructure: {row['avg_infra_score']:.2f}
                """,
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.7,
                weight=2
            ).add_to(m)
    
    st_folium(m, width=1200, height=600)
    
    st.info("**Map Legend:** 🔴 High Risk (>50%) | 🟠 Medium Risk (30-50%) | 🟢 Low Risk (<30%)")

elif sidebar_option == "Temporal Trends":
    st.header("📈 Temporal Trends - Dropout Risk Over Time")
    
    st.subheader("Risk Trend Across Years")
    
    risk_col = 'predicted_risk_label' if 'predicted_risk_label' in df.columns else 'dropout_risk'
    
    yearly_risk = df.groupby(['year', risk_col]).size().reset_index(name='count')
    yearly_risk['percentage'] = yearly_risk.groupby('year')['count'].transform(lambda x: 100 * x / x.sum())
    yearly_risk_pct = yearly_risk[['year', risk_col, 'percentage']]
    
    fig = px.line(
        yearly_risk_pct,
        x='year',
        y='percentage',
        color=risk_col,
        color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
        markers=True,
        labels={'percentage': 'Percentage of Schools (%)', 'year': 'Year', risk_col: 'Risk Level'}
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("State-wise Risk Evolution Heatmap")
    
    top_states = df_latest['state'].value_counts().head(15).index
    df_top_states = df[df['state'].isin(top_states)]
    
    heatmap_data = df_top_states.pivot_table(
        values='school_id',
        index='state',
        columns='year',
        aggfunc=lambda x: (df_top_states.loc[x.index, risk_col] == 'High').sum() / len(x) * 100
    )
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Year", y="State", color="High Risk %"),
        color_continuous_scale='Reds',
        aspect='auto'
    )
    fig.update_layout(height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Infrastructure Trends")
    
    infra_trend = df.groupby('year').agg({
        'infrastructure_score': 'mean',
        'pupil_teacher_ratio': 'mean',
        'gender_parity_index': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=1, cols=3,
        subplot_titles=('Avg Infrastructure Score', 'Avg Pupil-Teacher Ratio', 'Avg Gender Parity Index')
    )
    
    fig.add_trace(
        go.Scatter(x=infra_trend['year'], y=infra_trend['infrastructure_score'], 
                  mode='lines+markers', name='Infrastructure', line=dict(color='#007bff')),
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=infra_trend['year'], y=infra_trend['pupil_teacher_ratio'],
                  mode='lines+markers', name='PTR', line=dict(color='#dc3545')),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(x=infra_trend['year'], y=infra_trend['gender_parity_index'],
                  mode='lines+markers', name='Gender Parity', line=dict(color='#28a745')),
        row=1, col=3
    )
    
    fig.update_layout(height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

elif sidebar_option == "Dropout Risk Forecast":
    st.header("🔮 Dropout Risk Forecast (Next 2-3 Years)")
    
    st.markdown("""
    This section uses **Prophet time-series forecasting** to predict future dropout risk trends
    based on historical patterns in the data.
    """)
    
    forecast_periods = st.slider("Forecast Period (years)", min_value=1, max_value=5, value=3)
    
    with st.spinner("Generating forecasts..."):
        forecaster = DropoutForecaster()
        
        st.subheader("1. National-Level Forecast")
        
        national_forecast = generate_national_forecast(df, periods=forecast_periods)
        
        if national_forecast is not None:
            future_preds = forecaster.get_future_predictions(national_forecast, forecast_periods)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = go.Figure()
                
                historical = national_forecast[national_forecast['ds'] <= pd.Timestamp.now()]
                future = national_forecast[national_forecast['ds'] > pd.Timestamp.now()].head(forecast_periods)
                
                fig.add_trace(go.Scatter(
                    x=historical['ds'],
                    y=historical['y'],
                    mode='markers',
                    name='Historical Data',
                    marker=dict(color='blue', size=8)
                ))
                
                fig.add_trace(go.Scatter(
                    x=future['ds'],
                    y=future['yhat'],
                    mode='lines+markers',
                    name='Forecast',
                    line=dict(color='red', width=3),
                    marker=dict(size=10)
                ))
                
                fig.add_trace(go.Scatter(
                    x=future['ds'].tolist() + future['ds'].tolist()[::-1],
                    y=future['yhat_upper'].tolist() + future['yhat_lower'].tolist()[::-1],
                    fill='toself',
                    fillcolor='rgba(255,0,0,0.2)',
                    line=dict(color='rgba(255,255,255,0)'),
                    name='80% Confidence Interval'
                ))
                
                fig.update_layout(
                    title='National Dropout Risk Forecast',
                    xaxis_title='Year',
                    yaxis_title='High Risk Schools (%)',
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("**Forecast Summary**")
                for _, row in future_preds.iterrows():
                    risk_cat = forecaster.classify_forecasted_risk(row['yhat'])
                    color = '#dc3545' if risk_cat == 'High' else ('#ffc107' if risk_cat == 'Medium' else '#28a745')
                    st.markdown(f"""
                    **Year {int(row['year'])}**  
                    Predicted: {row['yhat']:.1f}%  
                    Range: {row['yhat_lower']:.1f}% - {row['yhat_upper']:.1f}%  
                    <span style='background-color:{color}; color:white; padding:2px 8px; border-radius:3px;'>{risk_cat} Risk</span>
                    """, unsafe_allow_html=True)
                    st.markdown("---")
        
        st.subheader("2. State-Level Forecast Summary")
        
        state_forecast_summary = forecaster.generate_state_level_forecast_summary(df, periods=forecast_periods)
        
        if not state_forecast_summary.empty:
            selected_year_forecast = st.selectbox(
                "Select Forecast Year",
                sorted(state_forecast_summary['year'].unique())
            )
            
            year_data = state_forecast_summary[state_forecast_summary['year'] == selected_year_forecast]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.bar(
                    year_data.sort_values('predicted_high_risk_pct', ascending=False).head(15),
                    x='state',
                    y='predicted_high_risk_pct',
                    color='risk_category',
                    color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
                    labels={'predicted_high_risk_pct': 'Predicted High Risk (%)', 'state': 'State'},
                    title=f'Top 15 States by Predicted High Risk - Year {selected_year_forecast}'
                )
                fig.update_layout(height=500, xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                risk_dist = year_data['risk_category'].value_counts().reindex(['Low', 'Medium', 'High'], fill_value=0)
                fig = px.pie(
                    values=risk_dist.values,
                    names=risk_dist.index,
                    color=risk_dist.index,
                    color_discrete_map={'Low': '#28a745', 'Medium': '#ffc107', 'High': '#dc3545'},
                    title=f'Risk Distribution - {selected_year_forecast}'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            st.subheader("3. Future Risk Map")
            
            state_coords = {
                'Uttar Pradesh': [27.0, 80.0], 'Maharashtra': [19.0, 76.0], 'Bihar': [25.5, 85.0],
                'West Bengal': [23.0, 87.5], 'Madhya Pradesh': [23.0, 77.0], 'Tamil Nadu': [11.0, 78.0],
                'Rajasthan': [27.0, 74.0], 'Karnataka': [15.0, 76.0], 'Gujarat': [22.0, 72.0],
                'Andhra Pradesh': [16.0, 80.0], 'Odisha': [20.5, 84.0], 'Telangana': [18.0, 79.0],
                'Kerala': [10.5, 76.0], 'Jharkhand': [23.5, 85.5], 'Assam': [26.0, 92.5],
                'Punjab': [31.0, 75.5], 'Chhattisgarh': [21.0, 82.0], 'Haryana': [29.0, 76.0],
                'Delhi': [28.7, 77.2], 'Jammu and Kashmir': [34.0, 75.0], 'Uttarakhand': [30.0, 79.0],
                'Himachal Pradesh': [32.0, 77.0], 'Tripura': [23.8, 91.3], 'Meghalaya': [25.5, 91.5],
                'Manipur': [24.8, 93.9], 'Nagaland': [26.2, 94.5], 'Goa': [15.3, 74.0],
                'Arunachal Pradesh': [28.2, 94.7], 'Mizoram': [23.7, 92.7], 'Sikkim': [27.6, 88.5]
            }
            
            m = folium.Map(location=[22.0, 78.0], zoom_start=5, tiles='OpenStreetMap')
            
            for _, row in year_data.iterrows():
                state = row['state']
                if state in state_coords:
                    coords = state_coords[state]
                    risk_pct = row['predicted_high_risk_pct']
                    category = row['risk_category']
                    
                    if category == 'High':
                        color = 'red'
                    elif category == 'Medium':
                        color = 'orange'
                    else:
                        color = 'green'
                    
                    folium.CircleMarker(
                        location=coords,
                        radius=max(5, risk_pct / 3),
                        popup=f"""
                        <b>{state}</b><br>
                        <b>Forecast Year: {selected_year_forecast}</b><br>
                        Risk Category: <b>{category}</b><br>
                        Predicted High Risk: {risk_pct:.1f}%<br>
                        Confidence Range: {row['lower_bound']:.1f}% - {row['upper_bound']:.1f}%
                        """,
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7,
                        weight=2
                    ).add_to(m)
            
            st_folium(m, width=1200, height=600)
            
            st.info(f"**Predicted Risk Map for Year {selected_year_forecast}**  \n🔴 High Risk (>50%) | 🟠 Medium Risk (30-50%) | 🟢 Low Risk (<30%)")
            
            csv = state_forecast_summary.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast Summary (CSV)",
                data=csv,
                file_name=f"dropout_forecast_{forecast_periods}years.csv",
                mime="text/csv"
            )

elif sidebar_option == "Model Performance":
    st.header("🎯 Model Performance Metrics")
    
    if os.path.exists('models/model_comparison.csv'):
        comparison_df = pd.read_csv('models/model_comparison.csv', index_col=0)
        
        st.subheader("Model Comparison")
        st.dataframe(comparison_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                comparison_df.reset_index(),
                x='index',
                y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                barmode='group',
                labels={'index': 'Model', 'value': 'Score'},
                title='Performance Metrics Comparison'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            best_model_name = comparison_df['F1-Score'].idxmax()
            st.success(f"**Best Performing Model:** {best_model_name.upper()}")
            st.metric("Best F1-Score", f"{comparison_df.loc[best_model_name, 'F1-Score']:.4f}")
            st.metric("Accuracy", f"{comparison_df.loc[best_model_name, 'Accuracy']:.4f}")
            st.metric("Precision", f"{comparison_df.loc[best_model_name, 'Precision']:.4f}")
            st.metric("Recall", f"{comparison_df.loc[best_model_name, 'Recall']:.4f}")
    
    st.subheader("Confusion Matrix")
    
    model_types = ['xgboost', 'randomforest', 'lightgbm']
    available_models = [m for m in model_types if os.path.exists(f'models/confusion_matrix_{m}.png')]
    
    if available_models:
        cols = st.columns(len(available_models))
        for idx, model_type in enumerate(available_models):
            with cols[idx]:
                st.image(f'models/confusion_matrix_{model_type}.png', 
                        caption=f'{model_type.upper()} Model',
                        use_container_width=True)
    else:
        st.warning("Confusion matrices not found. Please train the models first.")

elif sidebar_option == "Feature Importance":
    st.header("🔍 Feature Importance & Model Explainability")
    
    st.markdown("""
    **Understanding Dropout Risk Drivers**
    
    Feature importance shows which factors have the strongest influence on predicting dropout risk.
    This helps policymakers identify key intervention areas.
    """)
    
    model_types = ['xgboost', 'randomforest', 'lightgbm']
    available_models = [m for m in model_types if os.path.exists(f'models/feature_importance_{m}.png')]
    
    if available_models:
        selected_model = st.selectbox("Select Model", available_models)
        
        st.image(f'models/feature_importance_{selected_model}.png',
                caption=f'Top Features - {selected_model.upper()} Model',
                use_container_width=True)
        
        st.subheader("Feature Importance Table")
        
        loaded_model = DropoutRiskModel(model_type=selected_model)
        loaded_model.load_model(f'models/dropout_model_{selected_model}.pkl')
        
        importance_df = loaded_model.get_feature_importance(top_n=20)
        st.dataframe(importance_df, use_container_width=True)
        
        st.subheader("Key Insights")
        
        top_features = importance_df.head(5)['feature'].tolist()
        
        st.info(f"""
        **Top 5 Most Important Features:**
        1. {top_features[0] if len(top_features) > 0 else 'N/A'}
        2. {top_features[1] if len(top_features) > 1 else 'N/A'}
        3. {top_features[2] if len(top_features) > 2 else 'N/A'}
        4. {top_features[3] if len(top_features) > 3 else 'N/A'}
        5. {top_features[4] if len(top_features) > 4 else 'N/A'}
        """)
        
        st.markdown("""
        ### Actionable Recommendations
        
        Based on feature importance analysis:
        
        - **Pupil-Teacher Ratio**: Ensure adequate teacher recruitment in high-risk schools
        - **Infrastructure**: Prioritize basic facilities (toilets, drinking water, electricity)
        - **Gender Parity**: Focus on programs that encourage girls' education
        - **Enrollment Trends**: Monitor declining enrollment as an early warning signal
        - **Location**: Implement targeted interventions for rural schools
        """)
    else:
        st.warning("Feature importance plots not found. Please train the models first.")

st.sidebar.markdown("---")
st.sidebar.info("""
**About this System**

This AI-driven Early Warning System uses machine learning to predict school dropout risks across India using UDISE+ data.

**Models**: XGBoost, Random Forest

**Forecasting**: Prophet time-series model

**Features**: Infrastructure, enrollment, pupil-teacher ratio, gender parity, and more.

**Purpose**: Help policymakers identify at-risk schools and take preventive action.
""")
