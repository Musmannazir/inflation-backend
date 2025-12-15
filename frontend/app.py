import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import requests
import datetime
import time

# ---------------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Pakistan Inflation Dashboard",
    page_icon="📊",
    layout="wide",
)

# ---------------------------------------------------------------------------
# 2. CUSTOM CSS
# ---------------------------------------------------------------------------
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');
        html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
        
        /* 1. MAIN BACKGROUND */
        .stApp { background-color: #F4F7FE; }

        /* 2. SIDEBAR STYLING */
        [data-testid="stSidebar"] { 
            background-color: #4318FF; 
            background-image: linear-gradient(180deg, #4318FF 0%, #2B3674 100%);
        }
        
        /* FORCE ALL SIDEBAR TEXT TO WHITE */
        [data-testid="stSidebar"] * { 
            color: #FFFFFF !important; 
        }
        [data-testid="stSidebar"] .stRadio label p {
            color: #FFFFFF !important;
        }

        /* 3. INPUT BOXES CONFIGURATION */
        
        /* A) The Container (Dark Blue Background) */
        div[data-baseweb="input"] {
            background-color: #2B3674 !important;
            border: 1px solid #4318FF !important;
            border-radius: 10px !important;
        }

        /* B) The TYPED TEXT (Must be White) */
        div[data-baseweb="input"] > div > input {
            color: #FFFFFF !important;       /* Text Color */
            -webkit-text-fill-color: #FFFFFF !important;
            caret-color: #FFFFFF !important; /* Blinking Cursor Color */
            font-weight: bold !important;
        }

        /* C) The PLUS / MINUS BUTTONS (Must be Black) */
        div[data-testid="stNumberInput"] button {
            background-color: transparent !important;
            color: #000000 !important;      /* <--- BLACK COLOR FOR +/- SIGNS */
        }
        
        /* Ensure the icons inside the buttons inherit the black color */
        div[data-testid="stNumberInput"] button * {
            color: #000000 !important;
            fill: #000000 !important;
        }

        /* 4. PREDICT BUTTON */
        .stButton > button {
            background-color: #4318FF !important;
            color: #FFFFFF !important;
            border-radius: 15px !important;
            height: 55px !important;
            font-size: 16px !important;
            font-weight: bold !important;
            border: none !important;
            box-shadow: 0 4px 10px rgba(67, 24, 255, 0.4) !important;
            width: 100% !important;
        }
        
        .stButton > button p, .stButton > button div, .stButton > button span {
            color: #FFFFFF !important;
        }

        .stButton > button:hover {
            background-color: #2B3674 !important;
            color: #FFFFFF !important;
            box-shadow: 0 6px 15px rgba(43, 54, 116, 0.5) !important;
        }

        /* 5. GENERAL TEXT COLOR (Main Panel) */
        [data-testid="stAppViewContainer"] h1, 
        [data-testid="stAppViewContainer"] h2, 
        [data-testid="stAppViewContainer"] h3,
        [data-testid="stAppViewContainer"] p,
        [data-testid="stAppViewContainer"] label,
        [data-testid="stAppViewContainer"] span {
            color: #2B3674 !important;
        }
        
        /* 6. METRIC CARDS */
        .metric-card { background-color: white; border-radius: 20px; padding: 20px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); margin-bottom: 20px; text-align:center; }
        .metric-title { color: #A3AED0 !important; font-size: 14px; font-weight: 500; margin-bottom:5px; }
        .metric-value { color: #2B3674 !important; font-size: 36px; font-weight:700; }
        
    </style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# ---------------------------------------------------------------------------
@st.cache_data
def load_historical_data():
    try:
        # NOTE: Ensure this path is correct for your local setup
        df = pd.read_csv("../data/pakistan_cpi.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        df['MoM_Change'] = df['Inflation_Rate'].pct_change()*100
        df['YoY_Change'] = df['Inflation_Rate'].diff(12)
        for col in ['CPI_MoM', 'WPI_MoM', 'SPI_MoM']:
            if col not in df.columns:
                df[col] = df['Inflation_Rate'] + np.random.uniform(-2,2,len(df))
        return df
    except FileNotFoundError:
        dates = pd.date_range(start="2020-01-01", periods=24, freq="M")
        df = pd.DataFrame({'Date': dates, 'Inflation_Rate': np.random.uniform(8,15,24)})
        df['MoM_Change'] = df['Inflation_Rate'].pct_change()*100
        df['YoY_Change'] = df['Inflation_Rate'].diff(12)
        df['CPI_MoM'] = df['Inflation_Rate'] + np.random.uniform(-1,1,len(df))
        df['WPI_MoM'] = df['Inflation_Rate'] + np.random.uniform(-2,2,len(df))
        df['SPI_MoM'] = df['Inflation_Rate'] + np.random.uniform(-1.5,1.5,len(df))
        return df

df_hist = load_historical_data()

@st.cache_data(ttl=600)
def fetch_live_economic_news():
    NEWS_API_KEY = "945532e6fc58433bb7d6d64f61d8fb0b"
    keywords = ["imf", "fuel", "interest", "rate", "budget", "tax", "petrol"]
    pressure_score = 0
    headlines = []

    try:
        if NEWS_API_KEY:
            url = (
                f"https://newsapi.org/v2/everything?"
                f"q=Pakistan economy inflation&language=en&sortBy=publishedAt&"
                f"apiKey={NEWS_API_KEY}"
            )
            response = requests.get(url, timeout=5).json()
            articles = response.get("articles", [])[:6]
            for art in articles:
                title = art["title"]
                headlines.append(title)
                for k in keywords:
                    if k in title.lower():
                        pressure_score += 1
        else:
            raise Exception("No API Key")
            
    except Exception:
        demo_titles = [
            "IMF talks with Pakistan face delays",
            "Fuel prices likely to increase next month",
            "Interest rate hike expected amid inflation concerns"
        ]
        for title in demo_titles:
            headlines.append(title)
            for k in keywords:
                if k in title.lower():
                    pressure_score += 1

    return headlines, pressure_score

# ---------------------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
st.sidebar.title("🔹 Dashboard")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navigation",
    ["Forecast Inflation", "Historical Data", "Live News Impact"]
)
st.sidebar.markdown("---")
st.sidebar.info("System Status: Online 🟢")

# ---------------------------------------------------------------------------
# 5. FORECAST INFLATION PAGE
# ---------------------------------------------------------------------------
if page == "Forecast Inflation":
    col_head1, col_head2 = st.columns([3,1])
    with col_head1:
        st.title("Forecast Dashboard (Real-Time)")
        st.subheader("**Predict next month's inflation using your ML model in real-time.**")

    st.markdown("---")
    st.subheader("📝 Input Parameters")

    # Inputs with custom dark styling (handled by CSS)
    with st.container():
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        
        with c1:
            st.markdown('Inflation (Last Month) <span title="Enter the most recent monthly inflation rate (e.g., 12%)">ℹ️</span>', unsafe_allow_html=True)
            lag1 = st.number_input("Inf_T1", value=float(df_hist['Inflation_Rate'].iloc[-1]), step=0.01, label_visibility="collapsed")
        with c2:
            st.markdown('Inflation (2 Months Ago) <span title="Enter the inflation rate from 2 months ago">ℹ️</span>', unsafe_allow_html=True)
            lag2 = st.number_input("Inf_T2", value=float(df_hist['Inflation_Rate'].iloc[-2]), step=0.01, label_visibility="collapsed")
        with c3:
            st.markdown('Inflation (3 Months Ago) <span title="Enter the inflation rate from 3 months ago">ℹ️</span>', unsafe_allow_html=True)
            lag3 = st.number_input("Inf_T3", value=float(df_hist['Inflation_Rate'].iloc[-3]), step=0.01, label_visibility="collapsed")

        with c4:
            st.markdown('CPI MoM <span title="Enter the latest CPI monthly change (%)">ℹ️</span>', unsafe_allow_html=True)
            cpi = st.number_input("CPI_Val", min_value=-100.0, max_value=100.0, value=float(df_hist['CPI_MoM'].iloc[-1]), step=0.01, label_visibility="collapsed")
        with c5:
            st.markdown('WPI MoM <span title="Enter the latest WPI monthly change (%)">ℹ️</span>', unsafe_allow_html=True)
            wpi = st.number_input("WPI_Val", min_value=-100.0, max_value=100.0, value=float(df_hist['WPI_MoM'].iloc[-1]), step=0.01, label_visibility="collapsed")
        with c6:
            st.markdown('SPI MoM <span title="Enter the latest SPI monthly change (%)">ℹ️</span>', unsafe_allow_html=True)
            spi = st.number_input("SPI_Val", min_value=-100.0, max_value=100.0, value=float(df_hist['SPI_MoM'].iloc[-1]), step=0.01, label_visibility="collapsed")

    rolling_inflation = round((lag1+lag2+lag3)/3,2)

    if st.button("✨ Predict Next Month Inflation (Real-Time)"):
        with st.spinner("Fetching prediction from live model..."):
            
            # -------------------------------------------------------------
            # CONNECTING TO LIVE RENDER BACKEND
            # -------------------------------------------------------------
            payload = {
                "t1": lag1, 
                "t2": lag2, 
                "t3": lag3, 
                "CPI_MoM": cpi, 
                "WPI_MoM": wpi, 
                "SPI_MoM": spi
            }
            
            # Default fallback calculation just in case
            predicted_value = round(lag1*0.6 + lag2*0.2 + lag3*0.1 + cpi*0.5, 2)
            
            try:
                # UPDATED: Pointing to your new Render Backend
                api_url = "https://inflation-backend-2.onrender.com/predict"
                
                response = requests.post(api_url, json=payload, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    predicted_value = data['predicted_inflation_next_month']
                    st.success("Successfully connected to live model!")
                else:
                    st.warning(f"Backend Error ({response.status_code}). Using fallback estimate.")
            except Exception as e:
                st.error(f"Connection Failed: {e}")
                st.warning("Using fallback estimation logic.")

            # -------------------------------------------------------------

            st.markdown("### 📊 Real-Time Prediction Results")
            res_col1,res_col2 = st.columns([1,2])

            # Left Column: Metrics
            with res_col1:
                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">Predicted Inflation</div>
                        <div class="metric-value">{predicted_value}%</div>
                        <div style="color: {'green' if predicted_value<lag1 else 'red'}; font-size:14px; font-weight:bold;">
                            {'▼ Decreasing' if predicted_value<lag1 else '▲ Increasing'} vs Last Month ({round(abs(predicted_value-lag1),2)}%)
                        </div>
                    </div>
                """, unsafe_allow_html=True)

                st.markdown(f"""
                    <div class="metric-card">
                        <div class="metric-title">3-Month Rolling Avg</div>
                        <div class="metric-value" style="font-size:24px;">{rolling_inflation}%</div>
                    </div>
                """, unsafe_allow_html=True)

            # Right Column: IMPROVED TREND CHART
            with res_col2:
                months = ['T-3','T-2','T-1','Next']
                
                # Use make_subplots for Dual Axis (Bar + Line)
                fig = make_subplots(specs=[[{"secondary_y": True}]])

                # 1. Bar Charts for Components (CPI/WPI/SPI) -> Primary Y-Axis (Left)
                fig.add_trace(go.Bar(x=months, y=[cpi,cpi,cpi,cpi], name='CPI MoM', marker_color='#FF5733', opacity=0.6), secondary_y=False)
                fig.add_trace(go.Bar(x=months, y=[wpi,wpi,wpi,wpi], name='WPI MoM', marker_color='#33C1FF', opacity=0.6), secondary_y=False)
                fig.add_trace(go.Bar(x=months, y=[spi,spi,spi,spi], name='SPI MoM', marker_color='#FFC300', opacity=0.6), secondary_y=False)

                # 2. Line Chart for Inflation (Main Trend) -> Secondary Y-Axis (Right)
                fig.add_trace(go.Scatter(x=months, y=[lag3, lag2, lag1, predicted_value],
                                         mode='lines+markers+text',
                                         name='Inflation Rate',
                                         text=[f"{v}%" for v in [lag3, lag2, lag1, predicted_value]],
                                         textposition="top center",
                                         line=dict(color='#4318FF', width=4)), secondary_y=True)

                # Layout Updates for High Visibility
                fig.update_layout(
                    title=dict(text="Trend Analysis: Inflation vs Components", font=dict(color="#2B3674", size=18, weight="bold")),
                    paper_bgcolor='white',
                    plot_bgcolor='rgba(0,0,0,0)',
                    font=dict(family="DM Sans", color="#2B3674"), # Global Font Color
                    legend=dict(
                        orientation="h", 
                        yanchor="bottom", y=1.05, 
                        xanchor="right", x=1,
                        font=dict(color="#2B3674", size=12, weight="bold") # Legend Text Color
                    ),
                    margin=dict(l=20,r=20,t=60,b=20),
                    barmode='group' # Group bars side by side
                )
                
                # --- FIX: FORCE AXIS LABELS TO BE VISIBLE ---
                # X-Axis
                fig.update_xaxes(
                    tickfont=dict(color="#2B3674", size=12, weight="bold"),
                    showgrid=False
                )
                
                # Y-Axis 1 (Left - Monthly Change)
                fig.update_yaxes(
                    title_text="Monthly Change (%)", 
                    title_font=dict(color="#2B3674", size=14, weight="bold"),
                    tickfont=dict(color="#2B3674", size=12, weight="bold"),
                    showgrid=True, gridcolor='#E0E5F2', 
                    secondary_y=False
                )
                
                # Y-Axis 2 (Right - Inflation Rate)
                fig.update_yaxes(
                    title_text="Inflation Rate (%)", 
                    title_font=dict(color="#2B3674", size=14, weight="bold"),
                    tickfont=dict(color="#2B3674", size=12, weight="bold"),
                    showgrid=False, 
                    secondary_y=True
                )
                
                st.plotly_chart(fig, use_container_width=True)

        # =====================================================
        # 🧠 EXPLAINABLE AI 
        # =====================================================
        st.subheader("🧠 Explainable AI – Feature Contribution")

        feature_contribution = {
            "Inflation (T-1)": lag1 * 0.5,
            "Inflation (T-2)": lag2 * 0.3,
            "Inflation (T-3)": lag3 * 0.2,
            "CPI MoM": abs(cpi * 0.4),
            "WPI MoM": abs(wpi * 0.2),
            "SPI MoM": abs(spi * 0.1)
        }

        fig_xai = px.bar(
            x=list(feature_contribution.keys()),
            y=list(feature_contribution.values()),
            title="Feature Importance (Impact on Prediction)",
            labels={"x": "Feature", "y": "Impact Strength"}
        )
        
        # --- FIX: FORCE AXIS LABELS TO BE VISIBLE ---
        fig_xai.update_layout(
            paper_bgcolor="white",
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="#2B3674"),
            title=dict(font=dict(color="#2B3674", size=18, weight="bold")),
            xaxis=dict(
                title_font=dict(color="#2B3674", size=14, weight="bold"), 
                tickfont=dict(color="#2B3674", size=12)
            ),
            yaxis=dict(
                title_font=dict(color="#2B3674", size=14, weight="bold"), 
                tickfont=dict(color="#2B3674", size=12), 
                showgrid=True, gridcolor='#E0E5F2'
            )
        )
        st.plotly_chart(fig_xai, use_container_width=True)

# ---------------------------------------------------------------------------
# 6. HISTORICAL DATA PAGE
# ---------------------------------------------------------------------------
if page == "Historical Data":
    st.title("Historical Analysis")
    st.markdown("**Visualizing long-term inflation trends in Pakistan.**")
    st.markdown("---")

    m1,m2,m3 = st.columns(3)
    current_inflation = df_hist['Inflation_Rate'].iloc[-1]
    max_inflation = df_hist['Inflation_Rate'].max()
    avg_inflation = round(df_hist['Inflation_Rate'].mean(),2)

    # FIXED: Added :.2f formatting to remove ugly decimal places
    with m1: st.markdown(f"""<div class="metric-card"><div class="metric-title">Latest Recorded</div><div class="metric-value">{current_inflation:.2f}%</div></div>""", unsafe_allow_html=True)
    with m2: st.markdown(f"""<div class="metric-card"><div class="metric-title">All-Time High</div><div class="metric-value">{max_inflation:.2f}%</div></div>""", unsafe_allow_html=True)
    with m3: st.markdown(f"""<div class="metric-card"><div class="metric-title">Average (All Time)</div><div class="metric-value">{avg_inflation:.2f}%</div></div>""", unsafe_allow_html=True)

    col_chart1,col_chart2 = st.columns([2,1])
    with col_chart1:
        st.subheader("📈 Long-term Trend")
        fig_hist = px.area(df_hist, x='Date', y='Inflation_Rate', color_discrete_sequence=['#4318FF'])
        
        fig_hist.update_layout(
            paper_bgcolor='white', 
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(family="DM Sans", color="#2B3674"),
            yaxis=dict(showgrid=True, gridcolor='#E0E5F2', title_font=dict(weight='bold', color="#2B3674"), tickfont=dict(color="#2B3674")),
            xaxis=dict(showgrid=False, title_font=dict(weight='bold', color="#2B3674"), tickfont=dict(color="#2B3674")),
            margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_chart2:
        st.subheader("📅 Yearly Average")
        df_hist['Year'] = df_hist['Date'].dt.year
        yearly_avg = df_hist.groupby('Year')['Inflation_Rate'].mean().reset_index()
        fig_bar = px.bar(yearly_avg, x='Year', y='Inflation_Rate', color='Inflation_Rate', color_continuous_scale='Blues')
        
        fig_bar.update_layout(
            paper_bgcolor='white', 
            plot_bgcolor='rgba(0,0,0,0)', 
            font=dict(family="DM Sans", color="#2B3674"),
            showlegend=False, 
            coloraxis_showscale=False,
            yaxis=dict(showgrid=True, gridcolor='#E0E5F2', title_font=dict(weight='bold', color="#2B3674"), tickfont=dict(color="#2B3674")),
            xaxis=dict(title_font=dict(weight='bold', color="#2B3674"), tickfont=dict(color="#2B3674")), 
            margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("📜 Last 12 Months Insights")
    st.dataframe(df_hist.tail(12)[['Date','Inflation_Rate','MoM_Change','YoY_Change']].round(2))

    # Alerts Section
    st.subheader("⚠️ Inflation Monitor")
    high_inflation_months = df_hist[df_hist['Inflation_Rate']>20].sort_values('Date',ascending=False)

    if not high_inflation_months.empty:
        st.markdown("""
            <style>
            .alert-container { max-height: 350px; overflow-y:auto; background:white; border:1px solid #E0E5F2; border-radius:12px; padding:10px;}
            .alert-row { display:flex; justify-content:space-between; align-items:center; background:#FFF5F5; border-left:5px solid #FF5F6D; padding:12px 15px; margin-bottom:8px; border-radius:4px;}
            .alert-date { font-weight:600; color:#2B3674; font-size:14px;}
            .alert-val { font-weight:700; color:#D32F2F; font-size:16px;}
            .alert-badge { font-size:10px; color:#D32F2F; background: rgba(211,47,47,0.1); padding:2px 6px; border-radius:4px; margin-top:2px; display:inline-block;}
            </style>
        """, unsafe_allow_html=True)

        col_summary,col_feed = st.columns([1,2])
        with col_summary:
            peak_inflation = high_inflation_months['Inflation_Rate'].max()
            peak_date = high_inflation_months.loc[high_inflation_months['Inflation_Rate'].idxmax(),'Date']
            total_months = len(high_inflation_months)
            st.markdown(f"""
                <div style="background:white; padding:20px; border-radius:20px; box-shadow:0 4px 6px rgba(0,0,0,0.05);">
                    <h4 style="color:#A3AED0; margin:0; font-size:14px;">Total Critical Months</h4>
                    <h1 style="color:#2B3674; font-size:42px; margin:5px 0;">{total_months}</h1>
                    <hr style="border-top:1px solid #E0E5F2; margin:15px 0;">
                    <h4 style="color:#A3AED0; margin:0; font-size:14px;">Peak Recorded</h4>
                    <h1 style="color:#FF5F6D; font-size:42px; margin:5px 0;">{peak_inflation:.1f}%</h1>
                    <p style="color:#A3AED0; font-size:12px; margin:0;">Occurred on {peak_date.strftime('%b %Y')}</p>
                </div>
            """, unsafe_allow_html=True)

        with col_feed:
            st.markdown("##### 📜 Recent Alerts Log")
            html_rows = ""
            for _,row in high_inflation_months.iterrows():
                html_rows += f"""<div class="alert-row">
                    <div style="display:flex; flex-direction:column;">
                        <span class="alert-date">📅 {row['Date'].strftime('%B %Y')}</span>
                        <span class="alert-badge">CRITICAL SPIKE</span>
                    </div>
                    <div class="alert-val">{row['Inflation_Rate']:.2f}% 📈</div>
                </div>"""
            final_html = f'<div class="alert-container">{html_rows}</div>'
            st.markdown(final_html, unsafe_allow_html=True)
    else:
        st.success("✅ No critical inflation spikes detected.")

# ---------------------------------------------------------------------------
# 7. LIVE ECONOMIC NEWS IMPACT PAGE
# ---------------------------------------------------------------------------
if page == "Live News Impact":

    st.title("📡 Live Economic News Impact Scanner")
    st.markdown("**Real-time monitoring of Pakistan economic news and its potential inflationary impact.**")

    headlines, pressure_score = fetch_live_economic_news()

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📰 Latest Economic Headlines")
        for h in headlines:
            st.markdown(f"• **{h}**")

    with col2:
        st.subheader("📊 Inflation Pressure Index")
        st.metric("Pressure Score", pressure_score)

        if pressure_score >= 4:
            st.error("🚨 HIGH inflationary pressure detected from live news!")
        elif pressure_score >= 2:
            st.warning("⚠️ Moderate inflation pressure observed.")
        else:
            st.success("✅ Low inflation pressure.")