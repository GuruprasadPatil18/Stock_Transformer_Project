import streamlit as st
import pandas as pd
import numpy as np
import torch
import plotly.graph_objects as go
import yfinance as yf
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from sklearn.preprocessing import MinMaxScaler
from model import StockTransformer
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from GoogleNews import GoogleNews
from streamlit_option_menu import option_menu # NEW LIBRARY

# --- Page Config (Must be first) ---
st.set_page_config(
    page_title="FinFormer | AI Stock Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Download NLTK ---
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

# --- Settings ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "transformer_stock_model.pth"

# --- Helper Functions (Same as before) ---
def fetch_stock_data(ticker):
    try:
        data = yf.download(ticker, period="2y", interval="1d")
        if data.empty: return None
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data['RSI'] = RSIIndicator(close=data['Close'], window=14).rsi()
        macd = MACD(close=data['Close'])
        data['MACD'] = macd.macd()
        data['EMA_20'] = EMAIndicator(close=data['Close'], window=20).ema_indicator()
        data.dropna(inplace=True)
        return data
    except: return None

def fetch_fundamentals(ticker):
    try:
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        return {
            "Market Cap": info.get("marketCap", "N/A"),
            "P/E Ratio": info.get("trailingPE", "N/A"),
            "52W High": info.get("fiftyTwoWeekHigh", "N/A"),
            "Sector": info.get("sector", "N/A"),
            "Business Summary": info.get("longBusinessSummary", "No summary.")[:200] + "..."
        }
    except: return None

def fetch_news_google(ticker):
    try:
        clean_ticker = ticker.replace(".NS", "")
        googlenews = GoogleNews(lang='en', region='IN')
        googlenews.clear()
        googlenews.search(f"{clean_ticker} stock financial news")
        googlenews.get_page(1)
        news_list = googlenews.results()
        if not news_list: return None, 0
        sia = SentimentIntensityAnalyzer()
        headlines = []
        total_score = 0
        for item in news_list[:5]:
            title = item.get('title', ''); link = item.get('link', '#')
            if not title: continue
            score = sia.polarity_scores(title)['compound']
            total_score += score
            headlines.append({'title': title, 'score': score, 'link': link})
        if not headlines: return None, 0
        return headlines, total_score / len(headlines)
    except: return None, 0

def preprocess_live_data(df):
    feature_cols = ['Close', 'RSI', 'MACD', 'EMA_20']
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(df[feature_cols].values)
    sequence_length = 60
    X = []
    for i in range(len(data_scaled) - sequence_length):
        X.append(data_scaled[i : i + sequence_length])
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).to(DEVICE)
    return X_tensor, scaler, data_scaled

@st.cache_resource
def load_model():
    model = StockTransformer(input_dim=4, d_model=64, nhead=4, num_layers=2, dropout=0.1)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

# --- CUSTOM CSS (The "Look Good" Part) ---
st.markdown("""
<style>
    /* Make metrics stand out */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #00FF00; /* Neon Green */
    }
    /* Card-like background for containers */
    div.css-1r6slb0.e1tzin5v2 {
        background-color: #1E1E1E;
        border: 1px solid #333;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# --- SIDEBAR (New Design) ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3309/3309991.png", width=90) # Placeholder Icon
    st.title("FinFormer")
    
    # NEW: Option Menu Navigation
    selected = option_menu(
        menu_title=None, 
        options=["Dashboard", "News Analysis", "Backtest", "About"], 
        icons=["graph-up-arrow", "newspaper", "cash-coin", "info-circle"], 
        menu_icon="cast", 
        default_index=0,
        styles={
            "container": {"padding": "0!important", "background-color": "#161B22"},
            "icon": {"color": "orange", "font-size": "18px"}, 
            "nav-link": {"font-size": "16px", "text-align": "left", "margin":"0px", "--hover-color": "#262626"},
            "nav-link-selected": {"background-color": "#00FF00", "color": "black"},
        }
    )
    
    st.divider()
    st.header("Search")
    default_ticker = "RELIANCE.NS"
    ticker_input = st.text_input("Ticker Symbol", default_ticker)
    
    if st.button("Run Analysis", type="primary"):
        st.session_state['ticker'] = ticker_input
    else:
        if 'ticker' not in st.session_state: st.session_state['ticker'] = default_ticker

current_ticker = st.session_state['ticker']
model = load_model()

# --- MAIN DATA LOADING ---
with st.spinner(f"Processing Billions of Parameters for {current_ticker}..."):
    df = fetch_stock_data(current_ticker)
    fundamentals = fetch_fundamentals(current_ticker)
    headlines, sentiment_score = fetch_news_google(current_ticker)

# --- TABS LOGIC ---
if df is not None:
    # Precompute Inference
    X_tensor, scaler, full_data_scaled = preprocess_live_data(df)
    with torch.no_grad():
        predictions_scaled = model(X_tensor).cpu().numpy()
        
    def inverse_scale(scaled, scaler):
        dummy = np.zeros((len(scaled), 4)); dummy[:, 0] = scaled.flatten()
        return scaler.inverse_transform(dummy)[:, 0]

    predicted_prices = inverse_scale(predictions_scaled, scaler)
    actual_prices = df['Close'].values[60:]
    
    # Future Prediction
    last_seq = torch.tensor(full_data_scaled[-60:], dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad(): future_scaled = model(last_seq).item()
    dummy_fut = np.zeros((1, 4)); dummy_fut[0, 0] = future_scaled
    future_price = scaler.inverse_transform(dummy_fut)[0, 0]
    current_price = df['Close'].iloc[-1]
    price_change = future_price - current_price

    # --- TAB 1: DASHBOARD ---
    if selected == "Dashboard":
        st.subheader(f"📊 Market Overview: {current_ticker}")
        
        # Fundamentals Row
        if fundamentals:
            c1, c2, c3, c4 = st.columns(4)
            def format_large(num):
                if isinstance(num, (int, float)):
                    if num > 1e12: return f"₹{num/1e12:.2f}T"
                    if num > 1e9: return f"₹{num/1e9:.2f}B"
                return num
            c1.metric("Market Cap", format_large(fundamentals["Market Cap"]))
            c2.metric("P/E Ratio", fundamentals["P/E Ratio"])
            c3.metric("52W High", f"₹{fundamentals['52W High']}")
            c4.metric("Sector", fundamentals["Sector"])
            st.markdown("---")

        # Prediction Row
        col1, col2, col3 = st.columns([1,1,2])
        col1.metric("Current Price", f"₹{current_price:.2f}")
        col2.metric("AI Forecast (24h)", f"₹{future_price:.2f}", delta=f"{price_change:.2f}")
        
        # Verdict Badge
        verdict = "HOLD"
        v_color = "gray"
        if price_change > 0: verdict = "BULLISH 🚀"; v_color = "green"
        if price_change < 0: verdict = "BEARISH 📉"; v_color = "red"
        
        col3.markdown(f"""
            <div style="text-align: center; background-color: #333; padding: 10px; border-radius: 10px;">
                <h3 style="margin:0; color: {v_color};">{verdict}</h3>
                <p style="margin:0;">AI Confidence Score</p>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("###") # Spacer
        
        # Chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=actual_prices[-200:], mode='lines', name='Actual', line=dict(color='#00BFFF', width=2)))
        fig.add_trace(go.Scatter(y=predicted_prices[-200:], mode='lines', name='AI Predicted', line=dict(color='#FF4B4B', dash='dash', width=2)))
        fig.update_layout(
            title=f"Transformer Model Performance (Last 200 Days)",
            height=500,
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)', # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: NEWS ANALYSIS ---
    elif selected == "News Analysis":
        st.subheader(f"📰 AI Sentiment Analysis")
        
        score_col, chart_col = st.columns([1, 2])
        
        with score_col:
            st.metric("Sentiment Score", f"{sentiment_score:.2f}")
            if sentiment_score > 0.1:
                st.success("Market Mood: POSITIVE")
            elif sentiment_score < -0.1:
                st.error("Market Mood: NEGATIVE")
            else:
                st.info("Market Mood: NEUTRAL")
                
        with chart_col:
            if headlines:
                for news in headlines:
                    st.markdown(f"""
                    <div style="background-color: #262730; padding: 15px; border-radius: 10px; margin-bottom: 10px;">
                        <a href="{news['link']}" style="text-decoration: none; color: white; font-weight: bold; font-size: 18px;">{news['title']}</a>
                        <br>
                        <span style="color: {'#00FF00' if news['score'] > 0 else '#FF4B4B'};">Sentiment: {news['score']:.2f}</span>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.write("No major news headlines found.")

    # --- TAB 3: BACKTEST ---
    elif selected == "Backtest":
        st.subheader("💰 Strategy Simulator")
        initial = 10000
        cash = initial; shares = 0; in_market = False
        
        # Fast Calculation
        for i in range(len(predicted_prices)-1):
            curr = actual_prices[i]; pred = predicted_prices[i+1]
            if pred > curr * 1.005 and not in_market:
                shares = cash / curr; cash = 0; in_market = True
            elif pred < curr and in_market:
                cash = shares * curr; shares = 0; in_market = False
        
        final_val = shares * actual_prices[-1] if in_market else cash
        ret = ((final_val - initial)/initial)*100
        buy_hold = initial * (actual_prices[-1]/actual_prices[0])
        
        c1, c2 = st.columns(2)
        c1.metric("Buy & Hold Strategy", f"₹{buy_hold:.2f}", delta=f"{((buy_hold-initial)/initial)*100:.1f}%")
        c2.metric("AI Model Strategy", f"₹{final_val:.2f}", delta=f"{ret:.1f}%")
        
        st.info("Logic: The AI buys when the model predicts a price increase > 0.5% for the next day, and sells otherwise.")

    # --- TAB 4: ABOUT ---
    elif selected == "About":
        st.markdown("## 🧠 About Stock Transformer")
        st.write("This tool uses a **Self-Attention Transformer Model** trained on 5 years of historical data.")
        st.write("It analyzes 4 distinct features per day:")
        st.code("1. Close Price\n2. RSI (14)\n3. MACD\n4. EMA (20)")
        st.write("Developed for Major Project Submission.")

else:
    st.error("Error fetching data. Please check ticker.")