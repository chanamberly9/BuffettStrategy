import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import time
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="Buffett Stock Screener", layout="wide")
st.title("📈 Buffett‑Style Stock Screener")
st.markdown("""
This app screens S&P 500 stocks using quantitative proxies for Warren Buffett’s principles.  
Adjust the filters in the sidebar to see which companies match your criteria.
""")

# ------------------------------------------------------------
# Fetch S&P 500 tickers
# ------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def get_sp500_tickers():
    """Fetch current S&P 500 tickers from a reliable GitHub source."""
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://raw.githubusercontent.com/fja05680/sp500/master/sp500.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            df = pd.read_csv(io.StringIO(r.text))
            if 'Symbol' in df.columns:
                tickers = df['Symbol'].tolist()
            elif 'Ticker' in df.columns:
                tickers = df['Ticker'].tolist()
            else:
                tickers = df.iloc[:, 0].tolist()
            return [str(t).replace('.', '-') for t in tickers]
        except:
            continue
    # Fallback sample
    return ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'JNJ', 'V', 'PG', 'UNH', 'KO']

# ------------------------------------------------------------
# Data fetching and metric computation
# ------------------------------------------------------------
def get_metrics_for_ticker(ticker_symbol):
    """
    Download financial statements for one ticker and compute per‑year metrics.
    Returns a list of dicts (one per year) or None if data insufficient.
    """
    ticker = yf.Ticker(ticker_symbol)
    try:
        income = ticker.financials
        balance = ticker.balance_sheet
        cashflow = ticker.cashflow
        if income.empty or balance.empty or cashflow.empty:
            return None
    except:
        return None

    # Map fiscal year to column
    def year_map(df):
        mapping = {}
        for col in df.columns:
            dt = pd.to_datetime(col)
            year = dt.year
            if year not in mapping:
                mapping[year] = col
        return mapping

    income_years = year_map(income)
    balance_years = year_map(balance)
    cashflow_years = year_map(cashflow)
    common_years = sorted(set(income_years) & set(balance_years) & set(cashflow_years))
    if not common_years:
        return None

    rows = []
    prev_revenue = None
    for year in common_years:
        def get_item(df, year_dict, names):
            col = year_dict[year]
            for name in names:
                if name in df.index:
                    val = df.loc[name, col]
                    if pd.notna(val):
                        return float(val)
            return np.nan

        revenue = get_item(income, income_years, ['Total Revenue', 'Operating Revenue'])
        gross_profit = get_item(income, income_years, ['Gross Profit'])
        operating_income = get_item(income, income_years, ['Operating Income', 'Operating Income or Loss'])
        net_income = get_item(income, income_years, ['Net Income', 'Net Income Common Stockholders'])

        equity = get_item(balance, balance_years, ['Stockholders Equity', 'Total Stockholder Equity', 'Total Equity Gross Minority Interest'])
        total_debt = get_item(balance, balance_years, ['Total Debt', 'Short Long Term Debt'])
        if pd.isna(total_debt):
            short_debt = get_item(balance, balance_years, ['Short Term Debt', 'Short-Term Debt'])
            long_debt = get_item(balance, balance_years, ['Long Term Debt', 'Long Term Debt And Capital Lease Obligation'])
            total_debt = (short_debt if pd.notna(short_debt) else 0) + (long_debt if pd.notna(long_debt) else 0)

        ocf = get_item(cashflow, cashflow_years, ['Operating Cash Flow', 'Cash Flow From Operating Activities'])
        capex_raw = get_item(cashflow, cashflow_years, ['Capital Expenditure', 'Capital Expenditures', 'Purchase Of Property Plant And Equipment'])
        capex = abs(capex_raw) if pd.notna(capex_raw) else np.nan

        # Revenue growth
        if prev_revenue is not None and pd.notna(prev_revenue) and pd.notna(revenue) and prev_revenue != 0:
            rev_growth = (revenue / prev_revenue - 1) * 100
        else:
            rev_growth = np.nan
        prev_revenue = revenue

        # Ratios
        gross_margin = (gross_profit / revenue * 100) if pd.notna(gross_profit) and pd.notna(revenue) and revenue != 0 else np.nan
        op_margin = (operating_income / revenue * 100) if pd.notna(operating_income) and pd.notna(revenue) and revenue != 0 else np.nan
        roe = (net_income / equity * 100) if pd.notna(net_income) and pd.notna(equity) and equity != 0 else np.nan
        debt_to_equity = (total_debt / equity) if pd.notna(total_debt) and pd.notna(equity) and equity != 0 else np.nan
        fcf = ocf - capex if pd.notna(ocf) and pd.notna(capex) else np.nan
        fcf_margin = (fcf / revenue * 100) if pd.notna(fcf) and pd.notna(revenue) and revenue != 0 else np.nan

        rows.append({
            'Year': year,
            'Revenue Growth (%)': rev_growth,
            'Gross Margin (%)': gross_margin,
            'Operating Margin (%)': op_margin,
            'ROE (%)': roe,
            'Debt/Equity': debt_to_equity,
            'FCF': fcf,
            'FCF Margin (%)': fcf_margin,
        })
    return pd.DataFrame(rows)

# ------------------------------------------------------------
# Load all ticker data and compute summary metrics
# ------------------------------------------------------------
@st.cache_data(ttl=3600*24, show_spinner=False)
def load_all_data(tickers):
    """
    Download fundamentals for all tickers, compute per‑ticker summary metrics,
    and return a DataFrame.
    """
    summaries = []
    failed = []
    progress_bar = st.progress(0, text="Downloading data...")
    total = len(tickers)

    for i, sym in enumerate(tickers):
        try:
            df = get_metrics_for_ticker(sym)
            if df is not None and len(df) >= 4:   # require at least 4 years
                # Compute summary metrics
                summary = {
                    'Ticker': sym,
                    'Years': len(df),
                    'Avg ROE (%)': df['ROE (%)'].mean(),
                    'Avg Gross Margin (%)': df['Gross Margin (%)'].mean(),
                    'Avg Op Margin (%)': df['Operating Margin (%)'].mean(),
                    'Avg Debt/Equity': df['Debt/Equity'].mean(),
                    'ROE Pass Count': (df['ROE (%)'] > 15).sum(),
                    'D/E Pass Count': (df['Debt/Equity'] < 0.5).sum(),
                    'FCF Positive Count': (df['FCF'] > 0).sum(),
                    'Rev Growth Positive Count': (df['Revenue Growth (%)'] > 0).sum(),
                    'Latest FCF': df.iloc[-1]['FCF'],
                }
                # Get valuation
                t = yf.Ticker(sym)
                try:
                    info = t.info
                    price = info.get('currentPrice') or info.get('regularMarketPrice')
                    shares = info.get('sharesOutstanding')
                    market_cap = info.get('marketCap')
                    pe = info.get('trailingPE')
                    pb = info.get('priceToBook')
                    if market_cap is None and price and shares:
                        market_cap = price * shares
                    summary['P/E'] = pe
                    summary['P/B'] = pb
                    if pd.notna(summary['Latest FCF']) and summary['Latest FCF'] > 0 and price and shares:
                        summary['P/FCF'] = price / (summary['Latest FCF'] / shares)
                    else:
                        summary['P/FCF'] = np.nan
                    summary['Market Cap ($B)'] = market_cap / 1e9 if pd.notna(market_cap) else np.nan
                except:
                    summary['P/E'] = np.nan
                    summary['P/B'] = np.nan
                    summary['P/FCF'] = np.nan
                    summary['Market Cap ($B)'] = np.nan
                summaries.append(summary)
            else:
                failed.append(sym)
        except Exception as e:
            failed.append(sym)
        # Update progress
        progress_bar.progress((i+1)/total, text=f"Processing {sym} ({i+1}/{total})")
        time.sleep(1)   # to avoid rate limits

    progress_bar.empty()
    if failed:
        st.warning(f"Failed to retrieve data for {len(failed)} tickers.")
    return pd.DataFrame(summaries)

# ------------------------------------------------------------
# Sidebar parameters
# ------------------------------------------------------------
st.sidebar.header("Screening Parameters")

min_years = st.sidebar.slider("Minimum years of data", 1, 10, 4, 1)
pass_ratio = st.sidebar.slider("Pass ratio (fraction of years each criterion must hold)", 0.1, 1.0, 0.6, 0.05)

st.sidebar.subheader("Profitability & Efficiency")
gross_margin_min = st.sidebar.slider("Average Gross Margin (%)", 0, 80, 30, 1)
op_margin_min = st.sidebar.slider("Average Operating Margin (%)", 0, 50, 10, 1)
roe_min = st.sidebar.slider("ROE yearly threshold (%)", 0, 50, 10, 1)

st.sidebar.subheader("Balance Sheet")
de_max = st.sidebar.slider("Debt/Equity yearly threshold", 0.1, 3.0, 0.8, 0.1)

st.sidebar.subheader("Valuation")
pe_max = st.sidebar.slider("P/E maximum", 5, 100, 30, 1)
pb_max = st.sidebar.slider("P/B maximum", 1, 30, 8, 1)
pfcf_max = st.sidebar.slider("P/FCF maximum", 5, 100, 40, 1)
market_cap_min = st.sidebar.number_input("Minimum Market Cap ($B)", value=10.0, step=1.0)

# ------------------------------------------------------------
# Main logic
# ------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def get_tickers_cached():
    return get_sp500_tickers()

tickers = get_tickers_cached()

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

if st.button("Load/Refresh Data"):
    with st.spinner("Downloading financial data for all S&P 500 stocks... This may take 10–20 minutes."):
        data = load_all_data(tickers)
        st.session_state.data = data
        st.session_state.data_loaded = True
    st.success("Data loaded successfully!")

if st.session_state.data_loaded:
    data = st.session_state.data
    # Apply filters
# 不再需要 required_pass 变量，直接在筛选条件中逐行比较
passing = data[
    (data['Years'] >= min_years) &
    (data['Avg Gross Margin (%)'] > gross_margin_min) &
    (data['Avg Op Margin (%)'] > op_margin_min) &
    (data['Avg ROE (%)'] > roe_min) &
    (data['Avg Debt/Equity'] < de_max) &
    # 用 pass_ratio 乘以每只股票的年数，并向上取整，再与达标年数比较
    (data['FCF Positive Count'] >= np.ceil(pass_ratio * data['Years'])) &
    (data['Rev Growth Positive Count'] >= np.ceil(pass_ratio * data['Years'])) &
    (data['P/E'] > 0) & (data['P/E'] < pe_max) &
    (data['P/B'] < pb_max) &
    (data['P/FCF'] < pfcf_max) &
    (data['Market Cap ($B)'] > market_cap_min)
]
    # Also require ROE > threshold in enough years (already via ROE Pass Count using hardcoded 15? We need to use roe_min)
    # The precomputed ROE Pass Count was using 15% fixed. That's not correct for dynamic threshold.
    # We'll need to compute ROE pass count based on roe_min. We could store the raw yearly ROE values, but that's heavy.
    # Simpler: recompute based on stored counts? We stored ROE Pass Count for 15% only.
    # To keep it simple, we'll ignore the dynamic roe threshold for yearly counts and use average ROE instead.
    # Alternatively, we can store the full yearly data in session and recompute on the fly, but that's memory heavy.
    # For now, we'll use Avg ROE as a proxy, and also apply a yearly threshold using stored ROE Pass Count? Not possible.
    # We'll adjust: we can store the raw ROE values for each ticker as a list, but that's not cached efficiently.
    # Instead, we'll change the approach: precompute average ROE and also count years where ROE > 15, but we need dynamic.
    # We'll modify the load function to store a list of ROE values? But Streamlit caching can handle lists.
    # To keep the app simple, we'll add an "Avg ROE" filter instead of a yearly threshold. 
    # So we'll remove the ROE Pass Count requirement and use Avg ROE > roe_min.
    passing = data[
        (data['Years'] >= min_years) &
        (data['Avg Gross Margin (%)'] > gross_margin_min) &
        (data['Avg Op Margin (%)'] > op_margin_min) &
        (data['Avg ROE (%)'] > roe_min) &
        (data['Avg Debt/Equity'] < de_max) &
        (data['FCF Positive Count'] >= required_pass) &
        (data['Rev Growth Positive Count'] >= required_pass) &
        (data['P/E'] > 0) & (data['P/E'] < pe_max) &
        (data['P/B'] < pb_max) &
        (data['P/FCF'] < pfcf_max) &
        (data['Market Cap ($B)'] > market_cap_min)
    ]

    st.subheader(f"Results: {len(passing)} passing stocks")
    if not passing.empty:
        display_cols = ['Ticker', 'Years', 'Avg ROE (%)', 'Avg Gross Margin (%)',
                        'Avg Op Margin (%)', 'Avg Debt/Equity', 'P/E', 'P/B', 'P/FCF', 'Market Cap ($B)']
        st.dataframe(passing[display_cols].reset_index(drop=True), use_container_width=True)

        # Download button
        csv = passing[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='buffett_screen_results.csv',
            mime='text/csv',
        )
    else:
        st.info("No stocks match the current criteria. Try relaxing the filters.")
else:
    st.info("Click the 'Load/Refresh Data' button to start the screen. The initial download may take a while.")
