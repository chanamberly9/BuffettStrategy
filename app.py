import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import time
import warnings
from scipy.optimize import minimize
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
                # Get company name and valuation
                t = yf.Ticker(sym)
                name = sym  # fallback to ticker
                try:
                    info = t.info
                    name = info.get('longName') or info.get('shortName') or sym
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
                summary['Name'] = name
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
# Portfolio optimization functions
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_historical_prices(tickers, period="5y"):
    """Fetch adjusted close prices for given tickers."""
    data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, progress=False)
    # If only one ticker, adjust format
    if len(tickers) == 1:
        prices = data['Close'] if 'Close' in data else data
    else:
        prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers if ticker in data.columns})
    return prices

def portfolio_stats(weights, mean_returns, cov_matrix):
    """Compute expected return, volatility, and Sharpe ratio."""
    ret = np.sum(mean_returns * weights) * 252   # annualized
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))  # annualized
    return ret, vol

def tangency_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02):
    """Compute tangency (maximum Sharpe) portfolio weights."""
    n = len(mean_returns)
    inv_cov = np.linalg.pinv(cov_matrix)  # use pseudo-inverse for stability
    ones = np.ones(n)
    mu_minus_rf = mean_returns - risk_free_rate
    numerator = inv_cov @ mu_minus_rf
    denominator = ones @ numerator
    if denominator == 0:
        return None
    weights = numerator / denominator
    # Clip small negative weights to zero and renormalize? 
    # We'll allow short sales? Better to constrain to long-only.
    # We'll use a constrained optimization for long-only to be safe.
    # Since we want long-only, we'll use scipy minimize with bounds.
    # Use the analytical solution as initial guess.
    return weights

def optimize_portfolio(tickers, risk_free_rate=0.02, period="5y"):
    """
    Fetch historical data, compute mean returns and covariance, 
    and return tangency portfolio weights and stats.
    """
    prices = fetch_historical_prices(tickers, period)
    if prices.empty or len(prices) < 2:
        return None, None, None
    
    # Compute daily returns
    returns = prices.pct_change().dropna()
    if returns.shape[0] < 10:
        return None, None, None
    
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    
    # Use scipy optimize for long-only tangency portfolio
    n = len(tickers)
    # Objective: minimize negative Sharpe (or maximize Sharpe)
    def neg_sharpe(w):
        ret = np.sum(mean_returns * w) * 252
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w)))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else 0
        return -sharpe
    
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial = np.ones(n) / n
    result = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        return None, None, None
    weights = result.x
    ret_ann, vol_ann = portfolio_stats(weights, mean_returns, cov_matrix)
    sharpe = (ret_ann - risk_free_rate) / vol_ann if vol_ann > 0 else np.nan
    return weights, ret_ann, vol_ann, sharpe

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
# Portfolio optimization UI settings (in sidebar)
# ------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Portfolio Optimization")
risk_free_rate = st.sidebar.number_input("Risk‑free rate (annual, %)", value=2.0, step=0.1) / 100
period_opt = st.sidebar.selectbox("Historical data period", ["1y", "2y", "3y", "5y", "10y"], index=3)

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
    passing = data[
        (data['Years'] >= min_years) &
        (data['Avg Gross Margin (%)'] > gross_margin_min) &
        (data['Avg Op Margin (%)'] > op_margin_min) &
        (data['Avg ROE (%)'] > roe_min) &
        (data['Avg Debt/Equity'] < de_max) &
        (data['FCF Positive Count'] >= np.ceil(pass_ratio * data['Years'])) &
        (data['Rev Growth Positive Count'] >= np.ceil(pass_ratio * data['Years'])) &
        (data['P/E'] > 0) & (data['P/E'] < pe_max) &
        (data['P/B'] < pb_max) &
        (data['P/FCF'] < pfcf_max) &
        (data['Market Cap ($B)'] > market_cap_min)
    ]

    st.subheader(f"Results: {len(passing)} passing stocks")
    if not passing.empty:
        display_cols = ['Ticker', 'Name', 'Years', 'Avg ROE (%)', 'Avg Gross Margin (%)',
                        'Avg Op Margin (%)', 'Avg Debt/Equity', 'P/E', 'P/B', 'P/FCF', 'Market Cap ($B)']
        st.dataframe(passing[display_cols].reset_index(drop=True), use_container_width=True)

        csv = passing[display_cols].to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download results as CSV",
            data=csv,
            file_name='buffett_screen_results.csv',
            mime='text/csv',
        )

        # ------------------------------------------------------------
        # Portfolio Builder Section
        # ------------------------------------------------------------
        st.markdown("---")
        st.subheader("📊 Build Optimal Risky Portfolio from Passing Stocks")
        st.markdown("Select stocks from the list below to construct the tangency portfolio (maximum Sharpe ratio).")

        # Get list of tickers and names
        ticker_list = passing['Ticker'].tolist()
        name_list = passing['Name'].tolist()
        options = [f"{ticker} - {name}" for ticker, name in zip(ticker_list, name_list)]
        default_selection = options[:min(10, len(options))]  # default to first 10

        selected_options = st.multiselect("Choose stocks for portfolio (at least 2)", options, default=default_selection)
        selected_tickers = [opt.split(" - ")[0] for opt in selected_options]

        if len(selected_tickers) < 2:
            st.warning("Please select at least 2 stocks.")
        else:
            if st.button("Optimize Portfolio"):
                with st.spinner("Fetching historical data and optimizing..."):
                    weights, ret, vol, sharpe = optimize_portfolio(selected_tickers, risk_free_rate, period_opt)
                    if weights is not None:
                        # Create results DataFrame
                        weight_df = pd.DataFrame({
                            'Ticker': selected_tickers,
                            'Weight (%)': weights * 100
                        })
                        # Add company names if available
                        name_map = {ticker: name for ticker, name in zip(ticker_list, name_list)}
                        weight_df['Name'] = weight_df['Ticker'].map(name_map)
                        weight_df = weight_df[weight_df['Weight (%)'] > 0.01]  # drop near-zero weights
                        weight_df = weight_df.sort_values('Weight (%)', ascending=False).reset_index(drop=True)

                        st.success("Portfolio optimized!")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Expected Annual Return", f"{ret*100:.2f}%")
                        col2.metric("Annual Volatility", f"{vol*100:.2f}%")
                        col3.metric("Sharpe Ratio", f"{sharpe:.3f}")
                        col4.metric("Number of Stocks", len(weight_df))

                        # Display weights
                        st.subheader("Optimal Weights")
                        st.dataframe(weight_df.style.format({'Weight (%)': '{:.2f}'}), use_container_width=True)

                        # Simple pie chart using st.bar_chart? But better to use plotly if available.
                        # We'll use a simple bar chart.
                        st.subheader("Weight Distribution")
                        st.bar_chart(weight_df.set_index('Ticker')['Weight (%)'])

                        # Description
                        st.subheader("📝 Portfolio Description")
                        desc = f"""
                        This **optimal risky portfolio** (tangency portfolio) is constructed from the {len(weight_df)} stocks 
                        that passed your Buffett‑style filters. Using historical returns over the past **{period_opt}**, 
                        it maximizes the Sharpe ratio given a risk‑free rate of **{risk_free_rate*100:.2f}%**.
                        
                        - **Expected annual return:** {ret*100:.2f}%  
                        - **Annual volatility (risk):** {vol*100:.2f}%  
                        - **Sharpe ratio:** {sharpe:.3f}  
                        
                        The portfolio is well‑diversified across the selected stocks, with the largest allocation to 
                        **{weight_df.iloc[0]['Name']} ({weight_df.iloc[0]['Weight (%)']:.2f}%)** 
                        and the smallest to **{weight_df.iloc[-1]['Name']} ({weight_df.iloc[-1]['Weight (%)']:.2f}%)**.
                        """
                        st.markdown(desc)

                        # Download portfolio weights CSV
                        csv_port = weight_df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Download portfolio weights as CSV",
                            data=csv_port,
                            file_name='optimal_portfolio_weights.csv',
                            mime='text/csv',
                        )
                    else:
                        st.error("Optimization failed. Try selecting a longer history period or a different set of stocks.")
    else:
        st.info("No stocks match the current criteria. Try relaxing the filters.")
else:
    st.info("Click the 'Load/Refresh Data' button to start the screen. The initial download may take a while.")
