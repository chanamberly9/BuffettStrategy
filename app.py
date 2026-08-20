import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
import io
import time
import warnings
from scipy.optimize import minimize
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
warnings.filterwarnings('ignore')

# ------------------------------------------------------------
# Page config
# ------------------------------------------------------------
st.set_page_config(page_title="Buffett Stock Screener", layout="wide")

# Sidebar navigation
page = st.sidebar.radio("📂 Select Page", ["Screener", "Backtest"])

st.sidebar.markdown("---")

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
# Data fetching and metric computation (unchanged)
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

        if prev_revenue is not None and pd.notna(prev_revenue) and pd.notna(revenue) and prev_revenue != 0:
            rev_growth = (revenue / prev_revenue - 1) * 100
        else:
            rev_growth = np.nan
        prev_revenue = revenue

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

@st.cache_data(ttl=3600*24, show_spinner=False)
def load_all_data(tickers):
    summaries = []
    failed = []
    progress_bar = st.progress(0, text="Downloading data...")
    total = len(tickers)
    for i, sym in enumerate(tickers):
        try:
            df = get_metrics_for_ticker(sym)
            if df is not None and len(df) >= 4:
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
                t = yf.Ticker(sym)
                name = sym
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
        except:
            failed.append(sym)
        progress_bar.progress((i+1)/total, text=f"Processing {sym} ({i+1}/{total})")
        time.sleep(1)
    progress_bar.empty()
    if failed:
        st.warning(f"Failed to retrieve data for {len(failed)} tickers.")
    return pd.DataFrame(summaries)

# ------------------------------------------------------------
# Portfolio optimization functions (for Screener page)
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_historical_prices(tickers, period="5y"):
    """Fetch adjusted close prices for given tickers."""
    data = yf.download(tickers, period=period, group_by='ticker', auto_adjust=True, progress=False)
    if len(tickers) == 1:
        if isinstance(data, pd.Series):
            return data.to_frame('Close')
        if 'Close' in data:
            prices = data['Close']
        else:
            prices = data
    else:
        prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers if ticker in data.columns})
    return prices

def portfolio_stats(weights, mean_returns, cov_matrix):
    ret = np.sum(mean_returns * weights) * 252
    vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    return ret, vol

def optimize_portfolio(tickers, risk_free_rate=0.02, period="5y"):
    prices = fetch_historical_prices(tickers, period)
    if prices.empty or len(prices) < 2:
        return None, None, None
    returns = prices.pct_change().dropna()
    if returns.shape[0] < 10:
        return None, None, None
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n = len(tickers)
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
# Backtest function (revised for multi‑horizon)
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def get_backtest_data(tickers, start_date, end_date=None, benchmark_ticker='SPY'):
    """
    Fetch price data for the given tickers from start_date to end_date.
    Returns a DataFrame of cumulative returns for the equal‑weighted portfolio
    and the benchmark, aligned on common dates.
    """
    if end_date is None:
        end_date = datetime.now()
    # Ensure datetime
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)
    # Add a buffer to get complete data
    all_tickers = list(set(tickers + [benchmark_ticker]))
    try:
        data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
        if data.empty:
            return None, None
        if 'Close' in data.columns:
            prices = data['Close']
        else:
            prices = data
    except:
        return None, None

    # Filter tickers with full data
    valid_tickers = []
    for t in tickers:
        if t in prices.columns and prices[t].notna().all():
            valid_tickers.append(t)
    if len(valid_tickers) == 0:
        return None, None
    # Portfolio daily returns (equal-weighted)
    port_returns = prices[valid_tickers].pct_change().mean(axis=1).dropna()
    # Benchmark
    if benchmark_ticker in prices.columns:
        bench_returns = prices[benchmark_ticker].pct_change().dropna()
        common_dates = port_returns.index.intersection(bench_returns.index)
        port_returns = port_returns.loc[common_dates]
        bench_returns = bench_returns.loc[common_dates]
    else:
        bench_returns = None

    # Cumulative returns
    port_cum = (1 + port_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod() if bench_returns is not None else None

    return port_cum, bench_cum, valid_tickers

def compute_horizon_returns(port_cum, bench_cum, start_date, end_date):
    """
    Given cumulative series, compute total return for a specific period
    by indexing at start and end dates.
    """
    # If port_cum is empty, return NaN
    if port_cum.empty:
        return np.nan, np.nan
    # Align start and end within the series date range
    # Get the closest available dates (we'll use actual trading days)
    # We'll find the nearest date <= end_date and >= start_date
    port_idx = port_cum.index
    # Find indices
    start_mask = port_idx >= start_date
    if not start_mask.any():
        return np.nan, np.nan
    start_idx = port_idx[start_mask].min()
    end_mask = port_idx <= end_date
    if not end_mask.any():
        return np.nan, np.nan
    end_idx = port_idx[end_mask].max()
    if start_idx > end_idx:
        return np.nan, np.nan
    port_ret = port_cum.loc[end_idx] / port_cum.loc[start_idx] - 1
    if bench_cum is not None and not bench_cum.empty:
        bench_idx = bench_cum.index
        # Use the same dates if available, else the nearest
        b_start = bench_idx[bench_idx >= start_idx].min() if (bench_idx >= start_idx).any() else None
        b_end = bench_idx[bench_idx <= end_idx].max() if (bench_idx <= end_idx).any() else None
        if b_start is not None and b_end is not None and b_start <= b_end:
            bench_ret = bench_cum.loc[b_end] / bench_cum.loc[b_start] - 1
        else:
            bench_ret = np.nan
    else:
        bench_ret = np.nan
    return port_ret, bench_ret

# ------------------------------------------------------------
# Sidebar parameters (shared across pages)
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

# Portfolio optimization settings (only used in Screener page)
risk_free_rate = st.sidebar.number_input("Risk‑free rate (annual, %)", value=2.0, step=0.1) / 100
period_opt = st.sidebar.selectbox("Historical data period", ["1y", "2y", "3y", "5y", "10y"], index=3)

st.sidebar.markdown("---")
st.sidebar.caption("All parameters apply to both pages.")

# ------------------------------------------------------------
# Main logic: load data and render page
# ------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def get_tickers_cached():
    return get_sp500_tickers()

tickers = get_tickers_cached()

if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False

# Load data button (always visible)
if st.sidebar.button("Load/Refresh Data"):
    with st.spinner("Downloading financial data for all S&P 500 stocks... This may take 10–20 minutes."):
        data = load_all_data(tickers)
        st.session_state.data = data
        st.session_state.data_loaded = True
    st.success("Data loaded successfully!")

if not st.session_state.data_loaded:
    st.info("Click the 'Load/Refresh Data' button to start. The initial download may take a while.")
    st.stop()

data = st.session_state.data

# Apply common filter to get passing stocks
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

# ------------------------------------------------------------
# PAGE 1: Screener
# ------------------------------------------------------------
if page == "Screener":
    st.title("📈 Buffett‑Style Stock Screener")
    st.markdown("""
    This app screens S&P 500 stocks using quantitative proxies for Warren Buffett’s principles.  
    Adjust the filters in the sidebar to see which companies match your criteria.
    """)

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

        # Portfolio Builder
        st.markdown("---")
        st.subheader("📊 Build Optimal Risky Portfolio from Passing Stocks")
        st.markdown("Select stocks from the list below to construct the tangency portfolio (maximum Sharpe ratio).")

        ticker_list = passing['Ticker'].tolist()
        name_list = passing['Name'].tolist()
        options = [f"{ticker} - {name}" for ticker, name in zip(ticker_list, name_list)]
        default_selection = options[:min(10, len(options))]
        selected_options = st.multiselect("Choose stocks for portfolio (at least 2)", options, default=default_selection)
        selected_tickers = [opt.split(" - ")[0] for opt in selected_options]

        if len(selected_tickers) < 2:
            st.warning("Please select at least 2 stocks.")
        else:
            if st.button("Optimize Portfolio"):
                with st.spinner("Fetching historical data and optimizing..."):
                    weights, ret, vol, sharpe = optimize_portfolio(selected_tickers, risk_free_rate, period_opt)
                    if weights is not None:
                        weight_df = pd.DataFrame({
                            'Ticker': selected_tickers,
                            'Weight (%)': weights * 100
                        })
                        name_map = {ticker: name for ticker, name in zip(ticker_list, name_list)}
                        weight_df['Name'] = weight_df['Ticker'].map(name_map)
                        weight_df = weight_df[weight_df['Weight (%)'] > 0.01]
                        weight_df = weight_df.sort_values('Weight (%)', ascending=False).reset_index(drop=True)

                        st.success("Portfolio optimized!")
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Expected Annual Return", f"{ret*100:.2f}%")
                        col2.metric("Annual Volatility", f"{vol*100:.2f}%")
                        col3.metric("Sharpe Ratio", f"{sharpe:.3f}")
                        col4.metric("Number of Stocks", len(weight_df))

                        st.subheader("Optimal Weights")
                        st.dataframe(weight_df.style.format({'Weight (%)': '{:.2f}'}), use_container_width=True)

                        st.subheader("Weight Distribution")
                        st.bar_chart(weight_df.set_index('Ticker')['Weight (%)'])

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

# ------------------------------------------------------------
# PAGE 2: Backtest (multi‑horizon)
# ------------------------------------------------------------
else:  # Backtest page
    st.title("⏳ Backtest the Screen")
    st.markdown("""
    Select a **past start date** – the app will compute returns for **1‑month, 3‑month, 6‑month, 1‑year, 2‑year, and 5‑year** horizons (as far as data allows).  
    The portfolio is **equal‑weighted** across all stocks that pass today's screen.
    """)

    # Backtest UI
    default_start = datetime.now() - timedelta(days=365)
    start_date = st.date_input("Start date", default_start, max_value=datetime.now() - timedelta(days=1))

    # Get the tickers that pass the screen
    if passing.empty:
        st.warning("No stocks pass the current screen. Adjust the parameters to get some stocks.")
        st.stop()

    ticker_list = passing['Ticker'].tolist()
    name_map = dict(zip(passing['Ticker'], passing['Name']))
    st.info(f"Using **{len(ticker_list)}** stocks that pass the current screen.")

    if st.button("Run Backtest"):
        with st.spinner("Fetching historical prices and computing returns for all horizons..."):
            # Fetch data from start date to today
            end_date = datetime.now()
            port_cum, bench_cum, valid_tickers = get_backtest_data(ticker_list, start_date, end_date)
            if port_cum is None or port_cum.empty:
                st.error("No price data available for the selected start date. Try a later date.")
            else:
                # Define horizons in days
                horizons = {
                    "1 Month": 30,
                    "3 Months": 90,
                    "6 Months": 180,
                    "1 Year": 365,
                    "2 Years": 730,
                    "5 Years": 1825
                }
                results = []
                # For each horizon, compute return from start_date to start_date + horizon
                for label, days in horizons.items():
                    horizon_end = start_date + timedelta(days=days)
                    # Cap at end_date (today) if horizon extends beyond
                    if horizon_end > end_date:
                        horizon_end = end_date
                    port_ret, bench_ret = compute_horizon_returns(port_cum, bench_cum, start_date, horizon_end)
                    results.append({
                        'Horizon': label,
                        'Portfolio Return (%)': port_ret * 100 if pd.notna(port_ret) else np.nan,
                        'S&P 500 Return (%)': bench_ret * 100 if pd.notna(bench_ret) else np.nan,
                        'Outperformance (%)': (port_ret - bench_ret) * 100 if pd.notna(port_ret) and pd.notna(bench_ret) else np.nan
                    })
                results_df = pd.DataFrame(results)

                # Display table
                st.subheader("Returns by Horizon")
                st.dataframe(results_df.style.format('{:.2f}'), use_container_width=True)

                # Bar chart
                fig_bar = go.Figure()
                fig_bar.add_trace(go.Bar(
                    x=results_df['Horizon'],
                    y=results_df['Portfolio Return (%)'],
                    name='Portfolio',
                    marker_color='blue'
                ))
                fig_bar.add_trace(go.Bar(
                    x=results_df['Horizon'],
                    y=results_df['S&P 500 Return (%)'],
                    name='S&P 500',
                    marker_color='red'
                ))
                fig_bar.update_layout(
                    title='Return Comparison per Horizon',
                    xaxis_title='Horizon',
                    yaxis_title='Return (%)',
                    barmode='group',
                    hovermode='x unified'
                )
                st.plotly_chart(fig_bar, use_container_width=True)

                # Cumulative chart from start to today (or longest available)
                st.subheader("Cumulative Performance")
                # Plot only if we have at least some data
                fig_cum = go.Figure()
                fig_cum.add_trace(go.Scatter(
                    x=port_cum.index,
                    y=port_cum,
                    mode='lines',
                    name='Portfolio (equal‑weight)',
                    line=dict(color='blue')
                ))
                if bench_cum is not None and not bench_cum.empty:
                    common_idx = port_cum.index.intersection(bench_cum.index)
                    if len(common_idx) > 0:
                        bench_cum_aligned = bench_cum.loc[common_idx]
                        fig_cum.add_trace(go.Scatter(
                            x=common_idx,
                            y=bench_cum_aligned,
                            mode='lines',
                            name='S&P 500 (SPY)',
                            line=dict(color='red', dash='dash')
                        ))
                # Add vertical lines for each horizon end (only if within data range)
                for label, days in horizons.items():
                    horizon_end = start_date + timedelta(days=days)
                    if horizon_end > end_date:
                        horizon_end = end_date
                    if horizon_end in port_cum.index or (horizon_end < port_cum.index[-1] and horizon_end > port_cum.index[0]):
                        # find nearest date
                        nearest = port_cum.index[port_cum.index <= horizon_end].max()
                        if pd.notna(nearest):
                            fig_cum.add_vline(x=nearest, line_width=1, line_dash="dash", line_color="grey", annotation_text=label, annotation_position="top")
                fig_cum.update_layout(
                    title=f'Cumulative Return from {start_date.strftime("%Y-%m-%d")} to Today',
                    xaxis_title='Date',
                    yaxis_title='Cumulative Return (starting at 1)',
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
                )
                st.plotly_chart(fig_cum, use_container_width=True)

                # List of stocks included
                st.subheader("Stocks Included")
                st.write(f"Out of {len(ticker_list)} passing stocks, **{len(valid_tickers)}** had complete price data from {start_date.strftime('%Y-%m-%d')} onwards.")
                st.dataframe(pd.DataFrame({'Ticker': valid_tickers, 'Name': [name_map.get(t, t) for t in valid_tickers]}))

                # Interpretation
                # Find the latest horizon with available data (the one with longest days that has a valid return)
                latest_valid = results_df[results_df['Portfolio Return (%)'].notna()].iloc[-1] if not results_df[results_df['Portfolio Return (%)'].notna()].empty else None
                if latest_valid is not None:
                    last_ret = latest_valid['Portfolio Return (%)']
                    st.subheader("📝 Interpretation")
                    st.markdown(f"""
                    Over the period from **{start_date.strftime('%Y-%m-%d')}** to today, the equal‑weighted portfolio 
                    of Buffett‑screened stocks has shown a cumulative return of **{last_ret:.2f}%** for the **{latest_valid['Horizon']}** horizon.
                    The table above shows performance for all shorter horizons, giving you a complete picture of how the 
                    strategy performed over different time frames.
                    """)
                else:
                    st.info("No return data could be computed for any horizon. Try a later start date.")
