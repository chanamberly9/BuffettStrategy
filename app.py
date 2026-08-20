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
page = st.sidebar.radio("📂 Select Page", ["Screener (Today)", "Historical Backtest"])

st.sidebar.markdown("---")

# ------------------------------------------------------------
# Fetch S&P 500 tickers (current)
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
# Data fetching and metric computation (with optional cutoff date)
# ------------------------------------------------------------
def get_metrics_for_ticker(ticker_symbol, cutoff_date=None):
    """
    Download financial statements and compute per‑year metrics.
    If cutoff_date is given, only include fiscal years ending <= cutoff_date.
    Returns DataFrame of yearly metrics or None if insufficient.
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

    # Filter columns by cutoff_date if provided
    def filter_by_cutoff(df, cutoff):
        if cutoff is None:
            return df
        cutoff = pd.to_datetime(cutoff)
        keep_cols = [col for col in df.columns if pd.to_datetime(col) <= cutoff]
        return df[keep_cols]

    income = filter_by_cutoff(income, cutoff_date)
    balance = filter_by_cutoff(balance, cutoff_date)
    cashflow = filter_by_cutoff(cashflow, cutoff_date)

    if income.empty or balance.empty or cashflow.empty:
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
    if len(common_years) < 4:   # need at least 4 years for screening
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
def load_all_data_as_of(cutoff_date, tickers):
    """
    Load fundamentals for all tickers, but only use financial statement dates <= cutoff_date.
    Returns DataFrame with summary metrics for stocks that have at least 4 years of data.
    """
    summaries = []
    failed = []
    total = len(tickers)
    progress_bar = st.progress(0, text=f"Loading data as of {cutoff_date.strftime('%Y-%m-%d')}...")
    for i, sym in enumerate(tickers):
        try:
            df = get_metrics_for_ticker(sym, cutoff_date)
            if df is not None and len(df) >= 4:
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
                # Get company name and valuation (using info as of today, but we keep it as proxy)
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
        time.sleep(0.5)   # avoid rate limits
    progress_bar.empty()
    if failed:
        st.warning(f"Failed to retrieve data for {len(failed)} tickers (as of {cutoff_date.strftime('%Y-%m-%d')}).")
    return pd.DataFrame(summaries)

# ------------------------------------------------------------
# Screening function (apply filters to a DataFrame)
# ------------------------------------------------------------
def apply_screen(df, min_years, pass_ratio, gross_margin_min, op_margin_min, roe_min, de_max, pe_max, pb_max, pfcf_max, market_cap_min):
    """Apply the Buffett filters to a DataFrame of summary metrics."""
    return df[
        (df['Years'] >= min_years) &
        (df['Avg Gross Margin (%)'] > gross_margin_min) &
        (df['Avg Op Margin (%)'] > op_margin_min) &
        (df['Avg ROE (%)'] > roe_min) &
        (df['Avg Debt/Equity'] < de_max) &
        (df['FCF Positive Count'] >= np.ceil(pass_ratio * df['Years'])) &
        (df['Rev Growth Positive Count'] >= np.ceil(pass_ratio * df['Years'])) &
        (df['P/E'] > 0) & (df['P/E'] < pe_max) &
        (df['P/B'] < pb_max) &
        (df['P/FCF'] < pfcf_max) &
        (df['Market Cap ($B)'] > market_cap_min)
    ]

# ------------------------------------------------------------
# Portfolio optimization functions (for historical data)
# ------------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_historical_prices_until(tickers, end_date, lookback_years=5):
    """
    Fetch adjusted close prices for tickers from (end_date - lookback_years) to end_date.
    """
    start = end_date - timedelta(days=lookback_years*365)
    data = yf.download(tickers, start=start, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        return pd.DataFrame()
    if 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    # Keep only tickers that are in the columns
    valid_tickers = [t for t in tickers if t in prices.columns]
    if not valid_tickers:
        return pd.DataFrame()
    return prices[valid_tickers]

def optimize_portfolio_with_history(tickers, end_date, risk_free_rate=0.02, lookback_years=5):
    """
    Optimize tangency portfolio using historical data up to end_date.
    Returns weights, annualized return, volatility, Sharpe, and the returns series.
    """
    prices = fetch_historical_prices_until(tickers, end_date, lookback_years)
    if prices.empty or len(prices) < 10:
        return None, None, None, None, None
    returns = prices.pct_change().dropna()
    if returns.shape[0] < 10:
        return None, None, None, None, None
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values
    n = len(tickers)
    def neg_sharpe(w):
        ret = np.sum(mean_returns * w) * 252
        vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix * 252, w)))
        sharpe = (ret - risk_free_rate) / vol if vol > 0 else -1e10
        return -sharpe
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n))
    initial = np.ones(n) / n
    result = minimize(neg_sharpe, initial, method='SLSQP', bounds=bounds, constraints=constraints)
    if not result.success:
        return None, None, None, None, None
    weights = result.x
    ret_ann = np.sum(mean_returns * weights) * 252
    vol_ann = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
    sharpe = (ret_ann - risk_free_rate) / vol_ann if vol_ann > 0 else np.nan
    return weights, ret_ann, vol_ann, sharpe, returns

# ------------------------------------------------------------
# Performance tracking from a given date with fixed weights
# ------------------------------------------------------------
def track_portfolio_performance(tickers, weights, start_date, end_date=None, benchmark='SPY'):
    """
    Track equal-weighted or weighted portfolio from start_date to end_date.
    Returns cumulative returns for portfolio and benchmark.
    """
    if end_date is None:
        end_date = datetime.now()
    all_tickers = list(set(tickers + [benchmark]))
    data = yf.download(all_tickers, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if data.empty:
        return None, None
    if 'Close' in data.columns:
        prices = data['Close']
    else:
        prices = data
    # Ensure all tickers exist
    valid_tickers = [t for t in tickers if t in prices.columns]
    if not valid_tickers:
        return None, None
    # Align weights to valid tickers
    w = np.array([weights[list(tickers).index(t)] for t in valid_tickers])
    w = w / w.sum()  # renormalise
    # Daily returns
    port_returns = prices[valid_tickers].pct_change().dot(w)
    port_returns = port_returns.dropna()
    if benchmark in prices.columns:
        bench_returns = prices[benchmark].pct_change().dropna()
        common_dates = port_returns.index.intersection(bench_returns.index)
        port_returns = port_returns.loc[common_dates]
        bench_returns = bench_returns.loc[common_dates]
    else:
        bench_returns = None
    port_cum = (1 + port_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod() if bench_returns is not None else None
    return port_cum, bench_cum

def compute_multi_horizon_returns(port_cum, bench_cum, start_date, end_date):
    """Compute cumulative returns for standard horizons."""
    horizons = {
        "1 Month": 30,
        "3 Months": 90,
        "6 Months": 180,
        "1 Year": 365,
        "2 Years": 730,
        "5 Years": 1825
    }
    results = []
    for label, days in horizons.items():
        horizon_end = start_date + timedelta(days=days)
        if horizon_end > end_date:
            horizon_end = end_date
        # Find nearest dates in the cumulative series
        port_idx = port_cum.index
        # start index: closest <= start_date
        start_mask = port_idx >= start_date
        if not start_mask.any():
            port_ret = np.nan
        else:
            start_idx = port_idx[start_mask].min()
            end_mask = port_idx <= horizon_end
            if not end_mask.any():
                port_ret = np.nan
            else:
                end_idx = port_idx[end_mask].max()
                if start_idx > end_idx:
                    port_ret = np.nan
                else:
                    port_ret = port_cum.loc[end_idx] / port_cum.loc[start_idx] - 1
        # Benchmark
        if bench_cum is not None and not bench_cum.empty:
            bench_idx = bench_cum.index
            b_start = bench_idx[bench_idx >= start_idx].min() if (bench_idx >= start_idx).any() else None
            b_end = bench_idx[bench_idx <= horizon_end].max() if (bench_idx <= horizon_end).any() else None
            if b_start is not None and b_end is not None and b_start <= b_end:
                bench_ret = bench_cum.loc[b_end] / bench_cum.loc[b_start] - 1
            else:
                bench_ret = np.nan
        else:
            bench_ret = np.nan
        results.append({
            'Horizon': label,
            'Portfolio Return (%)': port_ret * 100 if pd.notna(port_ret) else np.nan,
            'S&P 500 Return (%)': bench_ret * 100 if pd.notna(bench_ret) else np.nan,
            'Outperformance (%)': (port_ret - bench_ret) * 100 if pd.notna(port_ret) and pd.notna(bench_ret) else np.nan
        })
    return pd.DataFrame(results)

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

# Portfolio optimization settings (used in backtest)
risk_free_rate = st.sidebar.number_input("Risk‑free rate (annual, %)", value=2.0, step=0.1) / 100
lookback_years = st.sidebar.slider("Lookback years for optimisation", 1, 10, 5, 1)

st.sidebar.markdown("---")
st.sidebar.caption("All parameters apply to both pages.")

# ------------------------------------------------------------
# Main logic: load current data for the Screener page
# ------------------------------------------------------------
@st.cache_data(ttl=3600*24)
def get_tickers_cached():
    return get_sp500_tickers()

tickers = get_tickers_cached()

if 'data_loaded_today' not in st.session_state:
    st.session_state.data_loaded_today = False

# Load data for today's screener
if st.sidebar.button("Load/Refresh Today's Data"):
    with st.spinner("Downloading financial data for all S&P 500 stocks..."):
        today_data = load_all_data_as_of(datetime.now(), tickers)
        st.session_state.today_data = today_data
        st.session_state.data_loaded_today = True
    st.success("Today's data loaded successfully!")

if not st.session_state.data_loaded_today:
    st.info("Click 'Load/Refresh Today's Data' to start the screener.")
    # We still allow the backtest page to load its own data as needed.
    # But we need the tickers list anyway.

# ------------------------------------------------------------
# PAGE 1: Screener (using today's data)
# ------------------------------------------------------------
if page == "Screener (Today)":
    st.title("📈 Buffett‑Style Stock Screener")
    st.markdown("""
    This app screens S&P 500 stocks using quantitative proxies for Warren Buffett’s principles.  
    Adjust the filters in the sidebar to see which companies match your criteria.
    """)

    if st.session_state.data_loaded_today:
        data = st.session_state.today_data
        passing = apply_screen(data, min_years, pass_ratio, gross_margin_min, op_margin_min,
                               roe_min, de_max, pe_max, pb_max, pfcf_max, market_cap_min)

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

            # Portfolio Builder (same as before, using today's data)
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
                if st.button("Optimize Portfolio (Today)"):
                    with st.spinner("Fetching historical data and optimizing..."):
                        weights, ret, vol, sharpe, _ = optimize_portfolio_with_history(selected_tickers, datetime.now(), risk_free_rate, lookback_years)
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
                            that passed your Buffett‑style filters. Using historical returns over the past **{lookback_years} years**, 
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
                            st.error("Optimization failed. Try different stocks or a longer lookback period.")
        else:
            st.info("No stocks match the current criteria. Try relaxing the filters.")
    else:
        st.info("Please load today's data using the sidebar button.")

# ------------------------------------------------------------
# PAGE 2: Historical Backtest
# ------------------------------------------------------------
else:  # Historical Backtest
    st.title("⏳ Historical Backtest: Screen + Optimize at a Past Date")
    st.markdown("""
    Choose a **past start date**. The app will:
    1. **Re‑run the Buffett screen** using only financial statements published before that date.
    2. **Build the optimal risky portfolio** (tangency) using price history **up to** that date.
    3. **Track the portfolio’s performance** from that date to today across multiple horizons.
    """)

    # Date picker
    default_start = datetime.now() - timedelta(days=365)
    start_date = st.date_input("Backtest start date", default_start, max_value=datetime.now() - timedelta(days=1))

    # Option to choose equal-weight or optimal
    portfolio_type = st.radio("Portfolio weighting", ["Equal‑weighted", "Optimal (tangency)"])

    if st.button("Run Historical Backtest"):
        # 1. Load fundamentals as of start_date
        cutoff = pd.to_datetime(start_date)
        with st.spinner(f"Loading financial data as of {cutoff.strftime('%Y-%m-%d')}..."):
            hist_data = load_all_data_as_of(cutoff, tickers)
        if hist_data.empty:
            st.error("No fundamental data available for that date. Try a later date.")
            st.stop()

        # 2. Apply screens
        passing = apply_screen(hist_data, min_years, pass_ratio, gross_margin_min, op_margin_min,
                               roe_min, de_max, pe_max, pb_max, pfcf_max, market_cap_min)
        if passing.empty:
            st.warning("No stocks passed the screen on that date. Adjust parameters or choose another date.")
            st.stop()

        st.success(f"{len(passing)} stocks passed the screen as of {cutoff.strftime('%Y-%m-%d')}.")
        st.dataframe(passing[['Ticker', 'Name', 'Avg ROE (%)', 'Avg Gross Margin (%)', 'Avg Op Margin (%)']])

        # 3. Determine weights
        ticker_list = passing['Ticker'].tolist()
        if portfolio_type == "Equal‑weighted":
            weights = np.ones(len(ticker_list)) / len(ticker_list)
            st.info(f"Using equal weights across {len(ticker_list)} stocks.")
        else:  # Optimal
            with st.spinner(f"Optimizing portfolio using price data up to {cutoff.strftime('%Y-%m-%d')}..."):
                opt_weights, ret_ann, vol_ann, sharpe, _ = optimize_portfolio_with_history(ticker_list, cutoff, risk_free_rate, lookback_years)
                if opt_weights is None:
                    st.error("Optimization failed. Falling back to equal weights.")
                    weights = np.ones(len(ticker_list)) / len(ticker_list)
                else:
                    weights = opt_weights
                    # Show weights
                    weight_df = pd.DataFrame({
                        'Ticker': ticker_list,
                        'Weight (%)': weights * 100
                    })
                    name_map = dict(zip(passing['Ticker'], passing['Name']))
                    weight_df['Name'] = weight_df['Ticker'].map(name_map)
                    weight_df = weight_df[weight_df['Weight (%)'] > 0.01].sort_values('Weight (%)', ascending=False)
                    st.subheader("Optimal Weights (as of backtest date)")
                    st.dataframe(weight_df.style.format({'Weight (%)': '{:.2f}'}), use_container_width=True)

        # 4. Track performance from start_date to today
        with st.spinner("Tracking performance..."):
            port_cum, bench_cum = track_portfolio_performance(ticker_list, weights, cutoff, datetime.now())
            if port_cum is None or port_cum.empty:
                st.error("No price data available from that date. Try a more recent start.")
                st.stop()

            # Compute multi-horizon returns
            results_df = compute_multi_horizon_returns(port_cum, bench_cum, cutoff, datetime.now())

        # Display results
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

        # Cumulative chart
        st.subheader("Cumulative Performance")
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=port_cum.index,
            y=port_cum,
            mode='lines',
            name='Portfolio',
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
                    name='S&P 500',
                    line=dict(color='red', dash='dash')
                ))
        # Add vertical lines for each horizon end
        horizons = {"1M":30, "3M":90, "6M":180, "1Y":365, "2Y":730, "5Y":1825}
        for label, days in horizons.items():
            h_end = cutoff + timedelta(days=days)
            if h_end > datetime.now():
                h_end = datetime.now()
            # find nearest date in port_cum index
            nearest = port_cum.index[port_cum.index <= h_end]
            if len(nearest) > 0:
                nearest_date = nearest[-1]
                fig_cum.add_vline(x=nearest_date, line_width=1, line_dash="dash", line_color="grey",
                                  annotation_text=label, annotation_position="top")
        fig_cum.update_layout(
            title=f'Cumulative Return from {cutoff.strftime("%Y-%m-%d")} to Today',
            xaxis_title='Date',
            yaxis_title='Cumulative Return',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5)
        )
        st.plotly_chart(fig_cum, use_container_width=True)

        # List of stocks used
        st.subheader("Stocks Included in the Backtest")
        st.write(f"All {len(ticker_list)} passing stocks were used with {'equal' if portfolio_type == 'Equal‑weighted' else 'optimal'} weights.")
        st.dataframe(pd.DataFrame({'Ticker': ticker_list, 'Name': [dict(zip(passing['Ticker'], passing['Name'])).get(t, t) for t in ticker_list]}))

        # Interpretation
        latest_valid = results_df[results_df['Portfolio Return (%)'].notna()].iloc[-1] if not results_df[results_df['Portfolio Return (%)'].notna()].empty else None
        if latest_valid is not None:
            st.subheader("📝 Interpretation")
            st.markdown(f"""
            On **{cutoff.strftime('%Y-%m-%d')}**, the screen selected {len(ticker_list)} stocks.  
            The {portfolio_type.lower()} portfolio has achieved a cumulative return of **{latest_valid['Portfolio Return (%)']:.2f}%** over the **{latest_valid['Horizon']}** horizon.
            The table above shows performance across all shorter horizons.
            """)
        else:
            st.info("Not enough data to compute returns for any horizon. Try a later start date.")
