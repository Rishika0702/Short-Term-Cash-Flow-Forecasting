import io
import pandas as pd
import plotly.express as px
import streamlit as st

from cashflow_model import (
    load_and_prepare_cashflow,
    prepare_prophet_data,
    train_prophet_model,
    make_forecast,
    detect_cash_shortages,
)

# -------------------------------------------------
# Streamlit Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="SmartCash Cash Flow Forecasting",
    layout="wide",
)

st.title("SmartCash Short-Term Cash Flow Forecasting")

st.markdown(
    """
This tool helps small businesses **forecast short-term cash flow** by consolidating data from:
- Sales (inflows)  
- Expenses (outflows)  
- Inventory purchases (outflows)

Upload your CSVs or use sample data, set starting bank balance, and see **7–30 day forecasts** plus **shortage alerts**.
"""
)


# -------------------------------------------------
# Sidebar Inputs
# -------------------------------------------------
starting_balance = st.sidebar.number_input(
    "Starting Bank Balance (current)",
    min_value=-1_000_000.0,
    max_value=1_000_000_000.0,
    value=50_000.0,
    step=1_000.0,
)

forecast_days = st.sidebar.slider(
    "Forecast horizon (days)", min_value=7, max_value=60, value=30, step=1
)

shortage_threshold = st.sidebar.number_input(
    "Cash shortage threshold (e.g., 0 or 10000)",
    value=0.0,
    step=1000.0,
)

st.sidebar.markdown("---")
st.sidebar.write("**Required CSV format:**")
st.sidebar.code("date,amount", language="text")


# -------------------------------------------------
# Sample Data
# -------------------------------------------------
def load_sample_data():
    sales_sample = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "amount": [12000, 15000, 13000],
    })

    expenses_sample = pd.DataFrame({
        "date": ["2024-01-01", "2024-01-02", "2024-01-03"],
        "amount": [-5000, -6000, -5500],
    })

    inventory_sample = pd.DataFrame({
        "date": ["2024-01-02", "2024-01-03"],
        "amount": [-8000, -9000],
    })

    return sales_sample, expenses_sample, inventory_sample


# -------------------------------------------------
# Upload Section
# -------------------------------------------------
st.subheader("1️⃣ Upload Data")

use_sample = st.checkbox("Use sample data instead of uploading files")

if use_sample:
    st.info("Using built-in sample data.")
    sales_df, expenses_df, inventory_df = load_sample_data()

else:
    col1, col2, col3 = st.columns(3)

    with col1:
        sales_file = st.file_uploader("Sales (inflows)", type=["csv"], key="sales")

    with col2:
        expenses_file = st.file_uploader("Expenses (outflows)", type=["csv"], key="expenses")

    with col3:
        inventory_file = st.file_uploader("Inventory (outflows)", type=["csv"], key="inventory")

    if not (sales_file and expenses_file and inventory_file):
        st.info("Upload all three CSV files or enable sample data.")
        st.stop()

    # Helper to load uploaded CSVs
    def read_csv(file):
        return pd.read_csv(io.StringIO(file.getvalue().decode("utf-8")))

    try:
        sales_df = read_csv(sales_file)
        expenses_df = read_csv(expenses_file)
        inventory_df = read_csv(inventory_file)
    except Exception as e:
        st.error(f"Error reading CSVs: {e}")
        st.stop()


# -------------------------------------------------
# Preview Uploaded or Sample Data
# -------------------------------------------------
with st.expander("🔍 Preview data"):
    st.write("**Sales (first 5 rows)**")
    st.dataframe(sales_df.head())

    st.write("**Expenses (first 5 rows)**")
    st.dataframe(expenses_df.head())

    st.write("**Inventory (first 5 rows)**")
    st.dataframe(inventory_df.head())


# -------------------------------------------------
# Step 2 — Build Daily Cash Flow Series
# -------------------------------------------------
st.subheader("2️⃣ Build Daily Cash Flow Series")

try:
    cashflow_df = load_and_prepare_cashflow(sales_df, expenses_df, inventory_df)
except Exception as e:
    st.error(f"Error constructing cashflow: {e}")
    st.stop()

st.write("Daily aggregated cashflow:")
st.dataframe(cashflow_df.tail())

hist_flow_fig = px.line(
    cashflow_df,
    x="date",
    y=["inflow", "outflow", "net_flow"],
    title="Historical Daily Inflow / Outflow / Net Flow",
)
st.plotly_chart(hist_flow_fig, use_container_width=True)


# -------------------------------------------------
# Step 3 — Train Model
# -------------------------------------------------
st.subheader("3️⃣ Train Forecasting Model")

try:
    prophet_df = prepare_prophet_data(cashflow_df)
    model = train_prophet_model(prophet_df)
except Exception as e:
    st.error(f"Error training Prophet model: {e}")
    st.stop()

st.success("Model trained successfully!")


# -------------------------------------------------
# Step 4 — Forecast
# -------------------------------------------------
st.subheader("4️⃣ Forecast Future Cash Flow")

try:
    forecast_df = make_forecast(
        model=model,
        periods=forecast_days,
        freq="D",
        starting_balance=starting_balance,
        history_df=prophet_df,
    )
except Exception as e:
    st.error(f"Error generating forecast: {e}")
    st.stop()

last_history_date = cashflow_df["date"].max()
forecast_df["is_future"] = forecast_df["ds"] > last_history_date

st.write("Forecast (last 10 rows):")
st.dataframe(
    forecast_df[
        ["ds", "predicted_net_flow", "predicted_balance", "is_future"]
    ].tail(10)
)


flow_forecast_fig = px.line(
    forecast_df,
    x="ds",
    y="predicted_net_flow",
    color="is_future",
    title="Predicted Daily Net Cash Flow (History vs Forecast)",
)
st.plotly_chart(flow_forecast_fig, use_container_width=True)

balance_fig = px.line(
    forecast_df,
    x="ds",
    y="predicted_balance",
    title="Predicted Bank Balance",
)
balance_fig.add_hline(
    y=shortage_threshold,
    line_dash="dash",
    annotation_text=f"Threshold ({shortage_threshold:,.0f})",
    annotation_position="top left",
)
st.plotly_chart(balance_fig, use_container_width=True)


# -------------------------------------------------
# Step 5 — Cash Shortages
# -------------------------------------------------
st.subheader("5️⃣ Cash Shortage Alerts")

shortages = detect_cash_shortages(forecast_df, threshold=shortage_threshold)

if not shortages:
    st.success("✅ No predicted cash shortage within the forecast horizon.")
else:
    st.error("⚠️ Cash shortage(s) predicted:")
    for s in shortages:
        st.write(
            f"- Date: **{s['date']}**, predicted balance: **₹{s['predicted_balance']:,.0f}**"
        )


# -------------------------------------------------
# Step 6 — Download CSV
# -------------------------------------------------
st.subheader("6️⃣ Download Forecast Data")

csv_buf = io.StringIO()
forecast_export = forecast_df[
    ["ds", "predicted_net_flow", "predicted_balance", "is_future"]
].copy()
forecast_export.to_csv(csv_buf, index=False)

st.download_button(
    label="Download forecast as CSV",
    data=csv_buf.getvalue(),
    file_name="cashflow_forecast.csv",
    mime="text/csv",
)
