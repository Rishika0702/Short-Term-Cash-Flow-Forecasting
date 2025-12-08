from datetime import datetime, timedelta
import pandas as pd
import numpy as np

def _ensure_date(df, date_col='date'):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df

def aggregate_by_date(df, date_col='date', amount_col='amount'):
    df = _ensure_date(df, date_col)
    agg = df.groupby(df[date_col].dt.date)[amount_col].sum().rename('amount')
    idx = pd.date_range(start=agg.index.min(), end=agg.index.max())
    agg = agg.reindex(idx.date, fill_value=0)
    agg.index = pd.to_datetime(agg.index)
    return agg

def generate_forecast(sales_df=None, accounting_df=None, inventory_df=None, days=30):
    # Accept dataframes with at least ['date','amount'] columns
    # Positive amounts are cash in (sales), negative are cash out (expenses/purchases)
    parts = []
    if sales_df is not None:
        sales = aggregate_by_date(sales_df)
        parts.append(sales)
    if accounting_df is not None:
        acct = aggregate_by_date(accounting_df)
        parts.append(acct)
    if inventory_df is not None:
        inv = aggregate_by_date(inventory_df)
        parts.append(inv)

    if not parts:
        raise ValueError('At least one dataframe is required')

    # Align dates
    combined = pd.concat(parts, axis=1).fillna(0)
    combined['net'] = combined.sum(axis=1)

    # Historical net series
    hist = combined['net']
    last_date = hist.index.max()

    # Extract trend using linear fit (days -> net)
    x = np.arange(len(hist))
    if len(x) < 2:
        slope = 0.0
        intercept = float(hist.iloc[-1]) if len(hist) else 0.0
    else:
        slope, intercept = np.polyfit(x, hist.values, 1)

    # Use last-N average for level
    N = min(30, len(hist))
    level = hist[-N:].mean() if N > 0 else 0.0

    # Forecast next days
    forecasts = []
    start_idx = len(hist)
    running_balance_start = 0.0
    # If accounting has a balance column, try to use it. Else start at 0.
    # We'll compute running balance from history's sum
    running_balance_start = hist.cumsum().iloc[-1] if len(hist) else 0.0

    for i in range(days):
        di = start_idx + i
        trend_component = slope * di + intercept
        # blend level and trend to get a forecast net
        forecast_net = 0.6 * level + 0.4 * trend_component
        date = last_date + timedelta(days=i+1)
        forecasts.append((pd.to_datetime(date), float(forecast_net)))

    forecast_df = pd.DataFrame(forecasts, columns=['date', 'net'])
    forecast_df['running_balance'] = running_balance_start + forecast_df['net'].cumsum()
    return {'history': combined[['net']].rename(columns={'net':'net'}), 'forecast': forecast_df}
# simulate_data.py

import pandas as pd
import numpy as np

rng = np.random.default_rng(42)


dates = pd.date_range(start="2024-01-01", end="2024-06-30", freq="D")


sales = pd.DataFrame({
    "date": dates,
    "amount": rng.normal(loc=8000, scale=2000, size=len(dates)).clip(min=1000),
})


expenses = pd.DataFrame({
    "date": dates,
    "amount": rng.normal(loc=5000, scale=1500, size=len(dates)).clip(min=500),
})


inv_dates = dates[::7] 
inventory = pd.DataFrame({
    "date": inv_dates,
    "amount": rng.normal(loc=7000, scale=2500, size=len(inv_dates)).clip(min=1000),
})

sales.to_csv("sales.csv", index=False)
expenses.to_csv("expenses.csv", index=False)
inventory.to_csv("inventory.csv", index=False)

print("Generated sales.csv, expenses.csv, inventory.csv")
