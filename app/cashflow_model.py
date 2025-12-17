import pandas as pd
from prophet import Prophet


def load_and_prepare_cashflow(
    sales_df: pd.DataFrame,
    expenses_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Takes raw sales, expenses, and inventory DataFrames and returns a
    daily-level cashflow DataFrame with columns:
    date, inflow, outflow, net_flow
    """

    for df in (sales_df, expenses_df, inventory_df):
        if "date" not in df.columns or "amount" not in df.columns:
            raise ValueError("Each DataFrame must have 'date' and 'amount' columns.")

    sales = sales_df.copy()
    expenses = expenses_df.copy()
    inv = inventory_df.copy()

    sales["date"] = pd.to_datetime(sales["date"])
    expenses["date"] = pd.to_datetime(expenses["date"])
    inv["date"] = pd.to_datetime(inv["date"])

    expenses["amount"] = expenses["amount"].abs()
    inv["amount"] = inv["amount"].abs()


  
    inflow_df = sales[["date", "amount"]].rename(columns={"amount": "inflow"})
    inflow_df["outflow"] = 0.0


    outflow_df = pd.concat([expenses, inv], ignore_index=True)
    outflow_df = outflow_df[["date", "amount"]].rename(columns={"amount": "outflow"})
    outflow_df["inflow"] = 0.0

    cashflow = pd.concat([inflow_df, outflow_df], ignore_index=True)
    cashflow = (
        cashflow.groupby("date")[["inflow", "outflow"]]
        .sum()
        .sort_index()
        .reset_index()
    )
    cashflow["net_flow"] = cashflow["inflow"] - cashflow["outflow"]

    return cashflow


def prepare_prophet_data(cashflow_df: pd.DataFrame) -> pd.DataFrame:
    """
    Format cashflow DataFrame for Prophet.
    Expects column 'date' and 'net_flow'.
    Returns DataFrame with columns: ds, y
    """
    if "date" not in cashflow_df.columns or "net_flow" not in cashflow_df.columns:
        raise ValueError("cashflow_df must contain 'date' and 'net_flow' columns.")

    df = cashflow_df[["date", "net_flow"]].copy()
    df = df.rename(columns={"date": "ds", "net_flow": "y"})
    return df


def train_prophet_model(df: pd.DataFrame) -> Prophet:
    """
    Train a Prophet model on daily net cash flow.
    df must have columns: ds, y
    """
    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
    )
    model.fit(df)
    return model


def make_forecast(
    model: Prophet,
    periods: int = 30,
    freq: str = "D",
    starting_balance: float = 0.0,
    history_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Generate a forecast for 'periods' days into the future.
    Adds predicted_balance based on starting_balance + cumulative yhat.

    model: trained Prophet model
    history_df: original df with columns [ds, y]; needed to align history + forecast.
    """

    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)

    forecast["predicted_net_flow"] = forecast["yhat"].copy()

  
    if history_df is not None and not history_df.empty:
     
        history_df = history_df.sort_values("ds")
        historical_cum_net = history_df["y"].cumsum().iloc[-1]
        forecast["cumulative_net_flow"] = historical_cum_net + forecast[
            "predicted_net_flow"
        ].cumsum()
    else:
        forecast["cumulative_net_flow"] = forecast["predicted_net_flow"].cumsum()

    forecast["predicted_balance"] = (
        starting_balance + forecast["cumulative_net_flow"]
    )

    return forecast


def detect_cash_shortages(
    forecast_df: pd.DataFrame,
    threshold: float = 0.0,
) -> list[dict]:
    """
    Detect days where predicted_balance falls below threshold.
    Returns a list of dicts with 'date' and 'predicted_balance'.
    """
    if "predicted_balance" not in forecast_df.columns or "ds" not in forecast_df.columns:
        raise ValueError("forecast_df must contain 'ds' and 'predicted_balance'.")

    shortages = forecast_df[forecast_df["predicted_balance"] < threshold]
    results = []
    for _, row in shortages.iterrows():
        results.append(
            {
                "date": row["ds"].date(),
                "predicted_balance": float(row["predicted_balance"]),
            }
        )
    return results

