import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go 
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -------------------------------
# Configuration / file locations
# -------------------------------
DATA_DIR = "./"  # put your CSVs next to app.py, or change this path

KPI_CSV         = os.path.join(DATA_DIR, "kpi_summary_from_daily.csv")  # city-level KPIs
DAILY_CSV       = os.path.join(DATA_DIR, "daily_metrics.csv")           # daily metrics per city

# Optional for venue views (raw order-level sales). If you have them, set the filenames:
CITY1_SALES_CSV = os.path.join(DATA_DIR, "city1_sales")  # or your file name
CITY2_SALES_CSV = os.path.join(DATA_DIR, "city2_sales")  # or your file name

# -------------------------------
# Utilities
# -------------------------------
@st.cache_data
def load_csv(path, parse_dates=None):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=parse_dates)

def nice_eur(x):
    try:
        return f"€{x:,.0f}".replace(",", " ")  # thin space
    except Exception:
        return x

def guard_df(df, msg):
    if df is None or df.empty:
        st.info(msg)
        st.stop()

# -------------------------------
# Load data
# -------------------------------

def load_sales_raw(path, city_label):
    """Load a raw order-level sales CSV and normalize timestamp -> 'date'."""
    if not os.path.exists(path):
        st.warning(f"{city_label}: file not found at {path}")
        return None

    df = pd.read_csv(path, low_memory=False)
    # Required columns present?
    must = {"TIME_DELIVERED_UTC","USER_ID","VENUE_NAME","TOTAL_DELIVERY_TIME"}
    missing = must - set(df.columns)
    if missing:
        st.error(f"{city_label}: missing columns {missing}")
        return None

    # Parse timestamp and add normalized 'date'
    df["TIME_DELIVERED_UTC"] = pd.to_datetime(df["TIME_DELIVERED_UTC"], errors="coerce")
    df["date"] = df["TIME_DELIVERED_UTC"].dt.normalize()

    # (Optional) clean types
    if "DELIVERY_METHOD" in df.columns:
        df["DELIVERY_METHOD"] = df["DELIVERY_METHOD"].astype(str)

    if "PREORDER" in df.columns:
        # keep as string; the plotting helpers will treat yes/true as preorders to exclude
        df["PREORDER"] = df["PREORDER"].astype(str)

    return df


kpi_df   = load_csv(KPI_CSV)
daily_df = load_csv(DAILY_CSV, parse_dates=["date"])

# Optional raw sales (for venue tabs)
city1_sales_raw = load_sales_raw(CITY1_SALES_CSV, "City 1")
city2_sales_raw = load_sales_raw(CITY2_SALES_CSV, "City 2")

# -------------------------------
# App layout
# -------------------------------
st.set_page_config(page_title="Wolt Ops Case Dashboard", layout="wide")
st.title("Wolt KPI & Operations Dashboard")

# Sidebar filters
st.sidebar.header("Filters")

if daily_df is not None and "city" in daily_df.columns:
    cities = sorted(daily_df["city"].dropna().unique().tolist())
else:
    cities = ["City 1", "City 2"]

city_sel = st.sidebar.multiselect("City", options=cities, default=cities)

if daily_df is not None and "date" in daily_df.columns:
    dmin, dmax = daily_df["date"].min(), daily_df["date"].max()
    date_sel = st.sidebar.date_input("Date range", value=(dmin, dmax), min_value=dmin, max_value=dmax)
else:
    date_sel = None

st.sidebar.markdown("---")
show_annotations = st.sidebar.checkbox("Show value labels", value=True)

# Filter data by sidebar
def filter_daily(df):
    if df is None:
        return None
    out = df.copy()
    if city_sel:
        out = out[out["city"].isin(city_sel)]
    if date_sel and isinstance(date_sel, tuple) and len(date_sel) == 2:
        start, end = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
        out = out[(out["date"] >= start) & (out["date"] <= end)]
    return out

daily = filter_daily(daily_df)

def plot_courier_balance(df, city):
    d = df.sort_values("date").copy()

    # Derived fields if missing
    if "tot_daily_cost" not in d.columns and \
       {"TASK_COST","GUARANTEE_COST","OTHER_COST"}.issubset(d.columns):
        d["tot_daily_cost"] = d[["TASK_COST","GUARANTEE_COST","OTHER_COST"]].sum(axis=1)

    if "CPO" not in d.columns:
        d["CPO"] = np.where(d["orders"] > 0, d["tot_daily_cost"] / d["orders"], np.nan)

    if "Guarantee_pct" not in d.columns and \
       {"GUARANTEE_COST","tot_daily_cost"}.issubset(d.columns):
        d["Guarantee_pct"] = np.where(d["tot_daily_cost"] > 0,
                                      d["GUARANTEE_COST"] / d["tot_daily_cost"] * 100, np.nan)

    # Use total_revenue if present, otherwise revenue
    rev_col = "total_revenue" if "total_revenue" in d.columns else "revenue"

    fig = make_subplots(specs=[[{"secondary_y": True}]])  # gives yaxis and yaxis2

    # Left axis: bars
    fig.add_bar(x=d["date"], y=d[rev_col], name="Revenue (€)", marker_color="seagreen")
    fig.add_bar(x=d["date"], y=d["tot_daily_cost"], name="Courier Costs (€)", marker_color="crimson")

    # Right axis #1 (yaxis2): Orders
    fig.add_scatter(x=d["date"], y=d["orders"], mode="lines+markers",
                    name="Orders", line=dict(color="steelblue", width=2),
                    yaxis="y2")

    # Extra right axis (yaxis3): CPO + Guarantee %
    fig.add_scatter(x=d["date"], y=d["CPO"], mode="lines+markers",
                    name="CPO (€)", line=dict(color="orange", width=2),
                    yaxis="y3")
    fig.add_scatter(x=d["date"], y=d["Guarantee_pct"], mode="lines+markers",
                    name="Guarantee %", line=dict(color="goldenrod", dash="dot"),
                    yaxis="y3")

    # Layout: keep positions within [0, 1]
    fig.update_layout(
        title=f"{city} – Revenue vs Courier Costs, Orders, CPO & Guarantee %",
        barmode="group",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Revenue/Costs (€)"),
        # right axis for Orders
        yaxis2=dict(title="Orders", overlaying="y", side="right",
                    position=1.00, showgrid=False, zeroline=False),
        # second right axis for CPO/Guarantee%, slightly left of the first
        yaxis3=dict(title="CPO (€) / Guarantee %", overlaying="y", side="right",
                    position=0.97, showgrid=False, zeroline=False),
        legend=dict(orientation="h", y=1.15, x=0),
        margin=dict(t=90, r=20, l=10, b=40),
        hovermode="x unified"
    )
    return fig

def _ensure_date_col(df):
    """Return a datetime series to derive 'hour' from."""
    for c in ["TIME_DELIVERED_UTC", "datetime", "date"]:
        if c in df.columns:
            s = pd.to_datetime(df[c], errors="coerce")
            if s.notna().any():
                return s
    raise ValueError("No timestamp column found (TIME_DELIVERED_UTC/datetime/date).")

def prep_hourly(sales_raw: pd.DataFrame):
    """
    From raw order-level sales:
      - keep homedelivery & non-preorder
      - compute hour 0..23
      - return DF with columns: hour, orders, avg_delivery_time
      - ensure all hours exist (fill zeros / NaNs)
    """
    if sales_raw is None or sales_raw.empty:
        return None

    df = sales_raw.copy()

    # Filter to homedelivery
    if "DELIVERY_METHOD" in df.columns:
        df = df[df["DELIVERY_METHOD"].astype(str).str.lower().eq("homedelivery")]

    # Drop preorders
    if "PREORDER" in df.columns:
        pre = df["PREORDER"].astype(str).str.lower().isin(["true", "yes", "1"])
        df = df[~pre]

    # Timestamp -> hour
    ts = _ensure_date_col(df)
    df["hour"] = ts.dt.hour

    # Aggregate
    g = (df.groupby("hour")
            .agg(orders=("USER_ID", "count"),
                 avg_delivery_time=("TOTAL_DELIVERY_TIME", "mean"))
            .reset_index())

    # Ensure hours 0..23 exist
    hours = pd.DataFrame({"hour": np.arange(24)})
    g = hours.merge(g, on="hour", how="left")
    g["orders"] = g["orders"].fillna(0).astype(int)
    # avg_delivery_time left as NaN when no orders in that hour

    return g

def plot_hourly_orders_vs_avg(g: pd.DataFrame, city_name: str, height=420):
    """Bar (orders) + line (avg delivery time) with secondary y-axis."""
    fig = go.Figure()

    fig.add_bar(x=g["hour"], y=g["orders"], name="Orders/hour",
                marker_color="steelblue", opacity=0.75)

    fig.add_trace(go.Scatter(
        x=g["hour"], y=g["avg_delivery_time"],
        name="Avg delivery time (min)",
        mode="lines+markers", line=dict(color="crimson", width=3),
        yaxis="y2"
    ))

    fig.update_layout(
        title=f"{city_name} – Orders & Avg Delivery Time by Hour",
        height=height,
        xaxis=dict(title="Hour of day", dtick=1),
        yaxis=dict(title="Orders", rangemode="tozero"),
        yaxis2=dict(title="Avg delivery time (min)", overlaying="y", side="right"),
        margin=dict(t=50, r=40, l=10, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

def plot_orders_vs_cost_one_city(daily_city, city_name, height=420):
    """
    Scatter of daily Orders vs. total courier cost with OLS trendline.
    Shows the fitted equation and R^2 in the subtitle.
    """
    if daily_city is None or daily_city.empty:
        return go.Figure()

    # Build scatter with trendline
    fig = px.scatter(
        daily_city.sort_values("orders"),
        x="orders",
        y="tot_daily_cost",
        trendline="ols",
        opacity=0.8,
        title=f"{city_name} – Orders vs Courier Cost (daily)",
        labels={"orders": "Orders per day", "tot_daily_cost": "Total courier cost (€)"},
        color_discrete_sequence=["steelblue"],
        height=height,
    )

    # Extract OLS results to annotate equation
    try:
        results = px.get_trendline_results(fig)
        if len(results) > 0:
            model = results.iloc[0]["px_fit_results"]
            params = model.params  # Intercept & slope
            intercept = params[0]
            slope = params[1]
            # R^2
            r2 = model.rsquared
            fig.update_layout(
                title={
                    "text": f"{city_name} – Orders vs Courier Cost (daily)<br>"
                            f"<sup>Trendline: cost ≈ {slope:.2f} × orders + {intercept:.0f}   (R²={r2:.2f})</sup>"
                }
            )
    except Exception:
        pass

    fig.update_traces(marker=dict(size=8))
    fig.update_layout(margin=dict(t=70, r=20, l=10, b=40))
    return fig

def plot_orders_vs_cost_both_cities(daily_all, city_labels=("City 1","City 2"), height=520):
    """
    Combined scatter: Orders/day vs Total courier cost for two cities + separate trendlines.
    daily_all must have columns: city, orders, tot_daily_cost
    """

    need = {"city","orders","tot_daily_cost"}
    if daily_all is None or daily_all.empty or not need.issubset(daily_all.columns):
        return go.Figure()

    # Ensure cost column exists
    d = daily_all.copy()
    if "tot_daily_cost" not in d.columns:
        if {"TASK_COST","GUARANTEE_COST","OTHER_COST"}.issubset(d.columns):
            d["tot_daily_cost"] = d["TASK_COST"] + d["GUARANTEE_COST"] + d["OTHER_COST"]
        else:
            raise ValueError("tot_daily_cost missing and cannot be computed.")

    # Keep just two cities (first two alphabetically) unless the names are passed
    cits = sorted(d["city"].dropna().unique().tolist())
    if len(cits) < 2:
        return go.Figure()

    c1, c2 = (city_labels if set(city_labels).issubset(set(cits)) else cits[:2])

    d1 = d[d["city"] == c1].sort_values("orders")
    d2 = d[d["city"] == c2].sort_values("orders")

    fig = go.Figure()

    # scatter points
    fig.add_scatter(x=d1["orders"], y=d1["tot_daily_cost"], mode="markers",
                    name=f"{c1} (data)", marker=dict(color="royalblue", size=8, opacity=0.9))
    fig.add_scatter(x=d2["orders"], y=d2["tot_daily_cost"], mode="markers",
                    name=f"{c2} (data)", marker=dict(color="darkorange", size=8, opacity=0.9))

    # trendline (numpy polyfit)
    for df, color, label in [(d1, "royalblue", c1), (d2, "darkorange", c2)]:
        if len(df) >= 2:
            m, b = np.polyfit(df["orders"], df["tot_daily_cost"], 1)
            xline = np.linspace(df["orders"].min(), df["orders"].max(), 50)
            yline = m * xline + b
            fig.add_scatter(x=xline, y=yline, mode="lines",
                            line=dict(color=color, width=3),
                            name=f"{label} trend (y={m:.2f}x+{b:.0f})")

    fig.update_layout(
        title=f"Daily Courier Cost vs Orders – {c1} vs {c2}",
        xaxis_title="Orders per day",
        yaxis_title="Total courier cost (€)",
        height=height,
        margin=dict(t=60, r=20, l=10, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
    )
    return fig

def plot_city_correlation_heatmap(daily_df, city_name, metrics=None, height=520):
    """
    Draw a lower-triangular correlation heatmap for a city's daily metrics.
    """
    if daily_df is None or daily_df.empty:
        return None

    d = daily_df.copy()

    # Derived fields if missing
    if "tot_daily_cost" not in d.columns and \
       {"TASK_COST","GUARANTEE_COST","OTHER_COST"}.issubset(d.columns):
        d["tot_daily_cost"] = d["TASK_COST"] + d["GUARANTEE_COST"] + d["OTHER_COST"]

    if "CPO" not in d.columns and "tot_daily_cost" in d.columns and "orders" in d.columns:
        d["CPO"] = np.where(d["orders"] > 0, d["tot_daily_cost"] / d["orders"], np.nan)

    if "Guarantee_pct" not in d.columns and \
       {"GUARANTEE_COST","tot_daily_cost"}.issubset(d.columns):
        d["Guarantee_pct"] = np.where(d["tot_daily_cost"] > 0,
                                      d["GUARANTEE_COST"] / d["tot_daily_cost"] * 100, np.nan)

    # Default metric set (matches your example)
    default_metrics = [
        "orders",
        "avg_delivery_time",
        "TASK_COST",
        "GUARANTEE_COST",
        "CPO",
        "contribution_margin",
        "Guarantee_pct",
    ]
    metrics = metrics or default_metrics
    use_cols = [c for c in metrics if c in d.columns]

    dc = d[d["city"] == city_name][use_cols].dropna(how="any")
    if dc.empty or len(use_cols) < 2:
        return None

    corr = dc.corr()

    # Mask upper triangle (show lower triangle like your figure)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True

    # Plot
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title(f"{city_name} Correlation Matrix", fontsize=18, pad=12)
    fig.set_figheight(height / 100.0)
    fig.tight_layout()
    return fig

import statsmodels.api as sm

def run_city_regression(daily_df, city_name):
    """
    Run OLS regression: avg_delivery_time ~ TASK_COST + GUARANTEE_COST + OTHER_COST + orders
    """
    d = daily_df[daily_df["city"] == city_name].dropna(subset=["avg_delivery_time","TASK_COST","GUARANTEE_COST","OTHER_COST","orders"])
    if d.empty:
        return None, None

    X = d[["TASK_COST","GUARANTEE_COST","OTHER_COST","orders"]]
    X = sm.add_constant(X)  # add intercept
    y = d["avg_delivery_time"]

    model = sm.OLS(y, X).fit()
    return model, d

import numpy as np
import plotly.graph_objects as go

def plot_city_correlation_heatmap(daily, city, metrics, height=520, triangle="lower"):
    df = daily[daily["city"] == city].copy()
    if df.empty:
        return go.Figure()

    cols = [c for c in metrics if c in df.columns]
    corr = df[cols].corr()

    # mask one triangle
    n = corr.shape[0]
    if triangle == "lower":
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)   # mask above diagonal
    else:
        mask = np.tril(np.ones((n, n), dtype=bool), k=-1)  # mask below diagonal

    z = corr.values.astype(float).copy()
    z[mask] = np.nan                                       # blank masked cells

    # text shown inside cells (only where not masked)
    txt = np.where(mask, "", np.round(corr.values, 2).astype(str))

    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=corr.columns,
            y=corr.index,
            zmin=-1, zmax=1,
            colorscale="RdBu",
            reversescale=True,
            colorbar=dict(title="ρ"),
            # put the numbers *inside* each square
            text=txt,
            texttemplate="%{text}",
            textfont={"size": 11, "color": "white"},
            hovertemplate="%{y} × %{x}<br>ρ = %{z:.3f}<extra></extra>",
        )
    )

    fig.update_layout(
        title=dict(text=f"{city} – Correlation Heatmap", x=0, xanchor="left"),
        xaxis=dict(tickangle=45),
        yaxis=dict(autorange="reversed"),
        height=height,
        margin=dict(l=20, r=20, t=60, b=60),
    )
    return fig

def compute_new_customer_rates_from_sales(df: pd.DataFrame, city_label: str) -> dict | None:
    """
    Returns dict:
    {
        'city': <str>,
        'customer_level_rate': float,   # % of unique users that are first-timers
        'order_level_rate': float       # % of orders placed by first-timers
    }
    If raw sales df is missing/empty or has no delivered orders, returns None.
    """
    if df is None or df.empty:
        return None

    d = df.copy()

    # Keep delivered orders only
    if "STATUS" in d.columns:
        d = d[d["STATUS"].astype(str).str.lower() == "delivered"]

    # Exclude preorders if present
    if "PREORDER" in d.columns:
        d = d[d["PREORDER"].astype(str).str.lower().isin(["no", "false", "0", "nan"])]

    if len(d) == 0 or "USER_ID" not in d.columns or "FIRST_WOLT_PURCHASE" not in d.columns:
        return None

    # Customer-level share
    unique_customers = d["USER_ID"].nunique()
    unique_new = d.loc[d["FIRST_WOLT_PURCHASE"].astype(str).str.lower() == "yes", "USER_ID"].nunique()
    customer_level_rate = (unique_new / unique_customers * 100.0) if unique_customers > 0 else float("nan")

    # Order-level share
    new_orders = (d["FIRST_WOLT_PURCHASE"].astype(str).str.lower() == "yes").sum()
    total_orders = len(d)
    order_level_rate = (new_orders / total_orders * 100.0) if total_orders > 0 else float("nan")

    return {
        "city": city_label,
        "customer_level_rate": round(float(customer_level_rate), 1),
        "order_level_rate": round(float(order_level_rate), 1),
    }



# -------------------------------
# Tabs
# -------------------------------
tabs = st.tabs([
    "KPI Overview",
    "Daily Trends",
    "Courier Costs & Balance",
    "City 1 – Venues",
    "City 2 – Venues"
])


# ============================================================
# TAB 1 – KPI Overview (City 1 vs City 2)
# ============================================================
# ============================================================
# TAB 1 – KPI Overview
# ============================================================
with tabs[0]:
    st.subheader("KPI Summary (by City)")
    guard_df(kpi_df, "Upload kpi_summary_from_daily.csv to use this page.")

    # ---------- helpers ----------
    def norm_lookup(df, *cands):
        if df is None or df.empty:
            return None
        idx = {c.lower().replace(" ", "_"): c for c in df.columns}
        for c in cands:
            key = c.lower().replace(" ", "_")
            if key in idx:
                return df[idx[key]]
        return None

    def weighted_avg(vals: pd.Series, weights: pd.Series) -> float:
        vals = pd.to_numeric(vals, errors="coerce")
        weights = pd.to_numeric(weights, errors="coerce")
        w = weights.fillna(0)
        return float((vals * w).sum() / w.sum()) if w.sum() else float("nan")

    def compute_aov_from_sales(selected_cities: list[str]) -> float | None:
        total_goods, total_orders = 0.0, 0
        for c in selected_cities:
            if c == "City 1" and city1_sales_raw is not None:
                total_goods += pd.to_numeric(
                    city1_sales_raw["GOODS_TOTAL_TRANSACTION_VALUE_EUR"], errors="coerce"
                ).sum()
                total_orders += len(city1_sales_raw)
            elif c == "City 2" and city2_sales_raw is not None:
                total_goods += pd.to_numeric(
                    city2_sales_raw["GOODS_TOTAL_TRANSACTION_VALUE_EUR"], errors="coerce"
                ).sum()
                total_orders += len(city2_sales_raw)
        return float(total_goods / total_orders) if total_orders > 0 else None

    # ---------- filter ----------
    city_sel = st.multiselect("Filter cities", sorted(kpi_df["city"].unique()))
    scope = kpi_df[kpi_df["city"].isin(city_sel)] if city_sel else kpi_df.copy()
    sel_cities = scope["city"].tolist()

    # normalized series (as numeric where needed)
    s_city   = norm_lookup(scope, "city")
    s_orders = pd.to_numeric(norm_lookup(scope, "orders"), errors="coerce").fillna(0)
    s_adt    = pd.to_numeric(
        norm_lookup(scope, "avg_delivery_time", "avg_delivery_time_min", "avg_delivery_time_(min)"),
        errors="coerce"
    ).fillna(0)
    s_rev    = pd.to_numeric(norm_lookup(scope, "tot_revenue_eur", "total_revenue"), errors="coerce").fillna(0)
    s_cost   = pd.to_numeric(norm_lookup(scope, "tot_courier_cost_eur", "tot_daily_cost", "courier_cost"), errors="coerce").fillna(0)
    s_cm     = pd.to_numeric(norm_lookup(scope, "contribution_margin_eur", "contribution_margin"), errors="coerce").fillna(0)

    # ---------- KPIs ----------
    k_orders = int(s_orders.sum())
    k_adt    = weighted_avg(s_adt, s_orders)

    k_aov = compute_aov_from_sales(sel_cities)
    if k_aov is None:
        s_aov_col = norm_lookup(scope, "avg_order_value_eur", "avg_order_value", "average_order_value", "aov")
        if s_aov_col is not None:
            s_aov = pd.to_numeric(s_aov_col, errors="coerce")
            k_aov = float(s_aov.mean()) if s_aov.notna().any() else float("nan")
        else:
            k_aov = float("nan")

    k_rev  = float(s_rev.sum())
    k_cost = float(s_cost.sum())
    k_cm   = float(s_cm.sum())

    cols = st.columns(6)
    cols[0].metric("Orders", f"{k_orders:,}")
    cols[1].metric("Avg Delivery Time (min)", f"{k_adt:.1f}" if pd.notna(k_adt) else "—")
    cols[2].metric("Avg Order Value (€)", f"€{k_aov:,.1f}" if pd.notna(k_aov) else "—")
    cols[3].metric("Total Revenue (€)", nice_eur(k_rev))
    cols[4].metric("Courier Cost (€)", nice_eur(k_cost))
    cols[5].metric("Contribution Margin (€)", nice_eur(k_cm))

    st.markdown("")

# --- TWO ROWS of mini charts ------------------------------------------------
    row1c1, row1c2 = st.columns(2)
    row2c1, row2c2 = st.columns(2)

# 1) Revenue vs Courier Cost (by city)
with row1c1:
    fig1 = go.Figure()
    fig1.add_bar(name="Revenue", x=s_city, y=s_rev, marker_color="seagreen")
    fig1.add_bar(name="Courier Cost", x=s_city, y=s_cost, marker_color="tomato")
    fig1.update_layout(
        barmode="group",
        title="Revenue vs Courier Cost",
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        xaxis_title=None,
        yaxis_title="€",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.1)
    )
    st.plotly_chart(fig1, use_container_width=True)

# 2) Orders vs Avg Delivery Time (by city)
with row1c2:
    fig2 = go.Figure()
    fig2.add_bar(name="Orders", x=s_city, y=s_orders, marker_color="skyblue", yaxis="y1")
    fig2.add_scatter(name="Avg delivery time (min)", x=s_city, y=s_adt,
                     mode="lines+markers", marker=dict(color="crimson"), yaxis="y2")
    fig2.update_layout(
        title="Orders vs Avg Delivery Time",
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        xaxis_title=None,
        yaxis=dict(title="Orders"),
        yaxis2=dict(title="Min", overlaying="y", side="right"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.1)
    )
    st.plotly_chart(fig2, use_container_width=True)

# 3) Contribution Margin (by city)
with row2c1:
    cm_colors = ["#2e7d32" if v >= 0 else "#c62828" for v in s_cm]
    fig3 = go.Figure()
    fig3.add_bar(name="Contribution Margin", x=s_city, y=s_cm, marker_color=cm_colors)
    fig3.add_hline(y=0, line_width=1, line_dash="dot", line_color="gray")
    fig3.update_layout(
        title="Contribution Margin",
        margin=dict(l=10, r=10, t=40, b=10),
        height=300,
        xaxis_title=None,
        yaxis_title="€",
        showlegend=False
    )
    st.plotly_chart(fig3, use_container_width=True)

# 4) New Customer Share (Customer-level vs Order-level)
with row2c2:
    # Compute from raw sales if available
    shares = []
    if "City 1" in sel_cities and city1_sales_raw is not None:
        r1 = compute_new_customer_rates_from_sales(city1_sales_raw, "City 1")
        if r1: shares.append(r1)
    if "City 2" in sel_cities and city2_sales_raw is not None:
        r2 = compute_new_customer_rates_from_sales(city2_sales_raw, "City 2")
        if r2: shares.append(r2)

    if not shares:
        st.info("Upload raw sales (city1_sales.csv / city2_sales.csv) to show New Customer Share.")
    else:
        df_share = pd.DataFrame(shares)
        fig4 = go.Figure()
        fig4.add_bar(
            name="Customer-level new rate",
            x=df_share["city"], y=df_share["customer_level_rate"],
            marker_color="lightskyblue"
        )
        fig4.add_bar(
            name="Order-level new rate",
            x=df_share["city"], y=df_share["order_level_rate"],
            marker_color="royalblue"
        )
        fig4.update_layout(
            barmode="group",
            title="New Customer Share (%)",
            margin=dict(l=10, r=10, t=40, b=10),
            height=300,
            xaxis_title=None,
            yaxis_title="%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.1)
        )
        st.plotly_chart(fig4, use_container_width=True)




# ============================================================
# TAB 2 – Daily Trends
# ============================================================
with tabs[1]:
    st.subheader("Daily Trends")
    guard_df(daily, "Upload daily_metrics.csv to use this page.")

    colA, colB = st.columns(2)

    # Orders & Avg delivery time – dual axis line
    with colA:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=daily["date"], y=daily["orders"], name="Orders", marker_color="steelblue", opacity=0.65))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["avg_delivery_time"], name="Avg delivery time (min)",
                                 mode="lines+markers", line=dict(color="crimson", width=2), yaxis="y2"))
        fig.update_layout(
            title="Orders & Avg Delivery Time (daily)",
            yaxis=dict(title="Orders"),
            yaxis2=dict(title="Avg Delivery Time (min)", overlaying='y', side='right'),
            height=400, margin=dict(t=40, r=40, l=10, b=40)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Revenue vs Courier Cost – dual line
    with colB:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["total_revenue"], name="Revenue", mode="lines+markers", line=dict(color="seagreen")))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["tot_daily_cost"], name="Courier Cost", mode="lines+markers", line=dict(color="orange")))
        fig.update_layout(title="Revenue vs Courier Cost (daily)", height=400, margin=dict(t=40, r=20, l=10, b=40))
        st.plotly_chart(fig, use_container_width=True)

    colC, colD = st.columns(2)

    # Cost mix – stacked bars
    with colC:
        fig = go.Figure()
        fig.add_bar(x=daily["date"], y=daily["TASK_COST"], name="Task Cost")
        fig.add_bar(x=daily["date"], y=daily["GUARANTEE_COST"], name="Guarantee Cost")
        fig.add_bar(x=daily["date"], y=daily["OTHER_COST"], name="Other Cost")
        fig.update_layout(barmode="stack", title="Courier Cost Breakdown (daily)", height=400, margin=dict(t=40, r=20, l=10, b=40))
        st.plotly_chart(fig, use_container_width=True)

    # CPO & Contribution margin
    with colD:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["CPO"], name="CPO (€)", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=daily["date"], y=daily["contribution_margin"], name="Contribution Margin (€)", mode="lines+markers"))
        fig.update_layout(title="CPO & Contribution Margin (daily)", height=400, margin=dict(t=40, r=20, l=10, b=40))
        st.plotly_chart(fig, use_container_width=True)

    # New customer orders
    if "new_customer_orders" in daily.columns:
        fig = px.bar(daily, x="date", y="new_customer_orders", color="city", barmode="group",
                     title="New customer orders (daily)")
        st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 3 – Courier Costs & Balance
# ============================================================
with tabs[2]:
    st.subheader("Courier Balance")

    guard_df(daily, "Upload daily_metrics.csv to use this page.")

    # City selector inside the tab (single-select to match your slide flow)
    city_pick = st.selectbox("Choose city", options=sorted(daily["city"].unique()))
    d_city = daily[daily["city"] == city_pick]

    # 1) Main composite chart
    st.plotly_chart(
        plot_courier_balance(d_city, city_pick),
        use_container_width=True,
        key=f"balance_{city_pick}"
    )

    st.markdown("---")

    # 2) Orders vs Courier Cost – Both Cities (side-by-side)
    st.markdown("### Orders vs Courier Cost – Both Cities")
    present_cities = sorted(daily["city"].dropna().unique().tolist())
    if len(present_cities) == 0:
        st.info("No cities available for the current filters.")
    else:
        cols = st.columns(2)
        for i, city in enumerate(present_cities[:2]):
            d_c = daily[daily["city"] == city].copy()
            with cols[i]:
                st.plotly_chart(
                    plot_orders_vs_cost_one_city(d_c, city),
                    use_container_width=True,
                    key=f"orders_cost_{city}"
                )

    st.markdown("---")

    # 3) Guarantee share over time (for the selected city)
    d2 = d_city.copy()
    d2["Guarantee_pct"] = np.where(
        d2["tot_daily_cost"] > 0,
        d2["GUARANTEE_COST"] / d2["tot_daily_cost"] * 100,
        np.nan
    )
    fig = px.line(
        d2, x="date", y="Guarantee_pct",
        title=f"{city_pick} – Guarantee cost share (%)",
        markers=True, color_discrete_sequence=["goldenrod"]
    )
    st.plotly_chart(fig, use_container_width=True, key=f"guarantee_share_{city_pick}")

    st.markdown("---")

    # 4) Hourly Demand vs Speed (both cities)
    st.markdown("### Hourly Demand vs Speed")
    colH1, colH2 = st.columns(2)
    with colH1:
        if city1_sales_raw is None:
            st.info("Upload city1_sales.csv to show hourly view for City 1.")
        else:
            g1 = prep_hourly(city1_sales_raw)
            if g1 is not None:
                st.plotly_chart(
                    plot_hourly_orders_vs_avg(g1, "City 1"),
                    use_container_width=True,
                    key="hourly_city1"
                )
    with colH2:
        if city2_sales_raw is None:
            st.info("Upload city2_sales.csv to show hourly view for City 2.")
        else:
            g2 = prep_hourly(city2_sales_raw)
            if g2 is not None:
                st.plotly_chart(
                    plot_hourly_orders_vs_avg(g2, "City 2"),
                    use_container_width=True,
                    key="hourly_city2"
                )

    st.markdown("---")

    # 5) Daily Courier Cost vs Orders – Combined
    st.markdown("### Daily Courier Cost vs Orders – Combined")
    fig_combined = plot_orders_vs_cost_both_cities(daily, city_labels=("City 1", "City 2"))
    st.plotly_chart(fig_combined, use_container_width=True, key="orders_cost_combined")

    st.markdown("---")

    # 6) Correlation Matrix
    st.subheader("Correlation Matrix")
    city_corr = st.selectbox("Choose city for correlation", options=sorted(daily["city"].unique()), key="corr_city_pick")
    all_candidates = [
        "orders", "avg_delivery_time", "TASK_COST", "GUARANTEE_COST",
        "OTHER_COST", "tot_daily_cost", "CPO", "revenue", "total_revenue",
        "contribution_margin", "Guarantee_pct"
    ]
    metrics_pick = st.multiselect(
        "Metrics to include",
        options=[c for c in all_candidates if c in daily.columns],
        default=[m for m in ["orders","avg_delivery_time","TASK_COST","GUARANTEE_COST",
                             "CPO","contribution_margin","Guarantee_pct"]
                 if m in daily.columns],
        key="corr_metric_pick"
    )
    fig_corr = plot_city_correlation_heatmap(daily, city_corr, metrics=metrics_pick, height=520)
    st.plotly_chart(fig_corr, use_container_width=True, key=f"corr_heatmap_{city_corr}")

    st.markdown("---")

# 7) OLS Regression (directly under correlation matrix)
    st.subheader("OLS Regression: Costs vs Avg Delivery Time")

    city_reg = st.selectbox(
        "Choose city for regression",
        options=sorted(daily["city"].unique()),
        key="reg_city_pick"
    )

    model, df_reg = run_city_regression(daily, city_reg)
    if model is None:
        st.info("Not enough data for this city.")
    else:
        # Regression summary
        st.text(model.summary().as_text())

    # Plotly scatter + OLS trendline (needs a key)
    fig = px.scatter(
        df_reg, x="OTHER_COST", y="avg_delivery_time",
        trendline="ols", opacity=0.8,
        color_discrete_sequence=["steelblue"],
        title=f"{city_reg} – Other Cost vs Avg Delivery Time"
    )
    st.plotly_chart(fig, use_container_width=True, key=f"reg_scatter_{city_reg}")

    # Seaborn regplot (NO key here)
    if {"OTHER_COST", "avg_delivery_time"}.issubset(df_reg.columns):
        df_plot = df_reg.replace([np.inf, -np.inf], np.nan).dropna(
            subset=["OTHER_COST", "avg_delivery_time"]
        )

        fig2, ax = plt.subplots(figsize=(6, 5))
        sns.regplot(
            data=df_plot,
            x="OTHER_COST",
            y="avg_delivery_time",
            scatter_kws={"alpha": 0.85, "s": 45},
            line_kws={"linewidth": 2.0},
            color="#1f77b4",
        )
        ax.set_title("Other Cost vs Avg Delivery Time", fontsize=16, pad=10)
        ax.set_xlabel("OTHER_COST", fontsize=12)
        ax.set_ylabel("avg_delivery_time", fontsize=12)
        ax.grid(False)

        st.pyplot(fig2, clear_figure=True)  # <-- no key
        plt.close(fig2)
    else:
        st.info("Required columns for the regplot are missing in the filtered data.")


# ============================================================
# TAB 4 – City 1 Venues
# ============================================================
with tabs[3]:
    st.subheader("City 1 – Venue Performance")
    if city1_sales_raw is None:
        st.info("Upload city1_sales.csv (raw order-level) to use this view.")
    else:
        df = city1_sales_raw.copy()
        # Basic filters
        if "date" in df.columns:
            if date_sel:
                start, end = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
                df = df[(pd.to_datetime(df["date"]) >= start) & (pd.to_datetime(df["date"]) <= end)]
        # Filter to homedelivery + non-preorder if columns exist
        if "DELIVERY_METHOD" in df.columns:
            df = df[df["DELIVERY_METHOD"].str.lower() == "homedelivery"]
        if "PREORDER" in df.columns:
            pre = df["PREORDER"].astype(str).str.lower().isin(["true","yes","1"])
            df = df[~pre]

        if "VENUE_NAME" in df.columns and "TOTAL_DELIVERY_TIME" in df.columns:
            top_n = st.slider("Top N venues by orders", 5, 50, 15)
            ven = (df.groupby("VENUE_NAME")
                     .agg(orders=("USER_ID","count"),
                          avg_delivery=("TOTAL_DELIVERY_TIME","mean"))
                     .sort_values("orders", ascending=False)
                     .reset_index()
                   )
            ven = ven.head(top_n)
            fig = px.bar(ven, x="VENUE_NAME", y="orders", title="Orders by venue (top N)")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(ven, x="VENUE_NAME", y="avg_delivery", title="Avg delivery time by venue (top N)")
            st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 5 – City 2 Venues
# ============================================================
with tabs[4]:
    st.subheader("City 2 – Venue Performance")
    if city2_sales_raw is None:
        st.info("Upload city2_sales.csv (raw order-level) to use this view.")
    else:
        df = city2_sales_raw.copy()
        if "date" in df.columns:
            if date_sel:
                start, end = pd.to_datetime(date_sel[0]), pd.to_datetime(date_sel[1])
                df = df[(pd.to_datetime(df["date"]) >= start) & (pd.to_datetime(df["date"]) <= end)]
        if "DELIVERY_METHOD" in df.columns:
            df = df[df["DELIVERY_METHOD"].str.lower() == "homedelivery"]
        if "PREORDER" in df.columns:
            pre = df["PREORDER"].astype(str).str.lower().isin(["true","yes","1"])
            df = df[~pre]

        if "VENUE_NAME" in df.columns and "TOTAL_DELIVERY_TIME" in df.columns:
            top_n = st.slider("Top N venues by orders", 5, 50, 15, key="v2")
            ven = (df.groupby("VENUE_NAME")
                     .agg(orders=("USER_ID","count"),
                          avg_delivery=("TOTAL_DELIVERY_TIME","mean"))
                     .sort_values("orders", ascending=False)
                     .reset_index()
                   )
            ven = ven.head(top_n)
            fig = px.bar(ven, x="VENUE_NAME", y="orders", title="Orders by venue (top N)")
            st.plotly_chart(fig, use_container_width=True)

            fig = px.bar(ven, x="VENUE_NAME", y="avg_delivery", title="Avg delivery time by venue (top N)")
            st.plotly_chart(fig, use_container_width=True)