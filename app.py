"""
Campus Sales Tracker & Forecaster
==================================
A single-file Streamlit application for small campus businesses (fashion
brands, food vendors, gadget resellers) to track sales, understand what
drives them, and forecast near-term demand.

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats
from sklearn.linear_model import LinearRegression
from datetime import timedelta

# --------------------------------------------------------------------------
# PAGE CONFIG — wide layout collapses gracefully on mobile, sidebar becomes
# a top drawer on small screens automatically in Streamlit.
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="Campus Sales Tracker",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def check_password():
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    if st.session_state.password_correct:
        return True

    # Show input for password
    password = st.text_input("Enter Password to Access Tracker", type="password")
    if password == "YourSecretPassword123": # Change this to your password
        st.session_state.password_correct = True
        st.rerun()
    elif password:
        st.error("Incorrect password")
    return False

if not check_password():
    st.stop()  # Stops the rest of the app from running

# --- YOUR ACTUAL APP CODE STARTS HERE ---
st.title("Campus Sales Tracker & Forecaster")



REQUIRED_COLS = ["Date", "Item_Category", "Units_Sold", "Item_Price", "Ad_Spend", "Customer_Views"]


# --------------------------------------------------------------------------
# DATA HELPERS
# --------------------------------------------------------------------------
def generate_demo_data(n_days: int = 90) -> pd.DataFrame:
    """Creates a realistic mock transaction history so the app is testable
    instantly without a real file. Simulates a campus business with
    trending demand, seasonality (weekend spikes), and noise."""
    rng = np.random.default_rng(42)
    dates = pd.date_range(end=pd.Timestamp.today().normalize(), periods=n_days)
    categories = ["Hoodies", "Sneakers", "Phone Cases", "Snacks", "Earbuds"]

    rows = []
    for i, date in enumerate(dates):
        # Weekend boost (Fri/Sat/Sun = higher foot traffic on campus)
        weekend_boost = 1.4 if date.weekday() in [4, 5, 6] else 1.0
        # Slow upward growth trend over time (brand gaining traction)
        growth = 1 + (i / n_days) * 0.6

        for cat in categories:
            base_price = {"Hoodies": 25, "Sneakers": 60, "Phone Cases": 12,
                          "Snacks": 3, "Earbuds": 20}[cat]
            ad_spend = max(0, rng.normal(15, 6) * growth)
            views = max(0, rng.normal(200, 50) * growth * weekend_boost)
            # Units sold driven by ad spend + views + some randomness
            units = max(0, (0.05 * ad_spend + 0.02 * views) * weekend_boost
                        + rng.normal(0, 3))
            price = base_price + rng.normal(0, 1)

            rows.append({
                "Date": date,
                "Item_Category": cat,
                "Units_Sold": round(units),
                "Item_Price": round(max(1, price), 2),
                "Ad_Spend": round(ad_spend, 2),
                "Customer_Views": round(views),
            })

    return pd.DataFrame(rows)


def load_uploaded_file(uploaded_file) -> pd.DataFrame:
    """Reads a CSV or Excel file uploaded by the user."""
    if uploaded_file.name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)


def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Ensures required columns exist, parses dates, and derives Total_Sales."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        st.error(f"Your file is missing required columns: {', '.join(missing)}")
        st.stop()

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["Total_Sales"] = df["Units_Sold"] * df["Item_Price"]
    df = df.sort_values("Date")
    return df


# --------------------------------------------------------------------------
# SIDEBAR — DATA SOURCE
# --------------------------------------------------------------------------
st.title("📈 Campus Sales Tracker & Forecaster")
st.caption("Understand what drives your sales, and see where they're headed.")

with st.sidebar:
    st.header("1. Load Your Data")
    uploaded_file = st.file_uploader(
        "Upload sales history (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        help="Columns needed: Date, Item_Category, Units_Sold, Item_Price, Ad_Spend, Customer_Views",
    )
    demo_clicked = st.button("🎲 Load Demo Data", use_container_width=True)

# Decide which dataset to use, and remember the choice across reruns
if "data" not in st.session_state:
    st.session_state.data = None

if demo_clicked:
    st.session_state.data = generate_demo_data()
elif uploaded_file is not None:
    st.session_state.data = load_uploaded_file(uploaded_file)

if st.session_state.data is None:
    st.info("👋 Upload a sales file or click **Load Demo Data** in the sidebar to get started.")
    st.stop()

df = validate_and_clean(st.session_state.data)

# Aggregate to a clean daily total series — used throughout the app
daily = df.groupby("Date").agg(
    Total_Sales=("Total_Sales", "sum"),
    Units_Sold=("Units_Sold", "sum"),
    Ad_Spend=("Ad_Spend", "sum"),
    Customer_Views=("Customer_Views", "sum"),
    Item_Price=("Item_Price", "mean"),
).reset_index()
daily["Day_Index"] = np.arange(len(daily))  # numeric time axis for regression

# --------------------------------------------------------------------------
# QUICK OVERVIEW METRICS
# --------------------------------------------------------------------------
st.subheader("Business Snapshot")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Revenue", f"${df['Total_Sales'].sum():,.0f}")
c2.metric("Units Sold", f"{df['Units_Sold'].sum():,.0f}")
c3.metric("Avg. Daily Sales", f"${daily['Total_Sales'].mean():,.0f}")
c4.metric("Days Tracked", f"{len(daily)}")

st.divider()

# --------------------------------------------------------------------------
# 2. CORRELATION ENGINE
# --------------------------------------------------------------------------
st.subheader("🔍 What's Actually Driving Your Sales?")
st.caption(
    "Pearson finds **straight-line** relationships. Spearman catches "
    "**curving/ranked** relationships that Pearson can miss."
)

# Focus columns for correlation analysis
corr_cols = ["Total_Sales", "Units_Sold", "Ad_Spend", "Customer_Views", "Item_Price"]
corr_df = daily[corr_cols]

pearson_matrix = corr_df.corr(method="pearson")
spearman_matrix = corr_df.corr(method="spearman")

# Headline metrics: Ad Spend vs Sales (Pearson), Price vs Units (Spearman)
pearson_r, pearson_p = stats.pearsonr(daily["Ad_Spend"], daily["Total_Sales"])
spearman_r, spearman_p = stats.spearmanr(daily["Item_Price"], daily["Units_Sold"])

m1, m2 = st.columns(2)
with m1:
    st.metric("Ad Spend → Sales (Pearson r)", f"{pearson_r:.2f}")
    if pearson_r > 0.5:
        st.success("💰 Strong link: more ad spend is closely tied to more revenue.")
    elif pearson_r > 0.2:
        st.info("🙂 Moderate link: ads help, but other factors matter too.")
    else:
        st.warning("⚠️ Weak link: your ad spend isn't translating into sales well right now.")

with m2:
    st.metric("Price → Units Sold (Spearman ρ)", f"{spearman_r:.2f}")
    if spearman_r < -0.4:
        st.warning("📉 Clear pattern: raising prices is costing you volume.")
    elif spearman_r < -0.1:
        st.info("🙂 Mild pattern: price increases slightly reduce demand.")
    else:
        st.success("💪 Your customers aren't very price-sensitive — some room to raise prices.")

# Interactive heatmap comparing both correlation methods side by side
tab_pearson, tab_spearman = st.tabs(["📐 Pearson (Linear)", "📊 Spearman (Rank/Non-linear)"])
with tab_pearson:
    fig_p = px.imshow(
        pearson_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto", title="Pearson Correlation Matrix",
    )
    st.plotly_chart(fig_p, use_container_width=True)
with tab_spearman:
    fig_s = px.imshow(
        spearman_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, aspect="auto", title="Spearman Correlation Matrix",
    )
    st.plotly_chart(fig_s, use_container_width=True)

st.caption(
    "📖 **How to read this:** Values near **+1** mean two things rise and fall together. "
    "Values near **-1** mean when one goes up, the other goes down. Values near **0** mean "
    "little to no relationship."
)

st.divider()

# --------------------------------------------------------------------------
# 3. PREDICTIVE FORECASTING & WHAT-IF ANALYSIS
# --------------------------------------------------------------------------
st.subheader("🔮 Sales Forecast")

forecast_days = st.slider("How many days ahead do you want to forecast?", 1, 30, 7)

# --- Simple time-trend model: Total_Sales ~ Day_Index ---
X_time = daily[["Day_Index"]]
y_sales = daily["Total_Sales"]
trend_model = LinearRegression().fit(X_time, y_sales)

future_index = np.arange(daily["Day_Index"].max() + 1, daily["Day_Index"].max() + 1 + forecast_days)
future_dates = pd.date_range(daily["Date"].max() + timedelta(days=1), periods=forecast_days)
future_sales_pred = trend_model.predict(future_index.reshape(-1, 1))
future_sales_pred = np.clip(future_sales_pred, 0, None)  # sales can't go negative

forecast_df = pd.DataFrame({"Date": future_dates, "Total_Sales": future_sales_pred, "Type": "Forecast"})
history_df = daily[["Date", "Total_Sales"]].copy()
history_df["Type"] = "Actual"
combined = pd.concat([history_df, forecast_df], ignore_index=True)

fig_forecast = px.line(
    combined, x="Date", y="Total_Sales", color="Type",
    title=f"Revenue: Actual History + {forecast_days}-Day Forecast",
    markers=True,
)
st.plotly_chart(fig_forecast, use_container_width=True)

trend_slope = trend_model.coef_[0]
if trend_slope > 0:
    st.success(f"📈 Your sales are trending **up** by roughly ${trend_slope:,.2f}/day.")
else:
    st.warning(f"📉 Your sales are trending **down** by roughly ${abs(trend_slope):,.2f}/day.")

# --------------------------------------------------------------------------
# WHAT-IF SIMULATION — Multiple Regression on Units_Sold
# --------------------------------------------------------------------------
st.subheader("🧪 What-If Simulator")
st.caption("Adjust hypothetical ad spend and pricing to see the projected impact on next week's unit sales.")

# Multiple linear regression: Units_Sold ~ Ad_Spend + Item_Price + Customer_Views + Day_Index
features = ["Ad_Spend", "Item_Price", "Customer_Views", "Day_Index"]
X_multi = daily[features]
y_units = daily["Units_Sold"]
multi_model = LinearRegression().fit(X_multi, y_units)

w1, w2 = st.columns(2)
with w1:
    ad_spend_change = st.slider("Hypothetical Ad Spend Change (%)", -50, 100, 0, step=5)
with w2:
    price_change = st.slider("Hypothetical Price Adjustment (%)", -30, 30, 0, step=5)

# Build a 7-day what-if projection using recent averages as the baseline,
# then apply the user's hypothetical percentage adjustments.
sim_days = 7
baseline_ad = daily["Ad_Spend"].tail(14).mean()
baseline_price = daily["Item_Price"].tail(14).mean()
baseline_views = daily["Customer_Views"].tail(14).mean()

sim_ad = baseline_ad * (1 + ad_spend_change / 100)
sim_price = baseline_price * (1 + price_change / 100)
sim_index_start = daily["Day_Index"].max() + 1

sim_rows = []
baseline_rows = []
for d in range(sim_days):
    idx = sim_index_start + d
    # "What-if" scenario prediction
    sim_pred = multi_model.predict([[sim_ad, sim_price, baseline_views, idx]])[0]
    # Baseline (no change) prediction, for comparison
    base_pred = multi_model.predict([[baseline_ad, baseline_price, baseline_views, idx]])[0]
    sim_rows.append(max(0, sim_pred))
    baseline_rows.append(max(0, base_pred))

sim_dates = pd.date_range(daily["Date"].max() + timedelta(days=1), periods=sim_days)
whatif_df = pd.DataFrame({
    "Date": list(sim_dates) * 2,
    "Units_Sold": baseline_rows + sim_rows,
    "Scenario": ["Current Plan"] * sim_days + ["What-If Scenario"] * sim_days,
})

fig_whatif = px.line(
    whatif_df, x="Date", y="Units_Sold", color="Scenario", markers=True,
    title="Next 7 Days: Current Plan vs. What-If Scenario (Units Sold)",
)
st.plotly_chart(fig_whatif, use_container_width=True)

delta_units = sum(sim_rows) - sum(baseline_rows)
delta_pct = (delta_units / sum(baseline_rows) * 100) if sum(baseline_rows) > 0 else 0
if delta_units > 0:
    st.success(f"✅ This scenario projects **+{delta_units:.0f} extra units** ({delta_pct:+.1f}%) over the next week.")
elif delta_units < 0:
    st.warning(f"⚠️ This scenario projects **{delta_units:.0f} fewer units** ({delta_pct:+.1f}%) over the next week.")
else:
    st.info("No meaningful change projected for this scenario.")

st.divider()

# --------------------------------------------------------------------------
# 4. ACTIONABLE BUSINESS ALERTS
# --------------------------------------------------------------------------
st.subheader("🚨 Business Alerts")

alerts = []

# Alert 1: strong recent demand velocity → possible stockout risk
recent = daily.tail(7)
if len(recent) >= 2:
    recent_slope = np.polyfit(recent["Day_Index"], recent["Units_Sold"], 1)[0]
    if recent_slope > daily["Units_Sold"].mean() * 0.05:
        alerts.append(
            "📦 **Stockout Risk:** Unit sales are climbing fast this week. "
            "Reorder inventory ahead of the next peak weekend to avoid running out."
        )

# Alert 2: weak ad spend → sales correlation
if pearson_r < 0.2:
    alerts.append(
        "💸 **Ad Spend Efficiency:** Your ad spend isn't showing a strong link to revenue. "
        "Consider testing different ad creative or targeting before increasing budget further."
    )
elif pearson_r > 0.6 and trend_slope < 0:
    alerts.append(
        "📢 **Reinvest in Ads:** Ad spend is strongly tied to sales, but your overall trend is declining. "
        "A modest ad budget increase could help reverse the slide."
    )

# Alert 3: price sensitivity
if spearman_r < -0.4:
    alerts.append(
        "🏷️ **Price Sensitivity Warning:** Customers are clearly buying less at higher prices. "
        "Avoid further price hikes, or pair them with a value-add (bundle, discount code) to soften the impact."
    )

# Alert 4: declining category check
cat_trend = df.groupby("Item_Category")["Total_Sales"].sum().sort_values()
if len(cat_trend) > 0:
    weakest = cat_trend.index[0]
    alerts.append(
        f"📉 **Underperforming Category:** *{weakest}* has the lowest total revenue of all categories. "
        "Consider a promotion, bundling, or phasing it out in favor of stronger sellers."
    )

# Alert 5: forecast-based outlook
if forecast_days >= 1:
    forecast_total = forecast_df["Total_Sales"].sum()
    alerts.append(
        f"🔮 **Forecast Outlook:** Based on current trends, expect roughly **${forecast_total:,.0f}** "
        f"in revenue over the next {forecast_days} day(s)."
    )

if alerts:
    for a in alerts:
        st.markdown(f"- {a}")
else:
    st.info("No major alerts right now — your business metrics look stable.")

st.divider()
st.caption("Built for student entrepreneurs. Upload fresh sales data regularly for more accurate forecasts.")
