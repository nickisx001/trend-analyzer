"""
================================================================================
CAMPUS MARKETPLACE ANALYTICS ENGINE — Production Multi-Tenant Edition
================================================================================
Architecture:
  - PostgreSQL persistence via SQLAlchemy (st.connection)
  - Hashed-password vendor authentication (multi-tenant row isolation)
  - Robust CSV/Excel ingestion with dynamic column mapping + data cleaning
  - Random Forest demand model + prescriptive price optimization
  - Executive Plotly dashboard

Drop this file in as app.py. Configure `.streamlit/secrets.toml`:

    [postgres]
    host = "your-host"
    port = 5432
    dbname = "campus_market"
    user = "your_user"
    password = "your_password"

Run:  streamlit run app.py
================================================================================
"""

import hashlib
import hmac
import os
import secrets
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import text

# ==============================================================================
# 0. PAGE CONFIG
# ==============================================================================
st.set_page_config(
    page_title="Campus Marketplace Analytics",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==============================================================================
# 1. DATABASE CONNECTION (SQLAlchemy via st.connection)
# ==============================================================================
# st.connection("postgresql", type="sql") reads credentials from
# st.secrets["postgres"] automatically and manages a pooled SQLAlchemy engine
# for us. This is the officially recommended pattern for production Streamlit
# apps talking to relational databases.


@st.cache_resource
def get_connection():
    """Return a cached SQLAlchemy-backed Streamlit SQL connection."""
    try:
        conn = st.connection("postgresql", type="sql")
        return conn
    except Exception as e:
        st.error(
            "❌ Could not connect to the database. Verify `.streamlit/secrets.toml` "
            "contains a valid [postgres] block.\n\n"
            f"Details: {e}"
        )
        st.stop()


conn = get_connection()


# ==============================================================================
# 2. SCHEMA INITIALIZATION
# ==============================================================================
# Idempotent DDL — safe to run on every app boot. In a real production
# pipeline this would live in an Alembic migration instead of being run
# inline, but it's kept here so the script is fully self-contained.

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS vendors (
    vendor_id       SERIAL PRIMARY KEY,
    username        VARCHAR(64)  UNIQUE NOT NULL,
    password_hash   VARCHAR(256) NOT NULL,
    password_salt   VARCHAR(64)  NOT NULL,
    business_name   VARCHAR(128) NOT NULL,
    email           VARCHAR(128),
    created_at      TIMESTAMP DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS products (
    product_id      SERIAL PRIMARY KEY,
    vendor_id       INTEGER NOT NULL REFERENCES vendors(vendor_id) ON DELETE CASCADE,
    product_name    VARCHAR(128) NOT NULL,
    category        VARCHAR(64)  NOT NULL,
    base_price      NUMERIC(10, 2) NOT NULL,
    UNIQUE (vendor_id, product_name)
);

CREATE TABLE IF NOT EXISTS transactions (
    transaction_id  SERIAL PRIMARY KEY,
    product_id      INTEGER NOT NULL REFERENCES products(product_id) ON DELETE CASCADE,
    units_sold      INTEGER NOT NULL,
    item_price      NUMERIC(10, 2) NOT NULL,
    date_time       TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_products_vendor ON products(vendor_id);
CREATE INDEX IF NOT EXISTS idx_transactions_product ON transactions(product_id);
CREATE INDEX IF NOT EXISTS idx_transactions_datetime ON transactions(date_time);
"""


def init_db():
    with conn.session as s:
        for statement in SCHEMA_SQL.strip().split(";"):
            if statement.strip():
                s.execute(text(statement))
        s.commit()


init_db()

# ==============================================================================
# 3. AUTHENTICATION (salted hash, no plaintext passwords ever touch the DB)
# ==============================================================================


def hash_password(password: str, salt: str = None) -> tuple[str, str]:
    """Return (hash, salt) using PBKDF2-HMAC-SHA256."""
    salt = salt or secrets.token_hex(16)
    pw_hash = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()
    return pw_hash, salt


def verify_password(password: str, stored_hash: str, salt: str) -> bool:
    candidate, _ = hash_password(password, salt)
    return hmac.compare_digest(candidate, stored_hash)


def signup_vendor(username, password, business_name, email) -> tuple[bool, str]:
    existing = conn.query(
        "SELECT vendor_id FROM vendors WHERE username = :u",
        params={"u": username},
        ttl=0,
    )
    if not existing.empty:
        return False, "Username already taken."

    pw_hash, salt = hash_password(password)
    with conn.session as s:
        s.execute(
            text(
                """
                INSERT INTO vendors (username, password_hash, password_salt, business_name, email)
                VALUES (:u, :h, :s, :b, :e)
                """
            ),
            {"u": username, "h": pw_hash, "s": salt, "b": business_name, "e": email},
        )
        s.commit()
    return True, "Account created. Please log in."


def login_vendor(username, password):
    row = conn.query(
        "SELECT * FROM vendors WHERE username = :u",
        params={"u": username},
        ttl=0,
    )
    if row.empty:
        return None
    record = row.iloc[0]
    if verify_password(password, record["password_hash"], record["password_salt"]):
        return record
    return None


def render_auth_sidebar():
    """Renders login/signup forms; returns True once authenticated."""
    st.sidebar.title("🏪 Vendor Portal")

    if st.session_state.get("vendor") is not None:
        vendor = st.session_state["vendor"]
        st.sidebar.success(f"Signed in as **{vendor['business_name']}**")
        if st.sidebar.button("Log out", use_container_width=True):
            st.session_state["vendor"] = None
            st.rerun()
        return True

    tab_login, tab_signup = st.sidebar.tabs(["Log In", "Sign Up"])

    with tab_login:
        with st.form("login_form"):
            u = st.text_input("Username")
            p = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Log In", use_container_width=True)
            if submitted:
                vendor = login_vendor(u, p)
                if vendor is not None:
                    st.session_state["vendor"] = vendor
                    st.rerun()
                else:
                    st.error("Invalid username or password.")

    with tab_signup:
        with st.form("signup_form"):
            u = st.text_input("Choose a username")
            biz = st.text_input("Business name")
            email = st.text_input("Email (optional)")
            p1 = st.text_input("Password", type="password")
            p2 = st.text_input("Confirm password", type="password")
            submitted = st.form_submit_button("Create Account", use_container_width=True)
            if submitted:
                if not u or not p1 or not biz:
                    st.error("Username, business name and password are required.")
                elif p1 != p2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = signup_vendor(u, p1, biz, email)
                    (st.success if ok else st.error)(msg)

    return False


if "vendor" not in st.session_state:
    st.session_state["vendor"] = None

authenticated = render_auth_sidebar()

if not authenticated:
    st.title("📈 Campus Marketplace Analytics Engine")
    st.info("Log in or create a vendor account in the sidebar to access your dashboard.")
    st.stop()

VENDOR = st.session_state["vendor"]
VENDOR_ID = int(VENDOR["vendor_id"])

# ==============================================================================
# 4. TENANT-SCOPED DATA ACCESS LAYER
# ==============================================================================
# Every query below is parameterized on vendor_id, guaranteeing row-level
# multi-tenant isolation — a vendor can never see another vendor's records.


def get_vendor_transactions() -> pd.DataFrame:
    query = """
        SELECT
            t.transaction_id,
            p.product_id,
            p.product_name,
            p.category,
            p.base_price,
            t.units_sold,
            t.item_price,
            t.date_time
        FROM transactions t
        JOIN products p ON p.product_id = t.product_id
        WHERE p.vendor_id = :vid
        ORDER BY t.date_time
    """
    df = conn.query(query, params={"vid": VENDOR_ID}, ttl=30)
    if not df.empty:
        df["date_time"] = pd.to_datetime(df["date_time"])
        df["units_sold"] = df["units_sold"].astype(float)
        df["item_price"] = df["item_price"].astype(float)
        df["base_price"] = df["base_price"].astype(float)
        df["revenue"] = df["units_sold"] * df["item_price"]
        df["profit"] = (df["item_price"] - df["base_price"]) * df["units_sold"]
    return df


def get_or_create_product(product_name: str, category: str, base_price: float) -> int:
    existing = conn.query(
        "SELECT product_id FROM products WHERE vendor_id = :vid AND product_name = :pn",
        params={"vid": VENDOR_ID, "pn": product_name},
        ttl=0,
    )
    if not existing.empty:
        return int(existing.iloc[0]["product_id"])

    with conn.session as s:
        result = s.execute(
            text(
                """
                INSERT INTO products (vendor_id, product_name, category, base_price)
                VALUES (:vid, :pn, :cat, :bp)
                RETURNING product_id
                """
            ),
            {"vid": VENDOR_ID, "pn": product_name, "cat": category, "bp": base_price},
        )
        new_id = result.fetchone()[0]
        s.commit()
    return int(new_id)


def bulk_insert_transactions(rows: list[dict]):
    with conn.session as s:
        s.execute(
            text(
                """
                INSERT INTO transactions (product_id, units_sold, item_price, date_time)
                VALUES (:product_id, :units_sold, :item_price, :date_time)
                """
            ),
            rows,
        )
        s.commit()


# ==============================================================================
# 5. DATA CLEANING HELPERS
# ==============================================================================


def clean_currency_series(series: pd.Series) -> pd.Series:
    """Strip currency symbols (₦, $, €, £, commas, whitespace) and coerce to float."""
    cleaned = (
        series.astype(str)
        .str.replace(r"[₦$€£,]", "", regex=True)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = series.astype(str).str.replace(",", "", regex=False).str.strip()
    return pd.to_numeric(cleaned, errors="coerce")


# ==============================================================================
# 6. SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.divider()
page = st.sidebar.radio(
    "Navigate",
    ["📊 Performance Dashboard", "📥 Upload Records", "🤖 AI Profit Optimizer"],
    label_visibility="collapsed",
)

df = get_vendor_transactions()

# ==============================================================================
# 7. PAGE — PERFORMANCE DASHBOARD
# ==============================================================================
if page == "📊 Performance Dashboard":
    st.title("📊 Performance Dashboard")
    st.caption(f"Live tenant-scoped view for **{VENDOR['business_name']}**")

    if df.empty:
        st.warning("No transactions yet. Head to **📥 Upload Records** to ingest your sales history.")
    else:
        gross_revenue = df["revenue"].sum()
        volume_moved = df["units_sold"].sum()
        net_profit = df["profit"].sum()
        margin = (net_profit / gross_revenue * 100) if gross_revenue else 0

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("💰 Gross Revenue", f"₦{gross_revenue:,.0f}")
        c2.metric("📦 Volumes Moved", f"{volume_moved:,.0f} units")
        c3.metric("📈 Net Profit", f"₦{net_profit:,.0f}")
        c4.metric("🎯 Margin", f"{margin:.1f}%")

        st.divider()

        # --- Time-series sales velocity with moving average ---------------------
        st.subheader("Sales Velocity Over Time")
        daily = (
            df.set_index("date_time")
            .resample("D")["revenue"]
            .sum()
            .reset_index()
            .rename(columns={"revenue": "daily_revenue"})
        )
        daily["7d_moving_avg"] = daily["daily_revenue"].rolling(7, min_periods=1).mean()

        fig_ts = go.Figure()
        fig_ts.add_trace(
            go.Bar(x=daily["date_time"], y=daily["daily_revenue"], name="Daily Revenue", marker_color="#93C5FD")
        )
        fig_ts.add_trace(
            go.Scatter(
                x=daily["date_time"],
                y=daily["7d_moving_avg"],
                name="7-Day Moving Avg",
                line=dict(color="#1D4ED8", width=3),
            )
        )
        fig_ts.update_layout(
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(t=30, l=10, r=10, b=10),
            height=420,
        )
        st.plotly_chart(fig_ts, use_container_width=True)

        # --- Market share breakdown ----------------------------------------------
        col_a, col_b = st.columns([1.3, 1])
        with col_a:
            st.subheader("Market Share by Product")
            share = (
                df.groupby("product_name")["revenue"]
                .sum()
                .sort_values(ascending=True)
                .reset_index()
            )
            fig_share = px.bar(
                share,
                x="revenue",
                y="product_name",
                orientation="h",
                text_auto=".2s",
                color="revenue",
                color_continuous_scale="Blues",
            )
            fig_share.update_layout(
                template="plotly_white",
                coloraxis_showscale=False,
                margin=dict(t=10, l=10, r=10, b=10),
                xaxis_title="Revenue (₦)",
                yaxis_title="",
                height=420,
            )
            st.plotly_chart(fig_share, use_container_width=True)

        with col_b:
            st.subheader("Category Mix")
            cat_mix = df.groupby("category")["revenue"].sum().reset_index()
            fig_pie = px.pie(cat_mix, names="category", values="revenue", hole=0.55)
            fig_pie.update_traces(textinfo="percent+label")
            fig_pie.update_layout(template="plotly_white", margin=dict(t=10, l=10, r=10, b=10), height=420)
            st.plotly_chart(fig_pie, use_container_width=True)

        with st.expander("🔍 Raw transaction records"):
            st.dataframe(df, use_container_width=True)

# ==============================================================================
# 8. PAGE — UPLOAD RECORDS (ingestion + dynamic column mapping)
# ==============================================================================
elif page == "📥 Upload Records":
    st.title("📥 Upload Records")
    st.caption("Ingest legacy spreadsheets into the production database with automatic cleaning.")

    uploaded = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx", "xls"])

    if uploaded is not None:
        try:
            raw_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"Could not parse file: {e}")
            st.stop()

        st.write("Preview of uploaded data:")
        st.dataframe(raw_df.head(10), use_container_width=True)

        st.subheader("🧭 Map Your Columns")
        st.caption("Match your spreadsheet headers to the fields our database expects.")

        cols = ["-- none --"] + list(raw_df.columns)
        m1, m2, m3 = st.columns(3)
        with m1:
            col_product = st.selectbox("Product Name column", cols, index=0)
            col_category = st.selectbox("Category column", cols, index=0)
        with m2:
            col_units = st.selectbox("Units Sold column", cols, index=0)
            col_price = st.selectbox("Item Price column", cols, index=0)
        with m3:
            col_date = st.selectbox("Date/Time column", cols, index=0)
            default_base_price = st.number_input(
                "Fallback base (cost) price for new products (₦)", min_value=0.0, value=0.0, step=50.0
            )

        required = {"Product Name": col_product, "Units Sold": col_units, "Item Price": col_price, "Date/Time": col_date}
        missing = [k for k, v in required.items() if v == "-- none --"]

        if missing:
            st.warning(f"Please map the following required fields: {', '.join(missing)}")
        else:
            if st.button("🧹 Clean & Preview", use_container_width=True):
                work = pd.DataFrame()
                work["product_name"] = raw_df[col_product].astype(str).str.strip()
                work["category"] = (
                    raw_df[col_category].astype(str).str.strip() if col_category != "-- none --" else "Uncategorized"
                )
                work["units_sold"] = clean_numeric_series(raw_df[col_units])
                work["item_price"] = clean_currency_series(raw_df[col_price])
                work["date_time"] = pd.to_datetime(raw_df[col_date], errors="coerce")

                before = len(work)
                work = work.dropna(subset=["product_name", "units_sold", "item_price", "date_time"])
                work = work[(work["units_sold"] > 0) & (work["item_price"] >= 0)]
                dropped = before - len(work)

                st.session_state["_clean_upload"] = work
                st.session_state["_default_base_price"] = default_base_price

                st.success(f"Cleaned {len(work)} valid rows ({dropped} rows dropped for missing/invalid data).")
                st.dataframe(work.head(20), use_container_width=True)

        if "_clean_upload" in st.session_state and st.button("💾 Commit to Database", type="primary", use_container_width=True):
            work = st.session_state["_clean_upload"]
            base_price_fallback = st.session_state.get("_default_base_price", 0.0)

            product_id_cache: dict[str, int] = {}
            rows_to_insert = []

            progress = st.progress(0.0, text="Writing to database...")
            for i, row in enumerate(work.itertuples(index=False)):
                key = (row.product_name, row.category)
                cache_key = f"{row.product_name}|{row.category}"
                if cache_key not in product_id_cache:
                    # Use the average observed price as base cost if no fallback given
                    est_base = base_price_fallback if base_price_fallback > 0 else row.item_price * 0.6
                    product_id_cache[cache_key] = get_or_create_product(row.product_name, row.category, est_base)

                rows_to_insert.append(
                    {
                        "product_id": product_id_cache[cache_key],
                        "units_sold": int(row.units_sold),
                        "item_price": float(row.item_price),
                        "date_time": row.date_time.to_pydatetime(),
                    }
                )
                progress.progress((i + 1) / len(work), text=f"Writing to database... {i + 1}/{len(work)}")

            bulk_insert_transactions(rows_to_insert)
            progress.empty()
            st.success(f"✅ Inserted {len(rows_to_insert)} transactions across {len(product_id_cache)} products.")
            del st.session_state["_clean_upload"]
            st.balloons()
            st.cache_data.clear()

    st.divider()
    with st.expander("➕ Or add a single record manually"):
        with st.form("manual_entry"):
            pn = st.text_input("Product name")
            cat = st.text_input("Category", value="Uncategorized")
            bp = st.number_input("Base (cost) price ₦", min_value=0.0, step=50.0)
            units = st.number_input("Units sold", min_value=1, step=1)
            price = st.number_input("Item (selling) price ₦", min_value=0.0, step=50.0)
            dt = st.date_input("Date", value=datetime.now())
            submitted = st.form_submit_button("Add Record", use_container_width=True)
            if submitted and pn:
                pid = get_or_create_product(pn, cat, bp)
                bulk_insert_transactions(
                    [{"product_id": pid, "units_sold": int(units), "item_price": float(price), "date_time": dt}]
                )
                st.success("Record added.")
                st.cache_data.clear()

# ==============================================================================
# 9. PAGE — AI PROFIT OPTIMIZER
# ==============================================================================
elif page == "🤖 AI Profit Optimizer":
    st.title("🤖 AI Profit Maximization Simulator")
    st.caption(
        "Trains a Random Forest demand model on your transaction history, capturing weekday/monthly "
        "campus buying patterns, then searches for the price that maximizes net profit."
    )

    MIN_ROWS_TO_TRAIN = 15

    if df.empty or len(df) < MIN_ROWS_TO_TRAIN:
        st.warning(
            f"Need at least {MIN_ROWS_TO_TRAIN} historical transactions to train a reliable model. "
            f"You currently have {len(df)}. Upload more records first."
        )
    else:
        categories = sorted(df["category"].unique().tolist())
        category = st.selectbox("Select a product category to optimize", categories)

        cat_df = df[df["category"] == category].copy()

        if len(cat_df) < 8:
            st.warning(f"Only {len(cat_df)} records in '{category}'. Need at least 8 to train per-category.")
        else:
            # ---------------- Feature engineering ----------------------------------
            cat_df["day_of_week"] = cat_df["date_time"].dt.dayofweek  # 0=Mon ... 6=Sun
            cat_df["month"] = cat_df["date_time"].dt.month
            cat_df["is_weekend"] = cat_df["day_of_week"].isin([5, 6]).astype(int)

            product_encoder = LabelEncoder()
            cat_df["product_encoded"] = product_encoder.fit_transform(cat_df["product_name"])

            feature_cols = ["item_price", "day_of_week", "month", "is_weekend", "product_encoded"]
            X = cat_df[feature_cols]
            y = cat_df["units_sold"]

            # ---------------- Train demand model ------------------------------------
            model = RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
            )
            model.fit(X, y)

            avg_base_price = cat_df["base_price"].mean()
            observed_min_price = cat_df["item_price"].min()
            observed_max_price = cat_df["item_price"].max()

            c1, c2 = st.columns(2)
            with c1:
                price_floor = st.number_input(
                    "Price search floor (₦)", value=float(max(avg_base_price, observed_min_price * 0.7)), step=10.0
                )
            with c2:
                price_ceiling = st.number_input(
                    "Price search ceiling (₦)", value=float(observed_max_price * 1.5), step=10.0
                )

            sim_context = st.selectbox(
                "Simulate for", ["Typical weekday", "Weekend (Fri–Sun)", "Post-allowance week (best historical month)"]
            )

            if sim_context == "Typical weekday":
                sim_dow, sim_is_weekend = 2, 0
            elif sim_context == "Weekend (Fri–Sun)":
                sim_dow, sim_is_weekend = 5, 1
            else:
                sim_dow, sim_is_weekend = 2, 0

            best_month = int(cat_df.groupby("month")["units_sold"].sum().idxmax())
            sim_month = best_month if sim_context.startswith("Post") else int(cat_df["month"].mode()[0])
            sim_product_encoded = int(cat_df["product_encoded"].mode()[0])
            sim_product_name = product_encoder.inverse_transform([sim_product_encoded])[0]

            if st.button("🚀 Run Profit Optimization", type="primary", use_container_width=True):
                price_grid = np.linspace(price_floor, price_ceiling, 200)
                sim_frame = pd.DataFrame(
                    {
                        "item_price": price_grid,
                        "day_of_week": sim_dow,
                        "month": sim_month,
                        "is_weekend": sim_is_weekend,
                        "product_encoded": sim_product_encoded,
                    }
                )

                predicted_units = model.predict(sim_frame[feature_cols])
                predicted_units = np.clip(predicted_units, 0, None)

                cost_basis = avg_base_price
                predicted_profit = (price_grid - cost_basis) * predicted_units
                predicted_revenue = price_grid * predicted_units

                best_idx = int(np.argmax(predicted_profit))
                sweet_spot_price = price_grid[best_idx]
                sweet_spot_profit = predicted_profit[best_idx]
                sweet_spot_units = predicted_units[best_idx]

                st.divider()
                st.subheader("🎯 Absolute Sweet Spot")
                m1, m2, m3 = st.columns(3)
                m1.metric("Optimal Price", f"₦{sweet_spot_price:,.0f}")
                m2.metric("Projected Units Sold", f"{sweet_spot_units:,.1f}")
                m3.metric("Projected Net Profit", f"₦{sweet_spot_profit:,.0f}")
                st.caption(
                    f"Simulation context: representative product **{sim_product_name}**, "
                    f"{'weekend' if sim_is_weekend else 'weekday'} demand pattern, month #{sim_month}. "
                    f"Cost basis assumed at avg base price ₦{cost_basis:,.0f}."
                )

                sim_result = pd.DataFrame(
                    {
                        "price": price_grid,
                        "predicted_units": predicted_units,
                        "predicted_profit": predicted_profit,
                        "predicted_revenue": predicted_revenue,
                    }
                )

                fig_opt = go.Figure()
                fig_opt.add_trace(
                    go.Scatter(
                        x=sim_result["price"],
                        y=sim_result["predicted_profit"],
                        name="Predicted Net Profit",
                        line=dict(color="#059669", width=3),
                    )
                )
                fig_opt.add_trace(
                    go.Scatter(
                        x=sim_result["price"],
                        y=sim_result["predicted_revenue"],
                        name="Predicted Revenue",
                        line=dict(color="#93C5FD", width=2, dash="dot"),
                    )
                )
                fig_opt.add_vline(
                    x=sweet_spot_price,
                    line_dash="dash",
                    line_color="#DC2626",
                    annotation_text="Sweet Spot",
                    annotation_position="top",
                )
                fig_opt.update_layout(
                    template="plotly_white",
                    xaxis_title="Price (₦)",
                    yaxis_title="₦",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    height=450,
                    margin=dict(t=30, l=10, r=10, b=10),
                )
                st.plotly_chart(fig_opt, use_container_width=True)

                with st.expander("📐 Model diagnostics"):
                    importances = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
                    st.write("**Feature importance** (what drives demand):")
                    st.bar_chart(importances)
                    st.caption(
                        "Random Forest trained with 300 trees, max depth 8, on category-scoped transaction "
                        "history. Re-trains fresh on every optimization run using the latest database state."
                    )
