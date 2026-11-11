from pathlib import Path
import warnings

import gdown
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

warnings.filterwarnings("ignore")

# ============================================================
# PROJECT PATHS AND DATA SOURCE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATA_FILE = DATA_DIR / "delivery_gap_analysis.csv"
GAP_FILE_ID = "12hs2Z9nrCwiTw-mSbf7goNlHIrx8Bizk"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
REPORT_DIR = OUTPUT_DIR / "reports"
TABLE_DIR = OUTPUT_DIR / "tables"

for folder in (DATA_DIR, REPORT_DIR, TABLE_DIR):
    folder.mkdir(parents=True, exist_ok=True)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Global Supply Chain Operations",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Global Supply Chain Operations")
st.subheader("Delivery Performance, Delay Risk, and Logistics Efficiency")
st.caption(
    "Interactive analysis of shipment delivery performance, operational "
    "risk, shipping modes, regions, markets, and customer segments."
)


# ============================================================
# DATA LOADING AND COLUMN IDENTIFICATION
# ============================================================

@st.cache_data(show_spinner="Downloading and loading logistics dataset...")
def load_data():
    if not DATA_FILE.exists():
        result = gdown.download(
            id=GAP_FILE_ID,
            output=str(DATA_FILE),
            quiet=False,
        )
        if not result or not DATA_FILE.exists():
            raise FileNotFoundError(
                "Could not download the delivery gap dataset from Google Drive."
            )

    df = pd.read_csv(DATA_FILE, low_memory=False)
    df.columns = df.columns.astype(str).str.strip()

    if df.empty:
        raise ValueError("The logistics dataset is empty.")

    # Reduce memory for low-cardinality text fields and numeric columns.
    # Original column names and values are retained.
    row_count = max(len(df), 1)
    object_columns = df.select_dtypes(include=["object"]).columns

    for col in object_columns:
        unique_count = df[col].nunique(dropna=False)
        if unique_count / row_count <= 0.05:
            df[col] = df[col].astype("category")

    integer_columns = df.select_dtypes(include=["int64"]).columns
    for col in integer_columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")

    float_columns = df.select_dtypes(include=["float64"]).columns
    for col in float_columns:
        df[col] = pd.to_numeric(df[col], downcast="float")

    return df, str(DATA_FILE)


def find_column(df, candidates):
    normalized = {
        str(col).strip().lower().replace("_", " ").replace("-", " "): col
        for col in df.columns
    }

    for candidate in candidates:
        key = (
            str(candidate)
            .strip()
            .lower()
            .replace("_", " ")
            .replace("-", " ")
        )
        if key in normalized:
            return normalized[key]

    return None


def prepare_data(source):
    # Keep a separate working frame so cached source data is not mutated.
    df = source.copy()

    columns = {
        "actual": find_column(
            df,
            [
                "Days for shipping (real)",
                "actual shipping duration",
                "actual duration",
            ],
        ),
        "scheduled": find_column(
            df,
            [
                "Days for shipment (scheduled)",
                "scheduled shipping duration",
                "scheduled duration",
            ],
        ),
        "gap": find_column(df, ["Delay Gap", "delivery gap", "delay gap"]),
        "status": find_column(df, ["Delivery Status"]),
        "risk": find_column(df, ["Late_delivery_risk", "late delivery risk"]),
        "mode": find_column(df, ["Shipping Mode"]),
        "region": find_column(df, ["Order Region"]),
        "country": find_column(df, ["Order Country"]),
        "market": find_column(df, ["Market"]),
        "segment": find_column(df, ["Customer Segment"]),
    }

    if columns["actual"] is None or columns["scheduled"] is None:
        raise ValueError(
            "Actual and scheduled shipping duration columns are required."
        )

    df[columns["actual"]] = pd.to_numeric(
        df[columns["actual"]], errors="coerce"
    )
    df[columns["scheduled"]] = pd.to_numeric(
        df[columns["scheduled"]], errors="coerce"
    )

    if columns["risk"] is not None:
        df[columns["risk"]] = pd.to_numeric(
            df[columns["risk"]], errors="coerce"
        )

    if columns["gap"] is not None:
        df["_delay_gap"] = pd.to_numeric(
            df[columns["gap"]], errors="coerce"
        )
    else:
        df["_delay_gap"] = (
            df[columns["actual"]] - df[columns["scheduled"]]
        )

    valid_duration = (
        df[columns["actual"]].notna()
        & df[columns["scheduled"]].notna()
        & (df[columns["actual"]] >= 0)
        & (df[columns["scheduled"]] >= 0)
        & df["_delay_gap"].notna()
    )

    cancelled = pd.Series(False, index=df.index)
    if columns["status"] is not None:
        cancelled = (
            df[columns["status"]]
            .astype("string")
            .str.strip()
            .str.lower()
            .str.contains("cancel", na=False)
        )

    df["_eligible"] = valid_duration & ~cancelled

    df["_classification"] = np.select(
        [
            df["_delay_gap"] > 0,
            df["_delay_gap"] == 0,
            df["_delay_gap"] < 0,
        ],
        ["Delayed", "On-time", "Early"],
        default="Invalid",
    )
    df.loc[~df["_eligible"], "_classification"] = "Ineligible"

    return df, columns


# ============================================================
# ELIGIBLE POPULATION AND KPI CALCULATIONS
# ============================================================

def eligible_records(df):
    return df.loc[df["_eligible"]]


def calculate_kpis(df, columns):
    eligible = eligible_records(df)

    if eligible.empty:
        return {
            "Total Eligible Shipments": 0,
            "On-Time Delivery Rate (%)": 0.0,
            "Average Delivery Delay (Days)": 0.0,
            "Late Delivery Risk Ratio (%)": 0.0,
            "Shipping Mode Efficiency Index": 0.0,
            "Regional Delay Index": 0.0,
        }

    gap = eligible["_delay_gap"]
    total = len(eligible)
    delayed = int((gap > 0).sum())
    sla_compliant = int((gap <= 0).sum())

    # Project KPI formula: SLA-compliant / eligible shipments.
    on_time_rate = sla_compliant / total * 100
    average_delay = float(gap[gap > 0].mean()) if delayed else 0.0

    risk_ratio = 0.0
    if columns["risk"] is not None:
        risk = eligible[columns["risk"]]
        risk = risk[risk.isin([0, 1])]
        if not risk.empty:
            risk_ratio = float(risk.mean() * 100)

    mode_table = calculate_shipping_modes(eligible, columns)
    region_table = calculate_regions(eligible, columns)

    mode_index = (
        float(mode_table["Shipping Mode Efficiency Index"].max())
        if not mode_table.empty
        else 0.0
    )
    region_index = (
        float(region_table["Regional Delay Index"].max())
        if not region_table.empty
        else 0.0
    )

    return {
        "Total Eligible Shipments": int(total),
        "On-Time Delivery Rate (%)": round(on_time_rate, 2),
        "Average Delivery Delay (Days)": round(average_delay, 2),
        "Late Delivery Risk Ratio (%)": round(risk_ratio, 2),
        "Shipping Mode Efficiency Index": round(mode_index, 2),
        "Regional Delay Index": round(region_index, 2),
    }


# ============================================================
# SHIPPING MODE PERFORMANCE
# ============================================================

def calculate_shipping_modes(df, columns):
    mode_col = columns["mode"]
    if mode_col is None or df.empty:
        return pd.DataFrame()

    records = []

    for mode, group in df.groupby(mode_col, dropna=False, observed=True):
        gap = group["_delay_gap"]
        delayed = gap[gap > 0]

        risk_ratio = np.nan
        if columns["risk"] is not None:
            risk = group[columns["risk"]]
            risk = risk[risk.isin([0, 1])]
            if not risk.empty:
                risk_ratio = risk.mean() * 100

        records.append(
            {
                "Shipping Mode": str(mode),
                "Shipment Count": len(group),
                "Delayed Shipments": int((gap > 0).sum()),
                "On-time Shipments": int((gap == 0).sum()),
                "Early Shipments": int((gap < 0).sum()),
                "SLA-Compliant Shipments": int((gap <= 0).sum()),
                "Average Delay (Days)": (
                    delayed.mean() if not delayed.empty else 0.0
                ),
                "On-Time Rate (%)": (gap == 0).mean() * 100,
                "SLA Compliance (%)": (gap <= 0).mean() * 100,
                "Late Risk (%)": risk_ratio,
            }
        )

    result = pd.DataFrame(records)
    if result.empty:
        return result

    valid_risk_mean = result["Late Risk (%)"].mean()
    if pd.isna(valid_risk_mean):
        valid_risk_mean = 0.0

    inverse_risk = 100 - result["Late Risk (%)"].fillna(valid_risk_mean)
    maximum_delay = result["Average Delay (Days)"].max()

    if maximum_delay > 0:
        delay_score = (
            1 - result["Average Delay (Days)"] / maximum_delay
        ) * 100
    else:
        delay_score = pd.Series(100.0, index=result.index)

    result["Shipping Mode Efficiency Index"] = (
        0.50 * result["SLA Compliance (%)"]
        + 0.30 * inverse_risk
        + 0.20 * delay_score
    ).clip(0, 100).round(2)

    result = result.sort_values(
        "Shipping Mode Efficiency Index", ascending=False
    ).reset_index(drop=True)
    result.insert(0, "Efficiency Rank", range(1, len(result) + 1))

    return result


# ============================================================
# REGIONAL PERFORMANCE
# ============================================================

def calculate_regions(df, columns):
    region_col = columns["region"]
    if region_col is None or df.empty:
        return pd.DataFrame()

    result = (
        df.groupby(region_col, dropna=False, observed=True)
        .agg(
            Shipment_Count=("_delay_gap", "size"),
            Delayed_Shipments=(
                "_delay_gap", lambda values: int((values > 0).sum())
            ),
            On_Time_Shipments=(
                "_delay_gap", lambda values: int((values == 0).sum())
            ),
            Early_Shipments=(
                "_delay_gap", lambda values: int((values < 0).sum())
            ),
            SLA_Compliant_Shipments=(
                "_delay_gap", lambda values: int((values <= 0).sum())
            ),
            Average_Delay_Days=(
                "_delay_gap",
                lambda values: (
                    values[values > 0].mean()
                    if (values > 0).any()
                    else 0.0
                ),
            ),
        )
        .reset_index()
    )

    result["Delayed_Percentage"] = (
        result["Delayed_Shipments"] / result["Shipment_Count"] * 100
    )
    result["SLA_Compliance_Percentage"] = (
        result["SLA_Compliant_Shipments"]
        / result["Shipment_Count"]
        * 100
    )

    maximum_delay = result["Average_Delay_Days"].max()
    if maximum_delay > 0:
        result["Regional Delay Index"] = (
            result["Average_Delay_Days"] / maximum_delay * 100
        )
    else:
        result["Regional Delay Index"] = 0.0

    result = result.rename(columns={region_col: "Order Region"})
    return result.sort_values(
        "Delayed_Percentage", ascending=False
    ).reset_index(drop=True)


# ============================================================
# COUNTRY PERFORMANCE
# ============================================================

def calculate_countries(df, columns):
    country_col = columns["country"]
    if country_col is None or df.empty:
        return pd.DataFrame()

    result = (
        df.groupby(country_col, dropna=False, observed=True)
        .agg(
            Shipment_Count=("_delay_gap", "size"),
            Delayed_Shipments=(
                "_delay_gap", lambda values: int((values > 0).sum())
            ),
            Average_Delay_Days=(
                "_delay_gap",
                lambda values: (
                    values[values > 0].mean()
                    if (values > 0).any()
                    else 0.0
                ),
            ),
        )
        .reset_index()
    )

    result["Delayed_Percentage"] = (
        result["Delayed_Shipments"] / result["Shipment_Count"] * 100
    )
    result = result.rename(columns={country_col: "Order Country"})

    return result.sort_values(
        "Delayed_Percentage", ascending=False
    ).reset_index(drop=True)


# ============================================================
# MARKET PERFORMANCE
# ============================================================

def calculate_markets(df, columns):
    market_col = columns["market"]
    if market_col is None or df.empty:
        return pd.DataFrame()

    result = (
        df.groupby(market_col, dropna=False, observed=True)
        .agg(
            Shipment_Count=("_delay_gap", "size"),
            Delayed_Shipments=(
                "_delay_gap", lambda values: int((values > 0).sum())
            ),
            On_Time_Shipments=(
                "_delay_gap", lambda values: int((values == 0).sum())
            ),
            SLA_Compliant_Shipments=(
                "_delay_gap", lambda values: int((values <= 0).sum())
            ),
            Average_Delay_Days=(
                "_delay_gap",
                lambda values: (
                    values[values > 0].mean()
                    if (values > 0).any()
                    else 0.0
                ),
            ),
        )
        .reset_index()
    )

    result["Delayed_Percentage"] = (
        result["Delayed_Shipments"] / result["Shipment_Count"] * 100
    )
    result["SLA_Compliance_Percentage"] = (
        result["SLA_Compliant_Shipments"]
        / result["Shipment_Count"]
        * 100
    )
    result = result.rename(columns={market_col: "Market"})

    return result.sort_values(
        "Delayed_Percentage", ascending=False
    ).reset_index(drop=True)


# ============================================================
# DOWNLOAD AND DISPLAY UTILITIES
# ============================================================

def download_csv(data, filename, label):
    if data is None or data.empty:
        st.info(f"No data available for {filename}.")
        return

    csv_data = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=label,
        data=csv_data,
        file_name=filename,
        mime="text/csv",
        use_container_width=True,
        key=f"download_{filename}",
    )


def display_chart(fig):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=30),
        legend_title_text="",
    )
    st.plotly_chart(fig, use_container_width=True)


def percentage(numerator, denominator):
    return numerator / denominator * 100 if denominator else 0.0


# ============================================================
# LOAD APPLICATION DATA
# ============================================================

try:
    source_data, source_path = load_data()
    data, columns = prepare_data(source_data)
except Exception as error:
    st.error(f"Dashboard data loading failed: {error}")
    st.exception(error)
    st.stop()


# ============================================================
# GLOBAL FILTERS
# ============================================================

st.sidebar.title("Dashboard Navigation")
st.sidebar.caption("Global analytical filters")

if st.sidebar.button("Reset All Filters", use_container_width=True):
    for key in (
        "filter_shipping_mode",
        "filter_region",
        "filter_market",
        "filter_segment",
    ):
        st.session_state.pop(key, None)
    st.rerun()


def apply_filter(df, column, key, label):
    if column is None:
        st.sidebar.warning(f"{label} is unavailable.")
        return df

    options = sorted(df[column].dropna().astype(str).unique().tolist())
    selected = st.sidebar.multiselect(
        label,
        options,
        default=options,
        key=key,
    )
    return df[df[column].astype(str).isin(selected)]


filtered_data = data.copy()

filtered_data = apply_filter(
    filtered_data,
    columns["mode"],
    "filter_shipping_mode",
    "Shipping Mode",
)
filtered_data = apply_filter(
    filtered_data,
    columns["region"],
    "filter_region",
    "Order Region",
)
filtered_data = apply_filter(
    filtered_data,
    columns["market"],
    "filter_market",
    "Market",
)
filtered_data = apply_filter(
    filtered_data,
    columns["segment"],
    "filter_segment",
    "Customer Segment",
)


# ============================================================
# DATE FILTER DEPENDENCY
# ============================================================

st.sidebar.subheader("Date Range Filter")

date_column = find_column(
    data,
    ["date", "order date", "shipping date", "delivery date"],
)

if date_column is None:
    st.sidebar.warning(
        "Date range filtering is blocked because the source dataset "
        "does not contain a genuine date column."
    )
    st.sidebar.caption(
        "Date filtering will be enabled only when genuine date data "
        "is available and validated. No artificial dates are created."
    )
else:
    st.sidebar.warning(
        "A date-like field exists, but date filtering remains disabled "
        "until the field is verified as a genuine operational date."
    )


# ============================================================
# FILTERED RECORD COUNTS
# ============================================================

eligible = eligible_records(filtered_data)

st.sidebar.divider()
st.sidebar.subheader("Filtered Data Summary")
st.sidebar.metric("Source Records", f"{len(data):,}")
st.sidebar.metric("Filtered Records", f"{len(filtered_data):,}")
st.sidebar.metric("Eligible Shipments", f"{len(eligible):,}")
st.sidebar.metric(
    "Eligible Percentage",
    f"{percentage(len(eligible), len(filtered_data)):.2f}%",
)

if filtered_data.empty:
    st.warning("No records match the selected filters.")
    st.stop()

if eligible.empty:
    st.warning("No eligible delivery records are available.")
    st.stop()


# ============================================================
# CALCULATE DASHBOARD METRICS
# ============================================================

kpis = calculate_kpis(filtered_data, columns)
mode_performance = calculate_shipping_modes(eligible, columns)
region_performance = calculate_regions(eligible, columns)
country_performance = calculate_countries(eligible, columns)
market_performance = calculate_markets(eligible, columns)


# ============================================================
# FIVE REQUIRED LOGISTICS KPIs
# ============================================================

st.header("Five Required Logistics KPIs")

k1, k2, k3, k4, k5 = st.columns(5)

k1.metric(
    "On-Time Delivery Rate",
    f"{kpis['On-Time Delivery Rate (%)']:.2f}%",
)
k2.metric(
    "Average Delivery Delay",
    f"{kpis['Average Delivery Delay (Days)']:.2f} days",
)
k3.metric(
    "Late Delivery Risk Ratio",
    f"{kpis['Late Delivery Risk Ratio (%)']:.2f}%",
)
k4.metric(
    "Shipping Mode Efficiency Index",
    f"{kpis['Shipping Mode Efficiency Index']:.2f}",
)
k5.metric(
    "Regional Delay Index",
    f"{kpis['Regional Delay Index']:.2f}",
)

st.caption(
    "The project's On-Time Delivery Rate KPI is calculated as "
    "SLA-compliant shipments (delay gap <= 0) divided by eligible "
    "shipments. Exact on-time shipments (gap = 0) are reported separately. "
    "The SLA-compliant KPI includes early shipments."
)


# ============================================================
# FOUR ORIGINAL ANALYTICAL MODULES
# ============================================================

module = st.sidebar.radio(
    "Select Analytical Module",
    [
        "Delivery Performance Overview",
        "Delay Risk Analysis",
        "Shipping Mode Comparison",
        "Regional and Market Heatmaps",
    ],
)


# ============================================================
# MODULE 1: DELIVERY PERFORMANCE OVERVIEW
# ============================================================

if module == "Delivery Performance Overview":
    st.header("Delivery Performance Overview")

    gap = eligible["_delay_gap"]
    total = len(eligible)
    delayed = int((gap > 0).sum())
    on_time = int((gap == 0).sum())
    early = int((gap < 0).sum())
    sla_compliant = int((gap <= 0).sum())
    average_delay = gap[gap > 0].mean() if delayed else 0.0

    a, b, c, d, e = st.columns(5)
    a.metric("Total Eligible Shipments", f"{total:,}")
    b.metric("Exactly On-time", f"{on_time:,}")
    c.metric("Delayed", f"{delayed:,}")
    d.metric("Early", f"{early:,}")
    e.metric("Average Delay", f"{average_delay:.2f} days")

    st.subheader("Delivery Classification Summary")
    classification = pd.DataFrame(
        {
            "Classification": ["Delayed", "On-time", "Early"],
            "Shipment Count": [delayed, on_time, early],
        }
    )
    fig = px.bar(
        classification,
        x="Classification",
        y="Shipment Count",
        color="Classification",
        text_auto=True,
        title="Delivery Classification by Shipment Count",
    )
    display_chart(fig)

    st.subheader("Delivery Status Distribution")
    if columns["status"] is not None:
        status = (
            eligible[columns["status"]]
            .astype("string")
            .fillna("Unknown")
            .value_counts()
            .rename_axis("Delivery Status")
            .reset_index(name="Shipment Count")
        )
        fig = px.pie(
            status,
            names="Delivery Status",
            values="Shipment Count",
            hole=0.4,
            title="Shipment Distribution by Delivery Status",
        )
        display_chart(fig)
    else:
        st.info("Delivery Status is unavailable.")

    st.subheader("Actual vs Scheduled Shipping Duration")
    actual_col = columns["actual"]
    scheduled_col = columns["scheduled"]
    duration_data = pd.DataFrame(
        {
            "Duration": ["Actual", "Scheduled"],
            "Average Days": [
                eligible[actual_col].mean(),
                eligible[scheduled_col].mean(),
            ],
        }
    )
    fig = px.bar(
        duration_data,
        x="Duration",
        y="Average Days",
        text_auto=".2f",
        title="Average Actual and Scheduled Shipping Duration",
    )
    display_chart(fig)

    st.subheader("Delivery Performance Summary")
    st.write(f"Eligible shipments: **{total:,}**")
    st.write(
        f"Exactly on-time shipments: **{on_time:,} "
        f"({percentage(on_time, total):.2f}%)**"
    )
    st.write(
        f"Delayed shipments: **{delayed:,} "
        f"({percentage(delayed, total):.2f}%)**"
    )
    st.write(
        f"Early shipments: **{early:,} "
        f"({percentage(early, total):.2f}%)**"
    )
    st.write(
        f"SLA-compliant shipments: **{sla_compliant:,} "
        f"({percentage(sla_compliant, total):.2f}%)**"
    )
    st.write(
        f"Average delay among delayed shipments: **{average_delay:.2f} days**"
    )

    st.subheader("Interactive Filtered Shipment Records")
    st.dataframe(filtered_data.head(5000), use_container_width=True)

    download_csv(
        filtered_data,
        "filtered_delivery_performance.csv",
        "Download Filtered Delivery Data",
    )


# ============================================================
# MODULE 2: DELAY RISK ANALYSIS
# ============================================================

elif module == "Delay Risk Analysis":
    st.header("Delay Risk Analysis Dashboard")

    if columns["risk"] is None:
        st.error("Late_delivery_risk column is unavailable.")
        st.stop()

    risk_col = columns["risk"]
    risk_data = eligible.loc[eligible[risk_col].isin([0, 1])].copy()

    if risk_data.empty:
        st.warning("No valid binary risk labels are available.")
        st.stop()

    risk_rate = risk_data[risk_col].mean() * 100
    a, b, c = st.columns(3)
    a.metric("Valid Risk Records", f"{len(risk_data):,}")
    b.metric("Late Risk Percentage", f"{risk_rate:.2f}%")
    c.metric("Risk Label Count", f"{int(risk_data[risk_col].sum()):,}")

    st.subheader("Late Delivery Risk Distribution")
    risk_distribution = (
        risk_data[risk_col]
        .value_counts()
        .reindex([0, 1], fill_value=0)
        .rename_axis("Risk Label")
        .reset_index(name="Shipment Count")
    )
    risk_distribution["Risk Category"] = risk_distribution[
        "Risk Label"
    ].map({0: "Risk = 0", 1: "Risk = 1"})

    fig = px.pie(
        risk_distribution,
        names="Risk Category",
        values="Shipment Count",
        hole=0.4,
        title="Existing Late Delivery Risk Label Distribution",
    )
    display_chart(fig)

    st.subheader("Delay Gap Histogram")
    fig = px.histogram(
        eligible,
        x="_delay_gap",
        color="_classification",
        nbins=30,
        title="Delivery Delay Gap Distribution",
    )
    display_chart(fig)

    st.subheader("Delivery Classification")
    classification = (
        eligible["_classification"]
        .value_counts()
        .reindex(["Delayed", "On-time", "Early"], fill_value=0)
        .rename_axis("Classification")
        .reset_index(name="Shipment Count")
    )
    fig = px.bar(
        classification,
        x="Classification",
        y="Shipment Count",
        color="Classification",
        text_auto=True,
    )
    display_chart(fig)

    st.subheader("Risk Labels Compared with Actual Delivery Outcome")
    risk_data["_actual_outcome"] = np.where(
        risk_data["_delay_gap"] > 0,
        "Delayed",
        "Not Delayed",
    )
    risk_data["_risk_category"] = risk_data[risk_col].map(
        {0: "Risk = 0", 1: "Risk = 1"}
    )

    risk_comparison = pd.crosstab(
        risk_data["_actual_outcome"],
        risk_data["_risk_category"],
    ).reindex(
        index=["Delayed", "Not Delayed"],
        columns=["Risk = 0", "Risk = 1"],
        fill_value=0,
    ).reset_index()

    fig = px.bar(
        risk_comparison,
        x="_actual_outcome",
        y=["Risk = 0", "Risk = 1"],
        barmode="group",
        title="Existing Risk Labels vs Actual Delivery Outcome",
    )
    display_chart(fig)

    st.subheader("Risk Mismatch Analysis")
    risk_data["_risk_mismatch"] = (
        (risk_data[risk_col] == 1)
        != (risk_data["_delay_gap"] > 0)
    )
    mismatches = risk_data.loc[risk_data["_risk_mismatch"]].copy()

    m1, m2 = st.columns(2)
    m1.metric("Risk Mismatches", f"{len(mismatches):,}")
    m2.metric(
        "Mismatch Percentage",
        f"{percentage(len(mismatches), len(risk_data)):.2f}%",
    )

    st.dataframe(mismatches.head(5000), use_container_width=True)

    st.subheader("High-Risk Shipment Summary")
    high_risk = risk_data.loc[risk_data[risk_col] == 1]
    high_risk_delays = high_risk.loc[
        high_risk["_delay_gap"] > 0, "_delay_gap"
    ]
    high_risk_summary = pd.DataFrame(
        [
            {
                "Risk Label Shipments": len(high_risk),
                "Actually Delayed Shipments": int(
                    (high_risk["_delay_gap"] > 0).sum()
                ),
                "Average Positive Delay": (
                    high_risk_delays.mean()
                    if not high_risk_delays.empty
                    else 0
                ),
            }
        ]
    )
    st.dataframe(high_risk_summary, use_container_width=True)

    st.caption(
        "Late_delivery_risk is an existing dataset label, not a new "
        "prediction or a probability produced by this dashboard."
    )

    download_csv(
        risk_data,
        "filtered_delay_risk_analysis.csv",
        "Download Risk Analysis",
    )
    download_csv(
        mismatches,
        "risk_mismatch_records.csv",
        "Download Risk Mismatch Records",
    )


# ============================================================
# MODULE 3: SHIPPING MODE COMPARISON
# ============================================================

elif module == "Shipping Mode Comparison":
    st.header("Shipping Mode Comparison")

    if mode_performance.empty:
        st.warning("Shipping mode performance data is unavailable.")
        st.stop()

    selected_mode = st.selectbox(
        "Select Shipping Mode",
        ["All Shipping Modes"] + mode_performance["Shipping Mode"].tolist(),
    )
    mode_display = mode_performance.copy()

    if selected_mode != "All Shipping Modes":
        mode_display = mode_display.loc[
            mode_display["Shipping Mode"] == selected_mode
        ]

    a, b, c, d = st.columns(4)
    a.metric("Shipments", f"{mode_display['Shipment Count'].sum():,.0f}")
    b.metric(
        "Average Delay",
        f"{mode_display['Average Delay (Days)'].mean():.2f} days",
    )
    c.metric(
        "Average On-time Rate",
        f"{mode_display['On-Time Rate (%)'].mean():.2f}%",
    )
    d.metric(
        "Average SLA Compliance",
        f"{mode_display['SLA Compliance (%)'].mean():.2f}%",
    )

    st.subheader("Mode-wise Shipment Count")
    fig = px.bar(
        mode_display,
        x="Shipping Mode",
        y="Shipment Count",
        color="Shipping Mode",
        text_auto=True,
    )
    display_chart(fig)

    st.subheader("Mode-wise Average Delay")
    fig = px.bar(
        mode_display,
        x="Shipping Mode",
        y="Average Delay (Days)",
        color="Shipping Mode",
        text_auto=".2f",
    )
    display_chart(fig)

    st.subheader("Mode-wise On-time Delivery")
    fig = px.bar(
        mode_display,
        x="Shipping Mode",
        y="On-Time Rate (%)",
        color="Shipping Mode",
        title="Exactly On-time Delivery Percentage",
    )
    display_chart(fig)

    st.subheader("Mode-wise SLA Compliance")
    fig = px.bar(
        mode_display,
        x="Shipping Mode",
        y="SLA Compliance (%)",
        color="Shipping Mode",
        title="SLA Compliance Including Early Shipments",
    )
    display_chart(fig)

    st.subheader("Mode-wise Late Delivery Risk")
    fig = px.bar(
        mode_display,
        x="Shipping Mode",
        y="Late Risk (%)",
        color="Shipping Mode",
    )
    display_chart(fig)

    st.subheader("Shipping Mode Efficiency Ranking")
    fig = px.bar(
        mode_display.sort_values("Shipping Mode Efficiency Index"),
        x="Shipping Mode Efficiency Index",
        y="Shipping Mode",
        orientation="h",
        text="Shipping Mode Efficiency Index",
    )
    display_chart(fig)

    st.subheader("Shipping Mode Performance Table")
    st.dataframe(
        mode_display,
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Efficiency Index = 50% SLA compliance + 30% inverse late risk "
        "+ 20% relative delay-gap score. This is a project-defined index."
    )

    download_csv(
        mode_display,
        "shipping_mode_dashboard_performance.csv",
        "Download Shipping Mode Performance",
    )


# ============================================================
# MODULE 4: REGIONAL AND MARKET HEATMAPS
# ============================================================

elif module == "Regional and Market Heatmaps":
    st.header("Regional and Market Heatmaps")
    st.subheader("Region-wise Delivery Performance")

    if region_performance.empty:
        st.warning("Regional performance data is unavailable.")
    else:
        region_options = region_performance["Order Region"].astype(str).tolist()
        selected_region = st.selectbox(
            "Select Order Region",
            ["All Regions"] + region_options,
        )
        region_display = region_performance.copy()

        if selected_region != "All Regions":
            region_display = region_display.loc[
                region_display["Order Region"].astype(str) == selected_region
            ]

        region_heatmap = region_display.set_index("Order Region")[
            [
                "Delayed_Percentage",
                "SLA_Compliance_Percentage",
                "Regional Delay Index",
            ]
        ]
        fig = px.imshow(
            region_heatmap,
            aspect="auto",
            color_continuous_scale="Reds",
            text_auto=".2f",
            title="Regional Delivery Performance Heatmap",
        )
        display_chart(fig)

        st.subheader("Regional Delay Comparison")
        fig = px.bar(
            region_display.head(20),
            x="Order Region",
            y="Delayed_Percentage",
            color="Delayed_Percentage",
            title="Regional Delayed Shipment Percentage",
        )
        display_chart(fig)

        st.subheader("Regional Delay Index")
        fig = px.bar(
            region_display.head(20),
            x="Order Region",
            y="Regional Delay Index",
            color="Regional Delay Index",
        )
        display_chart(fig)

        st.dataframe(
            region_display,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Country-level Delivery Performance")

    if country_performance.empty:
        st.warning("Country performance data is unavailable.")
    else:
        max_countries = min(30, len(country_performance))
        top_n = st.slider(
            "Number of Countries to Display",
            min_value=1,
            max_value=max_countries,
            value=min(15, max_countries),
        )
        country_display = country_performance.head(top_n)
        country_heatmap = country_display.set_index("Order Country")[
            ["Delayed_Percentage", "Average_Delay_Days"]
        ]

        fig = px.imshow(
            country_heatmap,
            aspect="auto",
            color_continuous_scale="Reds",
            text_auto=".2f",
            title="Country-level Delivery Performance Heatmap",
        )
        display_chart(fig)

        fig = px.bar(
            country_display,
            x="Order Country",
            y="Delayed_Percentage",
            color="Delayed_Percentage",
            title="Highest Delay Countries",
        )
        display_chart(fig)

        st.dataframe(
            country_display,
            use_container_width=True,
            hide_index=True,
        )

    st.divider()
    st.subheader("Market-wise Performance")

    if market_performance.empty:
        st.warning("Market performance data is unavailable.")
    else:
        market_options = market_performance["Market"].astype(str).tolist()
        selected_market = st.selectbox(
            "Select Market",
            ["All Markets"] + market_options,
        )
        market_display = market_performance.copy()

        if selected_market != "All Markets":
            market_display = market_display.loc[
                market_display["Market"].astype(str) == selected_market
            ]

        fig = px.bar(
            market_display,
            x="Market",
            y="Delayed_Percentage",
            color="Delayed_Percentage",
            title="Market-wise Delayed Shipment Percentage",
        )
        display_chart(fig)

        fig = px.bar(
            market_display,
            x="Market",
            y="SLA_Compliance_Percentage",
            color="SLA_Compliance_Percentage",
            title="Market-wise SLA Compliance",
        )
        display_chart(fig)

        st.dataframe(
            market_display,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Regional and country heatmaps use categorical performance data. "
        "They do not represent coordinate-based destination maps. "
        "Customer coordinates are not treated as destination coordinates."
    )

    download_csv(
        region_performance,
        "regional_dashboard_performance.csv",
        "Download Regional Performance",
    )
    download_csv(
        country_performance,
        "country_dashboard_performance.csv",
        "Download Country Performance",
    )
    download_csv(
        market_performance,
        "market_dashboard_performance.csv",
        "Download Market Performance",
    )


# ============================================================
# GLOBAL REPORT DOWNLOADS
# ============================================================

st.sidebar.divider()
st.sidebar.header("Download Analytical Reports")

kpi_download = pd.DataFrame(
    [{"KPI": key, "Value": value} for key, value in kpis.items()]
)

download_csv(
    filtered_data,
    "complete_filtered_dashboard_data.csv",
    "Download Filtered Dataset",
)
download_csv(
    kpi_download,
    "five_required_dashboard_kpis.csv",
    "Download Five KPI Results",
)
download_csv(
    mode_performance,
    "dashboard_shipping_mode_report.csv",
    "Download Shipping Mode Report",
)
download_csv(
    region_performance,
    "dashboard_regional_report.csv",
    "Download Regional Report",
)
download_csv(
    market_performance,
    "dashboard_market_report.csv",
    "Download Market Report",
)


# ============================================================
# DASHBOARD DEFINITIONS AND DATA LIMITATIONS
# ============================================================

with st.expander("Dashboard Definitions and Data Limitations"):
    st.markdown(
        """
        **Delivery eligibility**

        Cancelled shipments and records with missing, invalid or negative
        actual or scheduled durations are excluded. Valid zero-duration
        Same Day shipments are retained.

        **Delay Gap**

        Actual shipping duration minus scheduled shipping duration.

        - Positive gap: Delayed.
        - Zero gap: Exactly on-time.
        - Negative gap: Early.

        **On-Time Delivery Rate KPI**

        Project-defined KPI: shipments with gap <= 0 divided by all eligible
        shipments, multiplied by 100. It represents SLA compliance and
        includes early deliveries.

        **Exact on-time percentage**

        Shipments with gap = 0 divided by all eligible shipments.

        **Average Delivery Delay**

        Mean positive delay gap among delayed eligible shipments only.

        **Late Delivery Risk Ratio**

        Mean of valid binary Late_delivery_risk labels multiplied by 100.
        It is an existing dataset label, not a new model prediction.

        **Date filter**

        The source dataset does not contain a genuine operational date
        column. Date filtering is blocked. No synthetic dates are created.

        **Geographic analysis**

        Regional and country visualizations use categorical summaries.
        Customer coordinates are not interpreted as destination coordinates.

        **Customer segment**

        Consumer, Corporate and Home Office are analyzed as supplied.
        Corporate is not automatically classified as premium or enterprise.

        **Shipping Mode Efficiency Index**

        Project-defined weighting: 50% SLA compliance, 30% inverse late
        risk and 20% relative delay-gap score.
        """
    )


# ============================================================
# APPLICATION FOOTER
# ============================================================

st.divider()
st.caption(
    "Global Supply Chain Operations | Delivery Performance and Logistics Efficiency"
)
st.caption(
    f"Source records: {len(data):,} | "
    f"Filtered records: {len(filtered_data):,} | "
    f"Eligible shipments: {len(eligible):,}"
)
