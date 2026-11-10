import numpy as np
import pandas as pd


def prepare_kpi_population(source_df):
    data = source_df.copy()

    required = [
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Delivery Status",
        "Late_delivery_risk",
        "Shipping Mode",
        "Order Region"
    ]

    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"Missing KPI columns: {missing}")

    data["Days for shipping (real)"] = pd.to_numeric(
        data["Days for shipping (real)"], errors="coerce"
    )
    data["Days for shipment (scheduled)"] = pd.to_numeric(
        data["Days for shipment (scheduled)"], errors="coerce"
    )
    data["Late_delivery_risk"] = pd.to_numeric(
        data["Late_delivery_risk"], errors="coerce"
    )

    data["Delivery Status Normalized"] = (
        data["Delivery Status"].astype("string").str.strip().str.lower()
    )
    data["Shipping Mode"] = (
        data["Shipping Mode"].astype("string").str.strip()
    )
    data["Order Region"] = (
        data["Order Region"].astype("string").str.strip()
    )

    data["Delay Gap"] = (
        data["Days for shipping (real)"]
        - data["Days for shipment (scheduled)"]
    )

    cancelled = data["Delivery Status Normalized"].eq("shipping canceled")

    valid_duration = (
        data["Days for shipping (real)"].notna()
        & data["Days for shipment (scheduled)"].notna()
        & data["Days for shipping (real)"].ge(0)
        & data["Days for shipment (scheduled)"].ge(0)
    )

    eligible = data.loc[~cancelled & valid_duration].copy()

    eligible["Delivery Classification"] = np.select(
        [
            eligible["Delay Gap"] > 0,
            eligible["Delay Gap"] == 0,
            eligible["Delay Gap"] < 0
        ],
        ["Delayed", "On-time", "Early"],
        default="Invalid"
    )

    eligible["SLA Compliant"] = eligible["Delay Gap"] <= 0
    eligible["Valid Late Risk"] = eligible["Late_delivery_risk"].isin([0, 1])

    eligible["Valid Shipping Mode"] = (
        eligible["Shipping Mode"].notna()
        & eligible["Shipping Mode"].ne("")
        & eligible["Shipping Mode"].ne("<NA>")
    )

    eligible["Valid Order Region"] = (
        eligible["Order Region"].notna()
        & eligible["Order Region"].ne("")
        & eligible["Order Region"].ne("<NA>")
    )

    return eligible


def calculate_required_kpis(source_df):
    eligible = prepare_kpi_population(source_df)

    if eligible.empty:
        raise ValueError("No eligible records available for KPI calculation.")

    total_eligible = len(eligible)

    on_time_rate = (
        eligible["SLA Compliant"].sum() / total_eligible * 100
    )

    delayed_gaps = eligible.loc[
        eligible["Delay Gap"] > 0, "Delay Gap"
    ]

    average_delivery_delay = (
        delayed_gaps.mean() if len(delayed_gaps) else 0.0
    )

    valid_risk = eligible.loc[
        eligible["Valid Late Risk"], "Late_delivery_risk"
    ]

    late_risk_ratio = (
        valid_risk.mean() * 100 if len(valid_risk) else np.nan
    )

    # Shipping Mode Efficiency Index

    mode_records = eligible.loc[
        eligible["Valid Shipping Mode"]
    ].copy()

    mode_rows = []

    if len(mode_records):
        for mode, group in mode_records.groupby("Shipping Mode"):
            valid_mode_risk = group.loc[
                group["Valid Late Risk"], "Late_delivery_risk"
            ]

            mode_rows.append({
                "Shipping Mode": mode,
                "Shipment Count": len(group),
                "SLA Compliance (%)": group["SLA Compliant"].mean() * 100,
                "Late Delivery Risk Ratio (%)": (
                    valid_mode_risk.mean() * 100
                    if len(valid_mode_risk) else np.nan
                ),
                "Average Positive Delay (Days)": (
                    group.loc[group["Delay Gap"] > 0, "Delay Gap"].mean()
                    if (group["Delay Gap"] > 0).any() else 0.0
                )
            })

    mode_table = pd.DataFrame(mode_rows)

    if len(mode_table):
        risk_values = mode_table["Late Delivery Risk Ratio (%)"]
        risk_reference = (
            risk_values.fillna(
                valid_risk.mean() * 100 if len(valid_risk) else 0.0
            )
        )

        mode_table["Inverse Late Risk Score"] = 100 - risk_reference

        delay_values = mode_table["Average Positive Delay (Days)"]
        delay_min = delay_values.min()
        delay_max = delay_values.max()

        if np.isclose(delay_max, delay_min):
            mode_table["Relative Delay Gap Score"] = 100.0
        else:
            mode_table["Relative Delay Gap Score"] = (
                (delay_max - delay_values) / (delay_max - delay_min) * 100
            )

        mode_table["Shipping Mode Efficiency Index"] = (
            0.50 * mode_table["SLA Compliance (%)"]
            + 0.30 * mode_table["Inverse Late Risk Score"]
            + 0.20 * mode_table["Relative Delay Gap Score"]
        )

        mode_table = mode_table.sort_values(
            "Shipping Mode Efficiency Index",
            ascending=False
        ).reset_index(drop=True)

        shipping_mode_index = mode_table[
            "Shipping Mode Efficiency Index"
        ].mean()

    else:
        shipping_mode_index = np.nan

    # Regional Delay Index

    region_records = eligible.loc[
        eligible["Valid Order Region"]
    ].copy()

    region_rows = []

    if len(region_records):
        for region, group in region_records.groupby("Order Region"):
            positive_gaps = group.loc[
                group["Delay Gap"] > 0, "Delay Gap"
            ]

            region_rows.append({
                "Order Region": region,
                "Shipment Count": len(group),
                "Delayed Count": int((group["Delay Gap"] > 0).sum()),
                "Average Positive Delay (Days)": (
                    positive_gaps.mean() if len(positive_gaps) else 0.0
                ),
                "Delayed Percentage (%)": (
                    (group["Delay Gap"] > 0).mean() * 100
                )
            })

    region_table = pd.DataFrame(region_rows)

    if len(region_table):
        highest_regional_delay = (
            region_table["Average Positive Delay (Days)"].max()
        )

        if highest_regional_delay > 0:
            region_table["Regional Delay Index"] = (
                region_table["Average Positive Delay (Days)"]
                / highest_regional_delay * 100
            )
        else:
            region_table["Regional Delay Index"] = 0.0

        region_table = region_table.sort_values(
            "Regional Delay Index",
            ascending=False
        ).reset_index(drop=True)

        regional_delay_index = region_table[
            "Regional Delay Index"
        ].mean()

    else:
        regional_delay_index = np.nan

    # Five Required KPI Values

    kpi_table = pd.DataFrame([
        {
            "KPI": "On-Time Delivery Rate (%)",
            "KPI Value": on_time_rate,
            "Unit": "%",
            "Calculation Population": total_eligible,
            "Interpretation": (
                "Higher values indicate a greater proportion of shipments "
                "meeting or beating scheduled shipping duration."
            )
        },
        {
            "KPI": "Average Delivery Delay (Days)",
            "KPI Value": average_delivery_delay,
            "Unit": "Days",
            "Calculation Population": len(delayed_gaps),
            "Interpretation": (
                "Lower values indicate shorter delays among delayed shipments. "
                "Only positive delivery gaps are included."
            )
        },
        {
            "KPI": "Late Delivery Risk Ratio",
            "KPI Value": late_risk_ratio,
            "Unit": "%",
            "Calculation Population": len(valid_risk),
            "Interpretation": (
                "Lower values indicate fewer eligible records labeled with "
                "Late_delivery_risk = 1."
            )
        },
        {
            "KPI": "Shipping Mode Efficiency Index",
            "KPI Value": shipping_mode_index,
            "Unit": "Index (0-100)",
            "Calculation Population": len(mode_records),
            "Interpretation": (
                "Higher values indicate better average shipping-mode efficiency "
                "under the defined weighted index."
            )
        },
        {
            "KPI": "Regional Delay Index",
            "KPI Value": regional_delay_index,
            "Unit": "Index (0-100)",
            "Calculation Population": len(region_records),
            "Interpretation": (
                "Higher values indicate greater relative average positive delay. "
                "The highest-delay region receives an index of 100."
            )
        }
    ])

    return {
        "eligible_records": eligible,
        "kpi_table": kpi_table,
        "shipping_mode_table": mode_table,
        "regional_table": region_table,
        "population_count": total_eligible
    }
