# Analysis in Global Supply Chain Operations

## Project overview

This project analyzes delivery performance, delivery delays, shipping risk,
shipping mode efficiency, regional differences, market performance, and
customer segment exposure using the supplied logistics dataset.

## Main analytical outputs

- Cleaned logistics dataset
- Delivery gap calculations and classifications
- Overall delivery performance analysis
- Shipping mode efficiency analysis
- Regional and market analysis
- Customer segment analysis
- Five required logistics KPIs
- Streamlit dashboard
- Research paper
- Government executive summary

## Dataset

Raw source: `data/raw/APL_Logistics.xlsx`

Cleaned dataset: `data/processed/cleaned_logistics_data.csv`

Delivery gap dataset: `data/processed/delivery_gap_analysis.csv`

The current source contains 180,519 observations. The cleaned dataset contains
32 columns. The delivery gap analysis dataset contains 39 columns.

## Delivery gap definition

Delivery gap = actual shipping duration - scheduled shipping duration.

- Positive gap: delayed delivery
- Zero gap: exactly on-time delivery
- Negative gap: early delivery

SLA compliance is defined as delivery gap less than or equal to zero.
Cancelled shipments and invalid duration records are excluded from the eligible
delivery performance population. Valid zero scheduled-duration Same Day records
are retained.

## Required KPIs

1. On-Time Delivery Rate
2. Average Delivery Delay
3. Late Delivery Risk Ratio
4. Shipping Mode Efficiency Index
5. Regional Delay Index

The Shipping Mode Efficiency Index and Regional Delay Index are project-defined
analytical indices, not external industry standards.

## Dashboard

Run the dashboard from the project root:

```powershell
streamlit run app.py
```

The dashboard contains four modules:

- Delivery Performance Overview
- Delay Risk Analysis
- Shipping Mode Comparison
- Regional and Market Heatmaps

## Data limitations

The supplied dataset does not contain a genuine operational date field suitable
for a date-range selector. Synthetic dates must not be generated. Customer
coordinates must not be interpreted as verified shipment destination
coordinates. Customer IDs must not be treated as order IDs.

## Main folders

- `data`: source and processed datasets
- `notebooks`: analytical notebooks
- `src`: reusable analytical modules
- `dashboard`: dashboard-related resources
- `models`: model artifacts, if applicable
- `outputs`: tables, figures, reports, and forecasts
- `research_paper`: research paper deliverables
- `government_executive_summary`: executive summary deliverables
- `documentation`: project documentation
