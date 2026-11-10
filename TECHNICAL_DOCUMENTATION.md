# Technical Documentation

## Project structure

The project separates raw data, processed data, notebooks, reusable source
modules, dashboard code, reports, figures, and final submission artifacts.

## Data pipeline

1. Load the original logistics workbook.
2. Inspect and validate source columns and values.
3. Clean the dataset while preserving valid observations.
4. Calculate delivery gap as actual duration minus scheduled duration.
5. Classify delivery records as delayed, exactly on time, or early.
6. Calculate delivery performance by mode, region, market, and segment.
7. Calculate the five required KPIs.
8. Provide analytical outputs through the Streamlit dashboard and reports.

## Eligibility rules

A delivery record is eligible for delivery performance calculations when
actual and scheduled shipping durations are present and non-negative and the
shipment is not cancelled.

A scheduled duration of zero is valid and represents Same Day shipping.
Negative delivery gaps are early deliveries and are SLA compliant.

## KPI formulas

On-Time Delivery Rate:

SLA-compliant eligible records / eligible records * 100

SLA compliance means delivery gap <= 0.

Average Delivery Delay:

Mean of positive delivery gaps among delayed eligible records.

Late Delivery Risk Ratio:

Mean of valid binary Late_delivery_risk values * 100.

Shipping Mode Efficiency Index:

50% SLA compliance + 30% inverse late risk + 20% relative delay-gap score.

Regional Delay Index:

Regional average positive delay / maximum regional average positive delay * 100.

The two composite indices are project-defined metrics.

## Dashboard architecture

The Streamlit application provides four analytical modules:
delivery performance overview, delay risk analysis, shipping mode comparison,
and regional and market heatmaps.

## Data limitations

The source data does not contain a genuine operational date field. The project
does not create synthetic dates. Customer coordinates are not verified
shipment destination coordinates. Customer IDs are not order IDs. The
Late_delivery_risk field is an existing binary label.

## Verification

The final verification process checks source files, schema, data quality,
duration and gap calculations, analytical groupings, KPIs, dashboard source,
documentation, output files, and ZIP archive integrity.
