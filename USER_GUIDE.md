# User Guide

## Dashboard access

From the project root, launch:

```powershell
streamlit run app.py
```

Open the local URL displayed by Streamlit.

## Dashboard modules

### Delivery Performance Overview

Use this module to review delivery performance, eligible shipment population,
delivery gap distribution, and SLA compliance.

### Delay Risk Analysis

Use this module to examine the supplied late-delivery risk indicator and
delivery delay patterns. The existing risk field is a dataset label and should
not be described as a newly trained prediction.

### Shipping Mode Comparison

Compare Standard Class, Second Class, First Class, and Same Day using the
available shipping mode performance measures.

### Regional and Market Heatmaps

Compare regional and market delivery performance using categorical
aggregations. Customer latitude and longitude are not verified shipment
destination coordinates and should not be used to create shipment destination
maps.

## Filters and downloads

Use the available shipping mode, region, market, and customer segment filters.
Use report download controls where provided.

## KPI interpretation

- On-Time Delivery Rate: share of eligible records with delivery gap <= 0.
- Average Delivery Delay: average positive delivery gap among delayed records.
- Late Delivery Risk Ratio: average valid binary late-risk label multiplied by 100.
- Shipping Mode Efficiency Index: project-defined composite index.
- Regional Delay Index: regional average positive delay relative to the highest
  regional average positive delay.

## Date filter limitation

The supplied dataset has no genuine operational date field. A date-range
selector is therefore unavailable until a valid date field is supplied.

## Interpretation cautions

Cancelled shipments and invalid duration records are excluded from delivery
performance denominators. Valid zero scheduled-duration Same Day records are
retained. Corporate is a customer segment and is not automatically equivalent
to premium or enterprise customers.
