# AI-Powered Predictive Bed Demand Forecasting for Albion Care Network

## Overview

Hospital bed demand forecasting is critical for ensuring efficient healthcare service delivery, patient safety, and optimal resource utilisation. Traditional bed management often relies on historical averages and reactive decision-making, making it difficult to anticipate capacity constraints caused by emergency admissions, elective surgeries, seasonal demand, and varying patient lengths of stay.

This project develops an AI-powered predictive analytics solution for **Albion Care Network**, a leading UK private healthcare provider operating more than 39 hospitals nationwide. By integrating multiple operational datasets into a unified analytical platform, the project aims to forecast hospital bed demand at hourly, daily, and weekly intervals to support proactive capacity planning and operational decision-making.

---

# Business Problem

Albion Care Network faces several operational challenges, including:

- Reactive bed planning based on historical demand.
- Fluctuating emergency admissions.
- Planned elective surgeries competing for bed availability.
- Seasonal variations in patient demand.
- Variable patient lengths of stay.
- Fragmented operational data across multiple hospital information systems.
- Difficulty aligning staffing resources with anticipated bed occupancy.

The project addresses these challenges by integrating operational datasets into a unified forecasting pipeline capable of supporting data-driven hospital capacity planning.

---

# Project Objectives

The project aims to:

- Explore historical trends and seasonal patterns in hospital admissions and bed occupancy.
- Forecast hourly, daily, and weekly bed demand across hospital wards and bed types.
- Improve workforce planning by aligning staffing levels with anticipated occupancy.
- Simulate operational scenarios such as flu outbreaks, delayed discharges, and emergency admission surges.
- Develop an interactive decision-support dashboard for proactive capacity management.
- Continuously improve forecasting performance using actual occupancy outcomes.

---

# Data Sources

The project integrates five core operational datasets:

### Admissions & Discharges
Captures patient inflow, discharge activity, admission characteristics and length of stay, providing the primary drivers of bed demand.

### Bed Inventory & Occupancy
Records bed capacity, staffing availability, occupancy levels and closed beds across hospital wards.

### Emergency Department / Outpatient Arrivals
Represents short-term unplanned demand that influences immediate bed requirements.

### Elective Surgery Schedule
Captures planned admissions and expected length of stay resulting from scheduled procedures.

### Staffing & Resource Data
Provides workforce availability and safe staffing indicators used to assess whether forecasted demand can be operationally supported.

---

# Data Model Relationships

The operational datasets are integrated around common temporal and operational dimensions including:

- Date
- Time
- Hospital
- Ward
- Specialty
- Bed Type

Key relationships include:

- Admissions generate bed occupancy by introducing new patients into hospital wards.
- Discharges release occupied beds and influence future bed availability.
- Bed occupancy reflects the interaction between admissions, discharges and available capacity.
- Emergency department arrivals create short-term fluctuations in demand.
- Elective surgery schedules introduce planned admissions and expected bed utilisation.
- Staffing data provides operational context for forecasting hospital capacity.

Together, these relationships enable the development of predictive models capable of forecasting future bed occupancy while accounting for both planned and unplanned patient demand.

---

# Project Scope

The project follows an end-to-end data science workflow comprising:

## 1. Data Integration & Preparation

- Consolidate multiple operational datasets.
- Clean and standardise data.
- Handle missing values, duplicates and inconsistencies.
- Align all datasets to common hourly and daily time granularity.

## 2. Feature Engineering

Create predictive features including:

- Lagged occupancy
- Lagged admissions
- Rolling averages
- Length of Stay (LOS)
- Calendar and seasonal indicators
- Emergency demand features
- Elective surgery features

## 3. Forecasting & Predictive Modelling

The project will develop and compare several forecasting approaches, including:

- SARIMA / SARIMAX
- Random Forest
- XGBoost
- LightGBM
- LSTM

Forecasts will be generated at:

- Hourly level
- Daily level
- Weekly level

Model performance will be evaluated using:

- MAE
- RMSE
- MAPE

## 4. Scenario Simulation

Assess the operational impact of:

- Flu outbreaks
- Emergency admission surges
- Delayed discharges
- Staffing shortages

## 5. Decision Support Dashboard

The dashboard will present:

- Current and forecasted bed occupancy
- Admissions and discharges
- Staffing levels
- Capacity risk indicators

to support proactive operational decision-making.

## 6. Operational Integration

The forecasting system is intended to support:

- Hospital planning workflows
- Capacity monitoring
- Automated operational alerts
- Continuous model improvement using newly available operational data

---

# Expected Outcomes

The project aims to deliver:

- Improved hospital bed demand forecasting accuracy.
- Better alignment between staffing and patient demand.
- Reduced elective surgery cancellations.
- Enhanced patient flow and hospital resource utilisation.
- Proactive, data-driven capacity planning.
- Improved operational efficiency and patient care.

---

# Repository Structure

```text
Predictive Bed Demand/
│
├── data/
│   ├── raw/
│   └── processed_data/
│
├── images/
│
├── models/
│
├── notebooks/
│
├── outputs/
│
├── Reports/
│
├── src/
│
└── README.md
```

---

# Project Status

🚧 **Work in Progress**

The project is currently progressing through the following stages:

- Data cleaning and preprocessing
- Exploratory Data Analysis (EDA)
- Feature engineering
- Time-series forecasting
- Machine learning model development
- Model evaluation
- Dashboard development

---

# Author

**Susan Dangana**

MSc Data Analytics | Data Science Portfolio Project