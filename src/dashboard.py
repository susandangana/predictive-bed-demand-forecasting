#!/usr/bin/env python
# coding: utf-8

# # AI-Powered Predictive Bed Demand Forecasting for Albion Care Network

# ## Streamlit Dashboard for Predictive Bed Demand

# ### Load Required Libraries

# In[1]:


# =============================================================================
# IMPORT REQUIRED LIBRARIES
# =============================================================================
#
# Purpose:
# --------
# Import all Python libraries required to build the Hospital Bed Demand
# Forecasting Dashboard using Streamlit.
#
# These libraries provide functionality for:
#
# • Creating the interactive web dashboard.
# • Loading and manipulating hospital datasets.
# • Performing mathematical calculations.
# • Visualising hospital bed demand forecasts.
# • Evaluating XGBoost forecasting performance.
#
# =============================================================================


# =============================================================================
# STREAMLIT AND PATH
# =============================================================================
#
# Streamlit is used to create an interactive web application.
#
# It provides:
# • Dashboard layout
# • Sidebar filters
# • KPI cards
# • Interactive charts
# • Data tables
# • User controls
#
# =============================================================================

import streamlit as st

from pathlib import Path

# =============================================================================
# PROJECT ROOT
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent


# =============================================================================
# PANDAS
# =============================================================================
#
# Pandas is used for data loading, cleaning and manipulation.
#
# Functions include:
# • Reading CSV files
# • Filtering data
# • Aggregating forecasts
# • Grouping hospital records
# • Time-series processing
#
# =============================================================================

import pandas as pd


# =============================================================================
# NUMPY
# =============================================================================
#
# NumPy performs numerical computations throughout the dashboard.
#
# Examples include:
# • Mathematical calculations
# • Array operations
# • Statistical summaries
# • RMSE calculation
# • Occupancy percentage calculations
#
# =============================================================================

import numpy as np


# =============================================================================
# MATPLOTLIB
# =============================================================================
#
# Matplotlib is used to create customised visualisations that are embedded
# within the Streamlit dashboard.
#
# Example charts:
# • Actual vs Predicted Bed Demand
# • Forecast Error Distribution
# • Feature Importance Charts
#
# =============================================================================

import matplotlib.pyplot as plt


# =============================================================================
# PLOTLY EXPRESS
# =============================================================================
#
# Plotly Express creates interactive visualisations that users can:
#
# • Zoom
# • Pan
# • Hover over values
# • Export figures
#
# Used for:
# • Bar charts
# • Line charts
# • Scatter plots
# • Time-series visualisations
#
# =============================================================================

import plotly.express as px


# =============================================================================
# PLOTLY GRAPH OBJECTS
# =============================================================================
#
# Plotly Graph Objects provides greater flexibility for building advanced
# interactive charts.
#
# Used when:
# • Combining multiple traces
# • Customising layouts
# • Building professional dashboard figures
#
# =============================================================================

import plotly.graph_objects as go


# =============================================================================
# SCIKIT-LEARN EVALUATION METRICS
# =============================================================================
#
# These metrics evaluate the accuracy of the XGBoost hospital bed demand
# forecasting model.
#
# Mean Squared Error (MSE):
# Measures the average squared difference between actual and predicted
# bed demand.
#
# Root Mean Squared Error (RMSE):
# Square root of MSE.
# Indicates the typical prediction error in bed demand.
#
# Mean Absolute Error (MAE):
# Measures the average absolute forecasting error.
#
# Mean Absolute Percentage Error (MAPE):
# Measures the average forecasting error as a percentage.
#
# R² Score (Coefficient of Determination):
# Indicates how well the model explains the variation in hospital bed demand.
#
# Higher R² values indicate better forecasting performance.
#
# =============================================================================

from sklearn.metrics import (

    mean_squared_error,

    mean_absolute_error,

    mean_absolute_percentage_error,

    r2_score

)

# =============================================================================
# END OF LIBRARY IMPORTS
# =============================================================================
#
# All required libraries have now been imported successfully.
#
# The dashboard is ready to:
#
# • Load hospital forecasting data.
# • Apply interactive filters.
# • Calculate forecasting KPIs.
# • Visualise hospital bed demand.
# • Display operational insights.
#
# =============================================================================

# Import the datetime module to access current date and time functions
from datetime import datetime


# In[ ]:





# In[ ]:





# In[3]:


# =============================================================================
# STREAMLIT DASHBOARD
# XGBOOST HOSPITAL BED DEMAND FORECASTING
# =============================================================================
#
# PROJECT OBJECTIVE:
# ------------------
# Deploy the trained XGBoost forecasting model into an interactive dashboard
# environment using Streamlit.
#
# The dashboard transforms model predictions into operational insights that
# support hospital capacity management and resource planning.
#
# BUSINESS OBJECTIVES:
# --------------------
#
# 1. Monitor current and future hospital bed demand.
#
# 2. Compare actual bed utilisation against model forecasts.
#
# 3. Identify high-demand periods requiring additional resources.
#
# 4. Support staffing, ward allocation and capacity planning decisions.
#
# 5. Provide multiple forecasting views:
#
#       - Hourly operational monitoring
#       - Daily hospital planning
#       - Weekly resource allocation
#       - Monthly capacity forecasting
#       - Seasonal demand analysis
#
# =============================================================================

# =============================================================================
# STREAMLIT DASHBOARD
# XGBOOST HOSPITAL BED DEMAND FORECASTING
# =============================================================================
#
# PURPOSE:
# --------
# Create an interactive dashboard to visualise hospital bed demand forecasts
# generated by the XGBoost machine learning model.
#
# The dashboard allows hospital managers to:
#
# 1. Monitor actual vs predicted bed demand.
# 2. Analyse hourly demand patterns.
# 3. Review daily, weekly and monthly demand trends.
# 4. Understand seasonal changes in bed utilisation.
# 5. Support operational capacity planning decisions.
#
# DATA SOURCE:
# ------------
# The dashboard uses the final processed dataframe:
#
# final_xgboost_bed_demand_dashboard.csv
#
# Variables included:
#
# datetime              -> Forecast date/time
# actual_bed_demand     -> Observed hospital bed demand
# predicted_bed_demand  -> XGBoost forecasted demand
# hour                 -> Hourly demand pattern
# day                  -> Daily demand pattern
# week                 -> Weekly demand pattern
# month                -> Monthly demand pattern
# season               -> Seasonal demand pattern
#
# =============================================================================


# =============================================================================
# STEP 1: IMPORT REQUIRED LIBRARIES
# =============================================================================
#
# Import libraries required for:
#
# Streamlit:
# - Building interactive dashboard components.
#
# Pandas:
# - Loading and manipulating forecast data.
#
# NumPy:
# - Calculating evaluation metrics.
#
# Matplotlib:
# - Creating forecast visualisations.
#
# =============================================================================
# =============================================================================
# STEP 2: CONFIGURE STREAMLIT PAGE
# =============================================================================
#
# Define dashboard appearance and layout.
#
# wide layout provides more space for:
# - Forecast charts
# - KPI indicators
# - Data tables
#
# =============================================================================


st.set_page_config(

    page_title="Hospital Bed Demand Forecasting Dashboard",

    layout="wide"

)

# =============================================================================
# AUTOMATIC DATE UPDATE GENERATOR
# =============================================================================
#
# Purpose:
# This code automatically captures the current system date and formats it into
# a readable format. It can be used to track when the forecasting pipeline,
# model development, or analysis outputs were last updated.
#
# =============================================================================


# Retrieve the current date and format it as:

updated_date = datetime.now().strftime("%d %B %Y")
st.caption(f"**Last Updated:** {updated_date}")

# Display the latest update date for reporting and version tracking purposes
print(f"Date Updated: {updated_date}")

# =============================================================================
# Example Output:
# Date Updated: 22 July 2026
# =============================================================================
# =============================================================================
# STEP 3: LOAD XGBOOST FORECAST DATASET
# =============================================================================
#
# Load the final dataframe produced from the XGBoost forecasting pipeline.
#
# The @st.cache_data decorator improves dashboard performance by preventing
# Streamlit from reloading the dataset every time a user interacts with filters.
#
# =============================================================================


@st.cache_data

def load_data():

    DATA_PATH = BASE_DIR / "data" / "processed_data" / "xgboost_forecast_dashboard_data.csv"

    df = pd.read_csv(DATA_PATH)


    # Convert datetime column into datetime format

    df["datetime"] = pd.to_datetime(
    df["datetime"],
    dayfirst=True,
    errors="coerce"
)


    return df



# Load dataset

df = load_data()


# =============================================================================
# STEP 4: CREATE DASHBOARD HEADER
# =============================================================================
#
# Display the dashboard title and description.
#
# =============================================================================

# =============================================================================
# STEP 4: DASHBOARD HEADER
# =============================================================================
#
# Display the dashboard title inside a professionally styled banner.
#
# Design:
# -------
# • Navy blue background
# • White title
# • Light subtitle
# • Rounded corners
# • Centre alignment
#
# =============================================================================

st.markdown(
    """
    <div style="
        background-color:#001F54;
        padding:25px;
        border-radius:12px;
        text-align:center;
        margin-bottom:25px;
        box-shadow:0px 4px 10px rgba(0,0,0,0.25);
    ">

    <h1 style="
        color:white;
        margin-bottom:8px;
        font-size:38px;
        font-weight:700;
    ">
    AI-Powered Hospital Bed Demand Forecasting Dashboard
    </h1>

    <p style="
        color:#E8EEF7;
        font-size:18px;
        margin-top:0px;
    ">
    Predictive Analytics for Operational Planning, Capacity Management and Clinical Decision Support
    </p>

    </div>
    """,
    unsafe_allow_html=True
)


# =============================================================================
# STEP 5: CREATE INTERACTIVE SIDEBAR FILTERS
# =============================================================================
#
# PURPOSE
# -------
# The sidebar allows hospital managers and decision-makers to interactively
# filter the forecasting results before visualisation.
#
# Instead of displaying the entire dataset, users can focus on:
#
# • A particular hospital
# • A specific ward
# • A month of interest
# • A season
# • A custom date range
# • A forecasting horizon
#
# Using filters makes the dashboard:
#
# ✓ Faster
# ✓ Easier to explore
# ✓ More suitable for operational decision making
#
# =============================================================================


# =============================================================================
# CREATE SIDEBAR TITLE
# =============================================================================

st.markdown("""
<style>

/* Sidebar background */
[data-testid="stSidebar"]{
    background-color:#001f3f;
}

/* Sidebar text */
[data-testid="stSidebar"] *{
    color:white !important;
}

/* Selectbox labels */
[data-testid="stSidebar"] label{
    color:white !important;
    font-weight:bold;
}

</style>
""", unsafe_allow_html=True)
# Display a heading at the top of the sidebar to group all forecasting
# controls together.
#
# =============================================================================

st.sidebar.header("Forecast Filters")


# =============================================================================
# HOSPITAL FILTER
# =============================================================================
#
# Allows users to select one or multiple hospitals.
#
# Why?
# ----
# Hospital managers usually analyse only their own hospital rather than
# the entire healthcare system.
#
# The multiselect widget enables comparison across multiple hospitals.
#
# Default:
# All hospitals are selected.
#
# =============================================================================

selected_hospital = st.sidebar.multiselect(

    label="Hospital",

    options=sorted(
        df["hospital_name"].dropna().unique()
    ),

    default=sorted(
        df["hospital_name"].dropna().unique()
    )

)


# =============================================================================
# WARD FILTER
# =============================================================================
#
# Hospital demand often differs across wards.
#
# Example:
#
# • ICU
# • Medical
# • Surgical
# • Emergency
#
# Users can select one or several wards.
#
# =============================================================================

selected_ward = st.sidebar.multiselect(

    label="Ward",

    options=sorted(
        df["ward"].dropna().unique()
    ),

    default=sorted(
        df["ward"].dropna().unique()
    )

)


# =============================================================================
# MONTH FILTER
# =============================================================================
#
# Monthly filtering allows users to investigate seasonal demand changes.
#
# Example:
#
# January:
#     Winter pressures
#
# July:
#     Lower elective activity
#
# December:
#     Holiday demand
#
# =============================================================================

selected_month = st.sidebar.multiselect(

    label="Month",

    options=sorted(
        df["month"].dropna().unique()
    ),

    default=sorted(
        df["month"].dropna().unique()
    )

)


# =============================================================================
# SEASON FILTER
# =============================================================================
#
# Hospital demand varies considerably by season.
#
# Example:
#
# Winter
#     Higher respiratory admissions
#
# Summer
#     Trauma and accident cases
#
# Spring / Autumn
#     Transitional demand patterns
#
# =============================================================================

selected_season = st.sidebar.multiselect(

    label="Season",

    options=sorted(
        df["season"].dropna().unique()
    ),

    default=sorted(
        df["season"].dropna().unique()
    )

)


# =============================================================================
# DATE RANGE FILTER
# =============================================================================
#
# Users can focus on a specific forecasting period.
#
# Examples:
#
# • One week
# • One month
# • Winter surge
# • Christmas period
#
# The minimum and maximum available dates are automatically obtained from
# the dataset.
#
# =============================================================================

min_date = df["datetime"].min().date()

max_date = df["datetime"].max().date()


selected_dates = st.sidebar.date_input(

    label="Date Range",

    value=(min_date, max_date),

    min_value=min_date,

    max_value=max_date

)


# =============================================================================
# FORECAST HORIZON FILTER
# =============================================================================
#
# Users may wish to view predictions at different planning levels.
#
# Hourly
# -------
# Supports operational monitoring.
#
# Daily
# -----
# Supports ward planning.
#
# Weekly
# -------
# Supports staffing allocation.
#
# Monthly
# --------
# Supports strategic capacity planning.
#
# =============================================================================

forecast_horizon = st.sidebar.selectbox(

    label="Forecast Horizon",

    options=[

        "Hourly",

        "Daily",

        "Weekly",

        "Monthly"

    ]

)


# =============================================================================
# APPLY FILTER BUTTON
# =============================================================================
#
# Rather than updating the dashboard after every selection,
# all filters are applied simultaneously when the user clicks
# the Apply Filters button.
#
# Benefits:
#
# ✓ Faster dashboard performance
# ✓ Better user experience
# ✓ Reduces unnecessary recalculations
#
# =============================================================================

apply_filters = st.sidebar.button(

    "🔍 Apply Filters"

)


# =============================================================================
# APPLY USER-SELECTED FILTERS
# =============================================================================
#
# If the Apply Filters button is clicked:
#
# Step 1
# -------
# Create a copy of the original dataset.
#
# Step 2
# -------
# Apply each selected filter sequentially.
#
# Step 3
# -------
# Return only observations matching all selected conditions.
#
# =============================================================================

if apply_filters:

    # =========================================================================
    # Create a working copy of the dataset
    # =========================================================================

    filtered_df = df.copy()


    # =========================================================================
    # Filter Hospital
    # =========================================================================

    filtered_df = filtered_df[

        filtered_df["hospital_name"].isin(

            selected_hospital

        )

    ]


    # =========================================================================
    # Filter Ward
    # =========================================================================

    filtered_df = filtered_df[

        filtered_df["ward"].isin(

            selected_ward

        )

    ]


    # =========================================================================
    # Filter Month
    # =========================================================================

    filtered_df = filtered_df[

        filtered_df["month"].isin(

            selected_month

        )

    ]


    # =========================================================================
    # Filter Season
    # =========================================================================

    filtered_df = filtered_df[

        filtered_df["season"].isin(

            selected_season

        )

    ]


    # =========================================================================
    # Filter Date Range
    # =========================================================================
    #
    # Only execute when both start and end dates are supplied.
    #
    # =========================================================================

    if len(selected_dates) == 2:

        start_date = pd.to_datetime(

            selected_dates[0]

        )

        end_date = pd.to_datetime(

            selected_dates[1]

        )


        filtered_df = filtered_df[

            (filtered_df["datetime"] >= start_date)

            &

            (filtered_df["datetime"] <= end_date)

        ]


# =============================================================================
# Apply Filters buttom
# =============================================================================
#
# If the Apply Filters button has not yet been clicked,
# display the complete dataset.
#
# This ensures that the dashboard loads immediately when opened.
#
# =============================================================================

else:

    filtered_df = df.copy()


# =============================================================================
# DISPLAY FILTER SUMMARY
# =============================================================================
#
# Show the number of observations remaining after filtering.
#
# This provides immediate feedback to users and confirms that the
# selected filters have been applied successfully.
#
# =============================================================================

st.sidebar.markdown("---")

st.sidebar.success(

    f"Filtered Records: {len(filtered_df):,}"

)

# =============================================================================
# AGGREGATE DATA ACCORDING TO SELECTED FORECAST HORIZON
# =============================================================================
#
# Purpose:
# --------
# Hospital managers often require forecasts at different time resolutions
# depending on the planning decision being made.
#
# This section aggregates the filtered dataset based on the forecast horizon
# selected from the sidebar.
#
# Available Forecast Horizons:
#
# • Hourly
#     Displays every hourly observation.
#     Suitable for operational bed management during the day.
#
# • Daily
#     Aggregates predictions by calendar day.
#     Useful for admission and discharge planning.
#
# • Weekly
#     Aggregates predictions by ISO week number.
#     Useful for workforce planning and ward allocation.
#
# • Monthly
#     Aggregates predictions by month.
#     Useful for long-term strategic capacity planning.
#
# Aggregation Method:
# -------------------
# Both Actual Bed Demand and Predicted Bed Demand are summed over the selected
# time period so that total demand can be compared.
#
# The resulting dataframe (dashboard_df) is used throughout the dashboard
# to generate charts and KPI summaries.
#
# =============================================================================


# =============================================================================
# HOURLY FORECAST
# =============================================================================
#
# No aggregation is performed.
#
# Each row represents one hourly observation.
#
# Business Use:
# • Real-time monitoring
# • Shift planning
# • Emergency demand tracking
#
# =============================================================================

if forecast_horizon == "Hourly":

    dashboard_df = filtered_df.copy()


# =============================================================================
# DAILY FORECAST
# =============================================================================
#
# Aggregate the data by calendar date.
#
# The datetime column is converted to its date component before grouping.
#
# Business Use:
# • Daily admission planning
# • Daily discharge planning
# • Bed occupancy monitoring
#
# =============================================================================

elif forecast_horizon == "Daily":

    dashboard_df = (

        filtered_df

        # Group observations by calendar date
        .groupby(

            filtered_df["datetime"].dt.date

        )

        # Sum actual and predicted demand for each day
        [[

            "actual_bed_demand",

            "predicted_bed_demand"

        ]]

        .sum()

        # Convert grouped index back into a dataframe
        .reset_index()

        # Rename the date column for clarity
        .rename(

            columns={

                "datetime": "Date"

            }

        )

    )


# =============================================================================
# WEEKLY FORECAST
# =============================================================================
#
# Aggregate observations by ISO week number.
#
# ISO weeks provide a consistent weekly calendar throughout the year.
#
# Business Use:
# • Weekly staffing decisions
# • Ward allocation
# • Capacity planning
#
# =============================================================================

elif forecast_horizon == "Weekly":

    dashboard_df = (

        filtered_df

        # Group observations by ISO week number
        .groupby(

            filtered_df["datetime"].dt.isocalendar().week

        )

        # Sum demand within each week
        [[

            "actual_bed_demand",

            "predicted_bed_demand"

        ]]

        .sum()

        # Convert grouped index back into a dataframe
        .reset_index()

        # Rename the grouping column
        .rename(

            columns={

                "week": "Week"

            }

        )

    )


# =============================================================================
# MONTHLY FORECAST
# =============================================================================
#
# Aggregate observations by month.
#
# Business Use:
# • Strategic resource planning
# • Seasonal demand monitoring
# • Budget planning
#
# =============================================================================

elif forecast_horizon == "Monthly":

    dashboard_df = (

        filtered_df

        # Group observations by month
        .groupby(

            "month"

        )

        # Sum demand within each month
        [[

            "actual_bed_demand",

            "predicted_bed_demand"

        ]]

        .sum()

        # Convert grouped index back into a dataframe
        .reset_index()

    )
#================================================================================

# =============================================================================
# STEP 6: DISPLAY KEY PERFORMANCE INDICATORS (KPIs)
# =============================================================================
# dashboard_df now contains the data aggregated at the selected
# forecast horizon and will be used in:
#
# • KPI Cards
# • Actual vs Predicted Charts
# • Trend Analysis
# • Forecast Evaluation
# • Capacity Planning Visualisations
#
# =============================================================================

# Purpose:
# --------
# Key Performance Indicators (KPIs) provide hospital managers with a quick
# summary of the current operational status and forecasting performance.
#
# Instead of analysing multiple charts, users can immediately understand:
#
# • Current hospital bed utilisation
# • Forecasted bed demand
# • Overall occupancy level
# • Forecast accuracy
# • Peak bed demand
#
# These indicators support rapid operational and strategic decision-making.
#
# Dashboard KPIs:
# ---------------
#
# 🛏 Current Beds
#     Average observed hospital bed demand.
#
# 🔮 Predicted Beds
#     Average number of beds forecasted by the XGBoost model.
#
# 📈 Occupancy %
#     Percentage of hospital bed capacity currently occupied.
#
# 📉 RMSE
#     Root Mean Squared Error.
#     Measures the average magnitude of prediction errors.
#
# 📊 MAE
#     Mean Absolute Error.
#     Represents the average absolute forecasting error.
#
# 📌 MAPE
#     Mean Absolute Percentage Error.
#     Measures prediction accuracy as a percentage.
#
# 🎯 R² Score
#     Indicates how well the model explains variation in bed demand.
#
# 🚨 Peak Demand
#     Highest observed bed demand within the selected filters.
#
# =============================================================================


st.subheader(

    "Hospital Bed Demand Forecast Summary"

)



# =============================================================================
# STEP 6.1: CALCULATE KPI VALUES
# =============================================================================
#
# Calculate summary statistics and forecasting metrics from the filtered data.
#
# These values automatically update whenever the user changes dashboard
# filters (Hospital, Ward, Season, Date Range, etc.).
#
# =============================================================================


# =============================================================================
# Current Bed Demand
# =============================================================================
#
# Calculate the average observed hospital bed demand.
#
# This represents the average number of beds occupied during the selected
# analysis period.
#
# =============================================================================

current_beds = (

    filtered_df["actual_bed_demand"]

    .mean()

)



# =============================================================================
# Predicted Bed Demand
# =============================================================================
#
# Calculate the average bed demand predicted by the XGBoost model.
#
# Comparing this with Current Beds provides a quick assessment of model
# performance.
#
# =============================================================================

predicted_beds = (

    filtered_df["predicted_bed_demand"]

    .mean()

)



# =============================================================================
# Peak Bed Demand
# =============================================================================
#
# Identify the highest observed hospital bed demand.
#
# Business Use:
# • Capacity planning
# • Surge management
# • Emergency preparedness
#
# =============================================================================

peak_demand = (

    filtered_df["actual_bed_demand"]

    .max()

)



# =============================================================================
# Occupancy Percentage
# =============================================================================
#
# Calculate the percentage of available hospital beds that are occupied.

# =============================================================================
# Calculate Occupancy Rate (%)
# =============================================================================

if "total_beds" in filtered_df.columns:

    occupancy_rate = (
        filtered_df["actual_bed_demand"].mean()
        / filtered_df["total_beds"].mean()
    ) * 100

elif (
    "occupied_beds" in filtered_df.columns
    and "staffed_beds" in filtered_df.columns
):

    occupancy_rate = (
        filtered_df["occupied_beds"].mean()
        / filtered_df["staffed_beds"].mean()
    ) * 100

else:
    occupancy_rate = np.nan



# =============================================================================
# Forecast Accuracy Metrics
# =============================================================================
#
# Calculate model evaluation statistics.
#
# These metrics quantify forecasting performance.
#
# =============================================================================


# =============================================================================
# LOAD XGBOOST MODEL PERFORMANCE METRICS
# =============================================================================

DATA_PATH = BASE_DIR / "data" / "processed_data" / "xgboost_model_performance.csv"

performance_df = pd.read_csv(DATA_PATH)

def get_metric(metric_name):

    result = performance_df.loc[
        performance_df["Metric"]
        .astype(str)
        .str.strip()
        .str.upper()
        ==
        metric_name.upper(),
        "Value"
    ]

    if not result.empty:
        return result.iloc[0]

    return None


rmse = get_metric("RMSE")

mae = get_metric("MAE")

mape = get_metric("MAPE (%)")

r2 = get_metric("R²")
# =============================================================================
# STEP 6.2: DISPLAY KPI CARDS
# =============================================================================
#
# Display KPI cards in two rows.
#
# The first row summarises hospital operational performance.
#
# The second row summarises forecasting accuracy.
#
# =============================================================================



# =============================================================================
# CREATE KPI CARD LAYOUT
# =============================================================================
# =============================================================================
# KPI CARDS
# =============================================================================

# =============================================================================
# KPI CARD STYLING
# =============================================================================
# Creates professional dashboard KPI cards:
# - Visible white background
# - Soft shadow effect
# - Light blue text colour
# =============================================================================

# =============================================================================
# KPI CARD STYLING
# =============================================================================
# Purpose:
# --------
# Apply custom CSS styling to all KPI cards displayed on the dashboard.
#
# Design Features:
# - White background
# - Rounded corners
# - Soft shadow effect
# - Light blue labels and values
# - Professional healthcare dashboard appearance
# =============================================================================

# =============================================================================
# KPI CARD STYLING
# =============================================================================
# Purpose:
# --------
# Custom CSS for displaying dashboard KPI cards.
#
# Features:
# - White background
# - Rounded corners
# - Soft shadow
# - Light blue text
# - Hover animation
# =============================================================================

st.markdown("""
<style>

/* =============================================================================
   KPI CARD
   ============================================================================= */

.kpi-card{

    background:#FFFFFF;
    border:1px solid #D6EAF8;
    border-radius:15px;

    padding:20px;

    text-align:center;

    margin-bottom:15px;

    box-shadow:0 4px 12px rgba(0,0,0,0.15);

    transition:all 0.3s ease;

}


/* Hover effect */

.kpi-card:hover{

    transform:translateY(-4px);

    box-shadow:0 8px 20px rgba(0,0,0,0.25);

}


/* KPI Title */

.kpi-title{

    color:#5DADE2;

    font-size:16px;

    font-weight:600;

    margin-bottom:10px;

}


/* KPI Value */

.kpi-value{

    color:#5DADE2;

    font-size:30px;

    font-weight:700;

    line-height:1.2;

}

</style>
""", unsafe_allow_html=True)


# =============================================================================
# OPERATIONAL KPI CARDS
# =============================================================================

col1, col2, col3, col4 = st.columns(4)

operational_kpis = [

    ("Current Beds", f"{current_beds:,.0f}"),

    ("Predicted Beds", f"{predicted_beds:,.0f}"),

    ("Occupancy Rate", f"{occupancy_rate:.1f}%"),

    ("Peak Demand", f"{peak_demand:,.0f}")

]

for col, (title, value) in zip(
    [col1, col2, col3, col4],
    operational_kpis
):

    with col:

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# MODEL PERFORMANCE KPI CARDS
# =============================================================================

col5, col6, col7, col8 = st.columns(4)

performance_kpis = [

    ("RMSE", f"{rmse:.2f}"),

    ("MAE", f"{mae:.2f}"),

    ("MAPE", f"{mape:.2f}%"),

    ("R² Score", f"{r2:.3f}")

]

for col, (title, value) in zip(
    [col5, col6, col7, col8],
    performance_kpis
):

    with col:

        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-title">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)
# Create eight equally sized columns.
#
# Each column contains one KPI card, providing a clean dashboard layout that
# allows hospital managers to compare operational metrics at a glance.
#
# =============================================================================

# =============================================================================
# END OF KPI SECTION
# =============================================================================
#
# The KPI cards provide an executive summary of hospital operational
# performance and forecasting accuracy, enabling decision-makers to assess
# current bed demand and model performance before exploring the detailed
# visualisations in the remainder of the dashboard.
#
# =============================================================================
# =============================================================================
# STEP 7: MODEL PERFORMANCE VISUALISATION
# =============================================================================
#
# Purpose
# -------
# This section provides a visual assessment of the predictive performance of
# the XGBoost model.
#
# Two complementary charts are displayed side-by-side:
#
# Left Panel:
#     Compares the actual hospital bed demand against the values predicted
#     by the XGBoost model.
#
#     This allows users to visually assess:
#         • prediction accuracy
#         • periods of under-prediction
#         • periods of over-prediction
#         • how closely predictions follow actual demand
#
# Right Panel:
#     Displays the twenty most influential predictor variables used by the
#     XGBoost model.
#
#     Higher importance scores indicate variables that contribute more to the
#     prediction of hospital bed demand.
#
# =============================================================================

# Create two equally sized dashboard columns
col1, col2 = st.columns(2)

# =============================================================================
# LEFT PANEL
# ACTUAL VS PREDICTED BED DEMAND
# =============================================================================

with col1:

    st.subheader("📈 Actual vs XGBoost Predicted Bed Demand")

    # Create matplotlib figure
    fig, ax = plt.subplots(figsize=(10,5))

    # Plot actual observations
    ax.plot(
        filtered_df["datetime"],
        filtered_df["actual_bed_demand"],
        label="Actual",
        linewidth=2
    )

    # Plot XGBoost predictions
    ax.plot(
        filtered_df["datetime"],
        filtered_df["predicted_bed_demand"],
        label="Predicted",
        linewidth=2
    )

    # Label chart axes
    ax.set_xlabel("Date")
    ax.set_ylabel("Hospital Bed Demand")

    # Display legend
    ax.legend()

    # Improve readability
    ax.grid(alpha=0.3)

    # Display figure in Streamlit
    st.pyplot(fig)


# =============================================================================
# RIGHT PANEL
# TOP 20 MOST IMPORTANT FEATURES
# =============================================================================

with col2:

    st.subheader("⭐ Top 20 Most Important Features")

    # Load feature importance generated during model training
    DATA_PATH = BASE_DIR / "data" / "processed_data" / "xgboost_feature_importance.csv"

    feature_importance = pd.read_csv(DATA_PATH)

    # Select the twenty most important variables
    feature_importance = (
        feature_importance
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(20)
    )

    # Create horizontal bar chart
    fig, ax = plt.subplots(figsize=(8,5))

    ax.barh(
        feature_importance["Feature"],
        feature_importance["Importance"]
    )

    # Highest importance displayed at the top
    ax.invert_yaxis()

    ax.set_xlabel("Importance Score")
    ax.set_title("Top 20 Predictors")

    st.pyplot(fig)



# =============================================================================
# STEP 8: MODEL DIAGNOSTICS
# =============================================================================
#
# Purpose
# -------
# Evaluate model errors (residuals).
#
# Residual:
#
#     Actual Bed Demand − Predicted Bed Demand
#
# A well-performing forecasting model should produce residuals that:
#
# • are centred around zero
# • have no systematic bias
# • contain relatively few large errors
#
# Two diagnostic plots are displayed:
#
# Left:
#     Histogram of forecast errors.
#
# Right:
#     Boxplot showing spread and outliers.
#
# =============================================================================

# Calculate forecast errors
forecast_errors = (
    dashboard_df["actual_bed_demand"]
    - dashboard_df["predicted_bed_demand"]
)

# Create two dashboard columns
col1, col2 = st.columns(2)

# =============================================================================
# FORECAST ERROR DISTRIBUTION
# =============================================================================

with col1:

    st.subheader("📉 Forecast Error Distribution")

    fig_error = px.histogram(

        x=forecast_errors,

        nbins=50,

        title="Forecast Error Distribution"

    )

    # Reference line representing perfect prediction
    fig_error.add_vline(

        x=0,

        line_dash="dash",

        line_color="red"

    )

    fig_error.update_layout(

        template="plotly_white",

        xaxis_title="Forecast Error",

        yaxis_title="Frequency",

        height=420

    )

    st.plotly_chart(

        fig_error,

        use_container_width=True

    )


# =============================================================================
# RESIDUAL HISTOGRAM
# =============================================================================
# Purpose:
# --------
# Visualises the distribution of forecast residuals (prediction errors).
#
# Residual = Actual Bed Demand - Predicted Bed Demand
#
# Interpretation:
# - Residuals centred around zero indicate unbiased predictions.
# - A wider spread indicates higher forecast uncertainty.
# - Extreme bars indicate periods with large prediction errors.
# =============================================================================


with col2:

    # Display section title for the residual analysis chart
    st.subheader("📊 Residual Error Histogram")


    # Create histogram showing the frequency distribution
    # of forecast errors produced by the XGBoost model
    fig_hist = px.histogram(

        # Residual error values to visualise
        x=forecast_errors,

        # Number of bins used to group residual values
        # Higher values provide more detailed distribution
        nbins=30,

        # Chart title displayed above the figure
        title="Residual Error Distribution"

    )


    # Customise chart appearance and layout
    fig_hist.update_layout(

        # Use clean white background suitable for dashboard presentation
        template="plotly_white",

        # Label for horizontal axis representing forecast errors
        xaxis_title="Forecast Error",

        # Label for vertical axis representing frequency counts
        yaxis_title="Frequency",

        # Set chart height for dashboard consistency
        height=420,

        # Add small spacing between histogram bars
        bargap=0.05

    )


    # Render interactive Plotly histogram inside Streamlit dashboard
    st.plotly_chart(

        fig_hist,

        # Automatically resize chart to available column width
        use_container_width=True

    )



# =============================================================================
# STEP 9: TIME HORIZON FORECASTS
# =============================================================================
#
# Purpose
# -------
# Hospital managers require forecasts across multiple operational planning
# horizons rather than only hourly predictions.
#
# This section summarises model predictions across:
#
# • Hourly
# • Daily
# • Weekly
# • Monthly
# • Seasonal
#
# Streamlit tabs provide an interactive interface for switching between
# different forecasting horizons.
#
# =============================================================================

st.markdown("---")

st.subheader("📅 Time Horizon Forecasts")

# Create interactive tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Hourly",
        "Daily",
        "Weekly",
        "Monthly",
        "Seasonal"
    ]
)

# =============================================================================
# HOURLY FORECAST
# =============================================================================

with tab1:

    # Average demand by hour
    hourly_analysis = (

        filtered_df

        .groupby("hour")[

            [
                "actual_bed_demand",
                "predicted_bed_demand"
            ]

        ]

        .mean()

    )

    st.line_chart(hourly_analysis)


# =============================================================================
# DAILY FORECAST
# =============================================================================

with tab2:

    daily_analysis = (

        filtered_df

        .groupby("day")[

            [
                "actual_bed_demand",
                "predicted_bed_demand"
            ]

        ]

        .mean()

    )

    st.line_chart(daily_analysis)


# =============================================================================
# WEEKLY FORECAST
# =============================================================================

with tab3:

    weekly_analysis = (

        filtered_df

        .groupby("week_number")[

            [
                "actual_bed_demand",
                "predicted_bed_demand"
            ]

        ]

        .mean()

    )

    st.bar_chart(weekly_analysis)


# =============================================================================
# MONTHLY FORECAST
# =============================================================================

with tab4:

    monthly_analysis = (

        filtered_df

        .groupby("month")[

            [
                "actual_bed_demand",
                "predicted_bed_demand"
            ]

        ]

        .mean()

    )

    st.bar_chart(monthly_analysis)


# =============================================================================
# SEASONAL FORECAST
# =============================================================================

with tab5:

    season_analysis = (

        filtered_df

        .groupby("season")[

            [
                "actual_bed_demand",
                "predicted_bed_demand"
            ]

        ]

        .mean()

    )

    st.bar_chart(season_analysis)



# =============================================================================
# STEP 10: FORECAST DATA TABLE
# =============================================================================
#
# Purpose
# -------
# Display the underlying forecast records used by the dashboard.
#
# Users can:
#
# • inspect predictions
# • compare actual and predicted demand
# • validate model outputs
# • export results if required
#
# =============================================================================

st.markdown("---")

st.subheader("📋 Forecast Records")

st.dataframe(

    filtered_df,

    use_container_width=True

)


# =============================================================================
# DASHBOARD FOOTER
# =============================================================================

st.markdown("---")

st.markdown(
    """
    <div style="
        text-align:center;
        background-color:#0B1F3A;
        color:white;
        padding:15px;
        border-radius:8px;
        font-size:14px;
    ">

    <b>XGBoost Machine Learning Prediction System</b><br><br>

    This interactive dashboard presents XGBoost-based forecasts of hospital
    bed demand for Albion Care Network. The predictive model combines
    historical demand patterns, operational hospital indicators, and
    engineered time-series features to support evidence-based capacity
    planning, resource allocation, and operational decision-making.

    <br><br>

    Hospital Bed Demand Forecasting Using Explainable Machine Learning<br>
    Last Updated: 27 July 2026

    </div>
    """,
    unsafe_allow_html=True
)
# =============================================================================
# END OF DASHBOARD PIPELINE
# =============================================================================
#
# The Streamlit dashboard is now ready to support:
#
# - Hospital bed monitoring
# - Forecast evaluation
# - Capacity planning
# - Operational decision making
#
# =============================================================================


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




