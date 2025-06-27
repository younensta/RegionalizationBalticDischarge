import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt


import abstract_models as am
import indicators as idcts
from base_models import *
from grouping_strategies import *
from neighboring_strategies import *


st.set_page_config(page_title="Data-Driven Regionalization Models for Surface Discharge")
st.title("Data-Driven Regionalization Models for Surface Discharge")

mode = st.sidebar.selectbox("Select Mode", ["Testing", "Prediction"])

if mode == "Testing":
    st.header("Testing Mode")
    

    #DATA UPLOAD AND CONFIGURATION
    with st.sidebar.expander("Data Configuration", expanded=True):
        data_source = st.radio(
            "Select Data Source",
            ("Upload your own data", "Use Baltic Sea GRDC dataset")
        )

        if data_source == "Use Baltic Sea GRDC dataset":
            st.info("Using the built-in Baltic Sea GRDC dataset.")
            uploaded_file = None  # Disable upload
        else:
            uploaded_file = st.file_uploader("Upload your data file (.csv)", type=["csv"])
            if uploaded_file is not None:
                st.success("File uploaded successfully!")
                df = pd.read_csv(uploaded_file)
                st.write("Data Preview:")
                st.dataframe(df.head())

        temporal_step = st.selectbox(
            "Select Temporal Step",
            ["Seasonal", "Monthly", "Yearly"]
        )

        def check_ask_for_column(df, default_value):
            if default_value in df.columns:
                st.success(f"Column '{default_value}' found in the dataframe. If it is not the correct column, please rename it in your data.")
            else:
                name = st.text_input(
                    f"Please enter the name of {default_value} column in your dataframe",
                    value=default_value
                )
                if name in df.columns:
                    st.success(f"Column '{name}' found in the dataframe.")
                    df.rename(columns={name: default_value}, inplace=True)
                else:
                    st.error(f"Column '{name}' not found in the dataframe. Please check your data.")



        if data_source == "Upload your own data" and uploaded_file is not None:
            st.write("Basin and Temporal Identification")
            if temporal_step == 'Yearly':
                check_ask_for_column(df, 'YEAR')
            
            elif temporal_step == 'Monthly':
                check_ask_for_column(df, 'MONTH')
                check_ask_for_column(df, 'YEAR')
            elif temporal_step == 'Seasonal':
                check_ask_for_column(df, 'SEASON')
                check_ask_for_column(df, 'YEAR')
            
            check_ask_for_column(df, 'ID')

            st.write("Discharge Column")
            check_ask_for_column(df, 'Q')
            st.write("Drainage Area Column (Models are using specific discharge)")
            check_ask_for_column(df, 'A')

            st.write("Optional: Basin Centroid Coordinates")
            st.info("Providing 'lat' and 'lon' columns (latitude and longitude) is optional, but required if you want to plot results on a geographical map.")
            
            show_coords = st.checkbox("Provide Basin Coordinate Columns (Required for some plots)", value=False)
            if show_coords:
                check_ask_for_column(df, 'lon')
                check_ask_for_column(df, 'lat')

            button = st.button("Validate Data")
            if button:
                if uploaded_file is not None:
                    if all(col in df.columns for col in ['ID', 'Q', 'A', 'YEAR'] if temporal_step == 'Yearly') and \
                    all(col in df.columns for col in ['ID', 'Q', 'A', 'YEAR', 'MONTH'] if temporal_step == 'Monthly') and \
                    all(col in df.columns for col in ['ID', 'Q', 'A', 'YEAR', 'SEASON'] if temporal_step == 'Seasonal') and\
                        (not show_coords or (show_coords and all(col in df.columns for col in ['lat', 'lon']))):
                        st.success("Data validation successful! You can proceed with the models.")
                        
                    else:
                        st.warning("Please ensure all requirments are fulfilled.")

        else:
            if temporal_step == 'Yearly':
                df = pd.read_csv('../DATA/DF/df_grdc.csv')
            elif temporal_step == 'Monthly':
                df = pd.read_csv('../DATA/DF/df_grdc_month.csv')
            elif temporal_step == 'Seasonal':
                df = pd.read_csv('../DATA/DF/df_grdc_season.csv')

    #MODEL SELECTION
    with st.sidebar.expander("Model Selection", expanded=True):
        st.write("Regression Models")
        st.write("This section allows you to select and configure regression models for your data. It will be used to predict the discharge based on the provided features on each group created.")
        model_option = st.selectbox(
            "Select Regression Model",
            ["Random Forest", "Multiple Linear Regression"]
        )

        pred = [col for col in df.columns if col not in ['ID', 'Q', 'A', 'YEAR', 'MONTH', 'SEASON', 'lat', 'lon']]
        predictors = st.multiselect(
            "Select Predictors",
            options=pred,
            default=pred[:]  # Default to all predictors
        )
        if model_option == "Random Forest":
            
            max_depth = st.number_input("Max Depth",min_value=1, max_value=100, value=10, step=5)
            n_trees = st.number_input("Number of Trees", min_value=1, max_value=200, value=30, step=5)
            model = LogRF(predictors, max_depth=max_depth, n_trees=n_trees)
        elif model_option == "Multiple Linear Regression":
            model = OlsLogMLR(predictors)
        

elif mode == "Prediction":
    st.header("Prediction Mode")
    # Add widgets and logic for prediction mode here
    st.write("You are in Prediction mode. Configure your prediction parameters below.")
    # Example: st.file_uploader, st.selectbox, etc.