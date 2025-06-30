import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

import abstract_models as am
import indicators as idcts
import base_models as bm
import grouping_strategies as gs
import neighboring_strategies as ns


st.session_state.mode = "Testing" # Default but can be "Prediction" also
if "step" not in st.session_state:
    st.session_state.step = "DATA_LOAD_PAGE"
if "train_df" not in st.session_state:
    st.session_state.train_df = None
if "models" not in st.session_state:
    st.session_state.models = []
if "indicators" not in st.session_state:
    st.session_state.indicators = None
if "show_coords" not in st.session_state:
    st.session_state.show_coords = False
if "nb_strategies" not in st.session_state:
    st.session_state.nb_strategies = 0
if "data_from_user_source" not in st.session_state:
    st.session_state.data_from_user_source = False  # Default data source
if "temp_step" not in st.session_state:
    st.session_state.temp_step = None  # Default temporal step
if "nb_models" not in st.session_state:
    st.session_state.nb_models = 0  # Default number of models
st.set_page_config(page_title="Data Driven Regionalization Models for Surface Discharge")
st.title("Data Driven Regionalization Models for Surface Discharge")
   


with st.sidebar:
    st.header("Mode Selection")
    mode = st.radio("Select Mode", ["Testing", "Prediction"], index=0)
    st.session_state.mode = mode

    if mode == "Testing":
        st.write("You are in Testing Mode. You can test the models with your own data or the built-in dataset.")
        if st.session_state.train_df is not None:
            st.write("Training Data is ready.")
            st.write("Preview")
            st.dataframe(st.session_state.train_df.head())
    else:
        st.write("You are in Prediction Mode. You can use the models to predict discharge based on your data.")
    
    st.write("---")
    st.write("This app allows you to test and use data-driven regionalization models for surface discharge. You can upload your own data or use the built-in Baltic Sea GRDC dataset.")


# DATA_LOAD_PAGE
if st.session_state.step == "DATA_LOAD_PAGE":
    st.header("Data Load and Preparation")
   
    data_source = st.radio(
            "Select Data Source",
            ("Use Baltic Sea GRDC dataset", "Upload your own data")
        )

    if data_source == "Use Baltic Sea GRDC dataset":
        st.info("Using the built-in Baltic Sea GRDC dataset.")
        st.session_state.data_from_user_source = False
        uploaded_file = None  # Disable upload
    else:
        uploaded_file = st.file_uploader("Upload your data file (.csv)", type=["csv"])
        if uploaded_file is not None:
            st.success("File uploaded successfully!")
            train_df = pd.read_csv(uploaded_file)
            st.session_state.data_from_user_source = True
            st.write("Data Preview:")
            st.dataframe(train_df.head())

    temporal_step = st.selectbox(
        "Select Temporal Step",
        ["Seasonal", "Monthly", "Yearly"]
    )

    def check_ask_for_column(train_df, default_value):
        if default_value in train_df.columns:
            st.success(f"Column '{default_value}' found in the dataframe. If it is not the correct column, please rename it in your data.")
        else:
            name = st.text_input(
                f"Please enter the name of {default_value} column in your dataframe",
                value=default_value
            )
            if name in train_df.columns:
                st.success(f"Column '{name}' found in the dataframe.")
                train_df.rename(columns={name: default_value}, inplace=True)
            else:
                st.error(f"Column '{name}' not found in the dataframe. Please check your data.")



    if data_source == "Upload your own data" and uploaded_file is not None:
        st.write("Basin and Temporal Identification")
        if temporal_step == 'Yearly':
            check_ask_for_column(train_df, 'YEAR')
            st.session_state.temp_step = 'YEAR'
        
        elif temporal_step == 'Monthly':
            check_ask_for_column(train_df, 'MONTH')
            check_ask_for_column(train_df, 'YEAR')
            st.session_state.temp_step = 'MONTH'
        elif temporal_step == 'Seasonal':
            check_ask_for_column(train_df, 'SEASON')
            check_ask_for_column(train_df, 'YEAR')
            st.session_state.temp_step = 'SEASON'
        
        check_ask_for_column(train_df, 'ID')

        st.write("Discharge Column")
        check_ask_for_column(train_df, 'Q')
        st.write("Drainage Area Column (Models are using specific discharge)")
        check_ask_for_column(train_df, 'A')

        st.write("Optional: Basin Centroid Coordinates")
        st.info("Providing 'lat' and 'lon' columns (latitude and longitude) is optional, but required if you want to plot results on a geographical map.")
        
        show_coords = st.checkbox("Provide Basin Coordinate Columns (Required for some plots)", value=False)
        if show_coords:
            check_ask_for_column(train_df, 'lon')
            check_ask_for_column(train_df, 'lat')
            st.session_state.show_coords = True
        else:
            st.session_state.show_coords = False


    else:
        if temporal_step == 'Yearly':
            train_df = pd.read_csv('../DATA/DF/df_grdc.csv')
            st.session_state.temp_step = 'YEAR'
        elif temporal_step == 'Monthly':
            train_df = pd.read_csv('../DATA/DF/df_grdc_month.csv')
            st.session_state.temp_step = 'MONTH'
        elif temporal_step == 'Seasonal':
            train_df = pd.read_csv('../DATA/DF/df_grdc_season.csv')
            st.session_state.temp_step = 'SEASON'
        st.session_state.show_coords = True  # Default to True for built-in dataset
        
        
    if data_source == "Use Baltic Sea GRDC dataset" or\
                (uploaded_file is not None and all(col in train_df.columns for col in ['ID', 'Q', 'A', 'YEAR'] if temporal_step == 'Yearly') and \
                all(col in train_df.columns for col in ['ID', 'Q', 'A', 'YEAR', 'MONTH'] if temporal_step == 'Monthly') and \
                all(col in train_df.columns for col in ['ID', 'Q', 'A', 'YEAR', 'SEASON'] if temporal_step == 'Seasonal') and\
                (not st.session_state.show_coords or (st.session_state.show_coords and all(col in train_df.columns for col in ['lat', 'lon'])))):
        st.success("Data validation successful! You can proceed with the models.")
        
        col1, col2 = st.columns([0.9, 0.15])
        with col2:
            if st.button("Next step"):
                st.session_state.train_df = train_df
                st.session_state.step = 'MODEL_SELECTION_PAGE'
                st.rerun()


                
    else:
        st.warning("Please ensure all requirments are fulfilled.")


if st.session_state.step == "MODEL_SELECTION_PAGE":
    st.header("Model Selection")
    
    if len(st.session_state.models) == 0:
        st.info("No models added yet. Please add models to proceed.")
    for i, model in enumerate(st.session_state.models):
        col1, col2 = st.columns([0.9, 0.1])
        with col1:
            st.write(f"Model: {model.name}")
        with col2:
            remove = st.button(f"Remove",
                               key=f"remove_{i}",)
        if remove:
            st.session_state.models.remove(model)
            st.rerun()

    add = st.button("Add Model")
    if add:
        st.session_state.step = "ADD_MODEL_PAGE"
        st.rerun()


    col1, col2 = st.columns([0.9, 0.15])
    with col1:
        if st.button("Back to Data Load"):
            st.session_state.step = "DATA_LOAD_PAGE"
            st.rerun()
    with col2:
        next = st.button("Next step")
    if next:
        if len(st.session_state.models) < 1:
            st.warning("Please add at least one model.")
        else:
            st.session_state.step = "INDICATORS_SELECTION_PAGE"
            st.rerun()

if st.session_state.step == "ADD_MODEL_PAGE":
    st.header("Model Selection")

    st.write("**Select the predictors to use for the model:**")
    pred = [col for col in st.session_state.train_df.columns if col not in ['ID', 'Q', 'A', 'YEAR', 'MONTH', 'SEASON', 'lat', 'lon']]
    predictors = st.multiselect(
        "Select Predictors",
        options=pred,
        default=pred[:]  # Default to all predictors
        )
    st.write("**Select the type of regression to use:**")
    regression_type = st.selectbox(
        "Regression Type",
        ["Multiple Linear Regression", "Random Forest"]        
    )

    if regression_type == "Multiple Linear Regression":
        reg = bm.OlsLogMLR(predictors=predictors)

    elif regression_type == "Random Forest":
        max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=10, step=5)
        n_trees = st.number_input("Number of Trees", min_value=5, max_value=200, value=20, step=5)
        reg = bm.LogRF(predictors=predictors, max_depth=max_depth, n_trees=n_trees)


    st.write("**Choose grouping strategies to apply (subgroups are created recursively if more then one are added):**")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        add = st.button("Add Grouping Strategy")
        if add:
            st.session_state.nb_strategies += 1
    with col2:
        rm = st.button("Remove Grouping Strategy")
        if rm:
            if st.session_state.nb_strategies > 0:
                st.session_state.nb_strategies -= 1

    st.session_state.grouping_strategies = [None]*st.session_state.nb_strategies
    predictors_for_clustering = [None]*st.session_state.nb_strategies
    if st.session_state.nb_strategies > 3:
        st.warning("You have added more than 3 grouping strategies. This may lead to a very long computation time and potential empty groups.")
    
    
    for i in range(st.session_state.nb_strategies):
        strat_type = st.selectbox(
            f"**Select Grouping Strategy {i+1}**",
            ["KMeans clustering", "Temporal clustering"],
            key=f"grouping_strategy_{i}"
        )
        if strat_type == "KMeans clustering":
            predictors_for_clustering[i] = st.multiselect(
                f"Select Predictors for clustering",
                options=pred+['A'],
                default=['A'],  # Default to all predictors
                key=f"predictors_for_clustering_{i}"
            )
            n_clusters = st.number_input(
                f"Number of Clusters",
                min_value=2, max_value=20, value=5, step=1,
                key=f"n_clusters_{i}"
            )
            min_size = st.number_input(
                f"Minimum Size of Cluster",
                min_value=1, max_value=100, value=5, step=1,
                key=f"min_size_{i}"
            )

            st.session_state.grouping_strategies[i] = gs.KmeansClustering(cluster_column=predictors_for_clustering[i], n_clusters=n_clusters, min_members=min_size)
        elif strat_type == "Temporal clustering":
            temporal_step = st.selectbox(
                f"Select Temporal Step for Strategy {i+1}",
                ["Yearly", "Monthly", "Seasonal"],
                key=f"temporal_step_{i}"
            )
            if temporal_step == "Yearly":
                temporal_step = "YEAR"
            elif temporal_step == "Monthly":
                if "MONTH" not in st.session_state.train_df.columns:
                    st.error("MONTH column is required for Monthly temporal step. Please check your data.")
                    temporal_step = None
                temporal_step = "MONTH"
            elif temporal_step == "Seasonal":
                if "SEASON" not in st.session_state.train_df.columns:
                    st.error("SEASON column is required for Seasonal temporal step. Please check your data.")
                    temporal_step = None
                temporal_step = "SEASON"

            st.session_state.grouping_strategies[i] = gs.TemporalGrouping(temporal_step)

    st.write("**Choose neighboring strategies to apply (One specific model will be trained only on the nearest neighbors):**")
    neighbor_strat = st.selectbox(
        "Select Neighboring Strategy",
        ["None", "Euclidean distance", "Geographical Neighbors"],
        key="neighboring"
    )
    
    if neighbor_strat == "None":
        st.session_state.neighboring_strategy = None
    elif neighbor_strat == "Euclidean distance":
        n_neighbors = st.number_input(
            "Number of Neighbors",
            min_value=1, max_value=100, value=5, step=1,
            key="n_neighbors"
        )
        columns = st.multiselect(
            "Select Columns for Euclidean Distance",
            options=pred+['A'],
            default=['A'],  # Default to all predictors
            key="euclidean_columns"
        )
        if len(columns) < 1:
            st.warning("Please select at least one column for Euclidean distance.")
        else:
            st.session_state.neighboring_strategy = ns.EuclidianNeighbors(n_neighbors=n_neighbors, columns=columns)
    elif neighbor_strat == "Geographical Neighbors":
        if not st.session_state.show_coords:
            st.warning("Geographical Neighbors requires basin centroid coordinates (lat, lon). Please provide them in the data loading.")
        else:
            with st.spinner("Preparing spatial data..."):
                ids = st.session_state.train_df[['ID', 'lon', 'lat']].drop_duplicates(subset='ID')
                
            # The following lines will be executed while the spinner is shown
                gdf = gpd.GeoDataFrame(
                    ids['ID'],
                    geometry=gpd.points_from_xy(ids['lon'], ids['lat']),
                    crs="EPSG:4326"
                )
                # convert to a metric CRS for distance calculations
                gdf.to_crs(epsg=3395, inplace=True)

            n_neighbors = st.number_input(
                "Number of Neighbors",
                min_value=1, max_value=100, value=5, step=1,
                key="geo_n_neighbors"
            )
            st.session_state.neighboring_strategy = ns.SpatialNeighboring(gdf, gdf, n_neighbors=n_neighbors)



    if len(st.session_state.grouping_strategies) == 0:
        st.session_state.grouping_strategies = None


    if st.button("Add this model"):
        model = am.GeneralModel(
            time_step=st.session_state.temp_step,
            reg_model=reg,
            grouping_strategy=st.session_state.grouping_strategies,
            neighboring_strategy=st.session_state.neighboring_strategy
        )
        st.session_state.models.append(model)
        st.session_state.step = "MODEL_SELECTION_PAGE"
        st.rerun()
        



if st.session_state.step == "INDICATORS_SELECTION_PAGE":
    pass


