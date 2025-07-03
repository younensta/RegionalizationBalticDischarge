import streamlit as st
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

import abstract_models as am
import indicators as idcts
import base_models as bm
import grouping_strategies as gs
import neighboring_strategies as ns
import plotting as pl

import plotly.express as px
import plotly.graph_objects as go



if "mode" not in st.session_state:    # Initialize session state variables
    st.session_state.mode = "Testing"  # Default mode
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
if "random_seed" not in st.session_state:
    st.session_state.random_seed = 42  # Default random seed for reproducibility
if "how" not in st.session_state:
    st.session_state.how = "Holdout-Validation"  # Default validation method
if "res_df" not in st.session_state:
    st.session_state.res_df = None  # Default result dataframe

if "res_fig" not in st.session_state:
    st.session_state.res_fig = None
st.set_page_config(page_title="Data Driven Regionalization Models for Surface Discharge")


# Session state initialization for model selection page
if "default_pred" not in st.session_state:
    st.session_state.default_pred = None  # Default option for prediction
if "default_model_type" not in st.session_state:
    st.session_state.default_model_type = "Multiple Linear Regression"  # Default model type
if "default_max_depth" not in st.session_state:
    st.session_state.default_max_depth = 30  # Default max depth for Random Forest
if "default_nb_tress" not in st.session_state:
    st.session_state.default_nb_tress = 20  # Default number of trees for Random Forest   


def check_ask_for_column(train_df, default_value):
            if default_value in train_df.columns:
                st.success(f"Column '{default_value}' found in the dataframe. If it is not the correct column, please rename it in your data as this name is reserved.")
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





with st.sidebar:
    st.header("Mode Selection")

    col1, col2 = st.columns([0.5, 0.5])
    with col1:
        test = st.button("Testing Mode")
    with col2:
        pred = st.button("Prediction Mode")
    if test:
        st.session_state.mode = "Testing"
        st.session_state.step = "DATA_LOAD_PAGE"

    elif pred:
        st.session_state.mode = "Prediction"
        st.session_state.step = "DATA_LOAD_PAGE"


if st.session_state.mode == "Testing":
    with st.sidebar:
        st.write("You are in Testing Mode. You can test the models with your own data or the built-in dataset.")



    if st.session_state.step == "DATA_LOAD_PAGE":
        st.title("**Testing models for discharge prediction**")

        st.title("Data Load and Preparation")
    
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
                train_df = pd.read_csv('DATA/DF/df_grdc.csv')
                st.session_state.temp_step = 'YEAR'
            elif temporal_step == 'Monthly':
                train_df = pd.read_csv('DATA/DF/df_grdc_month.csv')
                st.session_state.temp_step = 'MONTH'
            elif temporal_step == 'Seasonal':
                train_df = pd.read_csv('DATA/DF/df_grdc_season.csv')
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
        st.title("Model Selection")
        
        if len(st.session_state.models) == 0:
            st.info("No models added yet. Please add models to proceed.")
        for i, model in enumerate(st.session_state.models):
            col1, col2 = st.columns([0.8, 0.2])
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
                st.session_state.step = "COMPUTATION_PAGE"
                st.rerun()

    if st.session_state.step == "ADD_MODEL_PAGE":
        st.title("Model Selection")

        st.write("**Select the predictors to use for the model:**")
        pred = [col for col in st.session_state.train_df.columns if col not in ['ID', 'Q', 'YEAR', 'MONTH', 'SEASON']]
        predictors = st.multiselect(
            "Select Predictors",
            options=pred,
            default=pred if st.session_state.default_pred is None else st.session_state.default_pred
            )

        st.write("**Select the type of regression to use:**")
        regression_type = st.selectbox(
            "Regression Type",
            ["Multiple Linear Regression", "Random Forest"],
            index=0 if st.session_state.default_model_type == "Multiple Linear Regression" else 1
        )

        if regression_type == "Multiple Linear Regression":
            reg = bm.OlsLogMLR(predictors=predictors)

        elif regression_type == "Random Forest":
            max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=st.session_state.default_max_depth, step=5)
            n_trees = st.number_input("Number of Trees", min_value=5, max_value=200, value=st.session_state.default_nb_tress, step=5)
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
        model = am.GeneralModel(
                time_step=st.session_state.temp_step,
                reg_model=reg,
                grouping_strategy=st.session_state.grouping_strategies,
                neighboring_strategy=st.session_state.neighboring_strategy
            )

        if st.checkbox("Set a personalized name for this model"):
            name = st.text_input("Model Name", value=model.name)
            if name:
                model.name = name

        if st.button("Add this model"):
            # Adding default values
            st.session_state.default_pred = predictors  # Store selected predictors in session state
            st.session_state.default_model_type = regression_type  # Store model type in session state
            st.session_state.default_max_depth = max_depth if regression_type == "Random Forest" else st.session_state.default_max_depth
            st.session_state.default_nb_tress = n_trees if regression_type == "Random Forest" else st.session_state.default_nb_tress



            st.session_state.models.append(model)
            st.session_state.step = "MODEL_SELECTION_PAGE"
            st.rerun()
            
    if st.session_state.step == "COMPUTATION_PAGE":
        st.title("Computation Page")
        if st.session_state.show_coords:
            st.info("Basin coordinates are provided. A map can be displayed.")
        if len(st.session_state.models) == 1:
            st.info("One model is selected. Results will be displayed for this model only.")
        st.session_state.how = st.radio(
            "Please select the validation method",
            ("Holdout-Validation", "Leave-One-Out Cross-Validation")
        )
        if st.session_state.how == "Holdout-Validation":
            st.write("Holdout-Validation will be used. Please select the percentage of data to use for training.")
            train_percentage = st.slider(
                "Testing Data Percentage",
                min_value=1, max_value=50, value=10, step=1
            )
            if st.checkbox("Change random seed"):
                st.session_state.random_seed = st.number_input(
                    "Random Seed",
                    min_value=0, max_value=10000, value=42, step=1
                )
        elif st.session_state.how == "Leave-One-Out Cross-Validation":
            st.warning("Leave-One-Out Cross-Validation will be used. This method is computationally expensive and may take a long time to run, especially with multiple models and large datasets.")
            if st.checkbox("Change random seed"):
                st.session_state.random_seed = st.number_input(
                    "Random Seed",
                    min_value=0, max_value=10000, value=42, step=1
                )
        
        
        # Computation
        if len(st.session_state.models) == 0:
            st.warning("No models selected. Please go back to the model selection page and add at least one model.")
        elif len(st.session_state.models) == 1:
            # Initialize training state if not exists
            if "training_in_progress" not in st.session_state:
                st.session_state.training_in_progress = False
            
            if not st.session_state.training_in_progress:
                if st.button("Run training and validation"):
                    st.session_state.training_in_progress = True
                    st.rerun()
            else:
                # Show abort button when training is in progress
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.info("Training in progress...")
                with col2:
                    if st.button("Abort training"):
                        st.session_state.training_in_progress = False
                        st.rerun()
                
                # Run the actual training
                try:
                    if st.session_state.how == "Holdout-Validation":
                        with st.spinner("Running training and validation (this might take a while) ..."):
                            st.session_state.models[0].hold_out_validation(st.session_state.train_df,
                                                                percent=train_percentage,
                                                                random_seed=st.session_state.random_seed,
                                                                show_results=True,
                                                                grouped=True)
                            st.session_state.res_fig = plt.gcf()
                        
                        st.success("Training and validation completed!")
                        st.session_state.training_in_progress = False
                        st.session_state.step = "RESULTS_PAGE"
                        st.rerun()

                    elif st.session_state.how == "Leave-One-Out Cross-Validation":
                        with st.spinner("Running training and validation (this might take a while) ..."):
                            st.session_state.models[0].leave_one_out_validation(
                                st.session_state.train_df,
                                show_results=True,
                                grouped=True
                            )
                        
                        st.success("Training and validation completed!")
                        st.session_state.res_fig = plt.gcf()
                        st.session_state.training_in_progress = False
                        st.session_state.step = "RESULTS_PAGE"
                        st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.session_state.training_in_progress = False
        else:
            # Initialize training state for multiple models if not exists
            if "training_multiple_in_progress" not in st.session_state:
                st.session_state.training_multiple_in_progress = False
            
            if not st.session_state.training_multiple_in_progress:
                if st.button("Run training and validation for all models"):
                    st.session_state.training_multiple_in_progress = True
                    st.rerun()
            else:
                # Show abort button when training is in progress
                col1, col2 = st.columns([0.8, 0.2])
                with col1:
                    st.info("Training multiple models in progress...")
                with col2:
                    if st.button("Abort training"):
                        st.session_state.training_multiple_in_progress = False
                        st.rerun()
                
                # Run the actual training
                try:
                    if st.session_state.how == "Holdout-Validation":
                        with st.spinner("Running training and validation for all models (this might take a while) ..."):
                            for model in st.session_state.models:
                                model.hold_out_validation(st.session_state.train_df,
                                                        percent=train_percentage,
                                                        random_seed=st.session_state.random_seed,
                                                        show_results=False,
                                                        grouped=False)
                        
                        st.success("Training and validation completed!")
                        st.session_state.training_multiple_in_progress = False
                        st.session_state.step = "RESULTS_PAGE"
                        st.rerun()
                        
                    elif st.session_state.how == "Leave-One-Out Cross-Validation":
                        with st.spinner("Running training and validation for all models (this might take a while) ..."):
                            for model in st.session_state.models:
                                model.leave_one_out_validation(
                                    st.session_state.train_df,
                                    show_results=False,
                                    grouped=False
                                )
                        
                        st.success("Training and validation completed!")
                        st.session_state.training_multiple_in_progress = False
                        st.session_state.step = "RESULTS_PAGE"
                        st.rerun()
                except Exception as e:
                    st.error(f"Training failed: {e}")
                    st.session_state.training_multiple_in_progress = False

        back = st.button("Back to Model Selection")
        if back:
            st.session_state.step = "MODEL_SELECTION_PAGE"
            st.rerun()

    if st.session_state.step == "RESULTS_PAGE":
        st.title("Results Page")
        
        
        if len(st.session_state.models)==1:
            st.header("**Global Results:**")
            if st.session_state.res_fig is not None:  # Check if there is a result figure
                st.pyplot(st.session_state.res_fig)  # Get current figure and display it
                plt.close()  # Close the figure to free memory
            
            st.header("**Specific figures**")
            if st.session_state.show_coords:
                if st.session_state.how == "Holdout-Validation":
                    test_ids = st.session_state.models[0].basin_metrics.index
                    plot_df = st.session_state.train_df[['ID', 'lon', 'lat']].copy()
                    plot_df['color'] = "#DE686699"  # Color for training data
                    plot_df.loc[plot_df['ID'].isin(test_ids), 'color'] = "#6DF1597F"  # Color for testing data

                    st.write("**Repartition of Training and Testing Data**")
                    st.map(plot_df, color='color')
                    st.markdown("""
                    <span style="color:#6DF159; font-size: 20px;">●</span> Testing Station &nbsp;&nbsp;&nbsp;
                    <span style="color:#DE6866; font-size: 20px;">●</span> Training Station
                    """, unsafe_allow_html=True)
            
                name = st.selectbox(
                    "Select Indicator to Plot",
                    options=idcts.METRICS_dict.keys(),
                    key="indicator_map"
                )
                
                indic = idcts.METRICS_dict[name]
                df = st.session_state.models[0].basin_metrics[[indic.name]].copy()
                df['ID'] = df.index
                
                coords = st.session_state.train_df[['ID', 'lon', 'lat']].drop_duplicates(subset='ID').copy()
                df = df.merge(coords, on='ID', how='left')
            else:
                st.warning("Basin coordinates are not provided. Map cannot be displayed.")

            if st.session_state.show_coords:
                cmap = plt.get_cmap('RdYlGn')

                
                df[f'{indic.name} legend'] = np.clip(df[indic.name], indic.x_min, indic.x_max)
                
                fig = px.scatter_map(
                    df,
                    lat="lat", 
                    lon="lon", 
                    color=f'{indic.name} legend',
                    color_continuous_scale="RdYlGn_r" if not indic.anti else "RdYlGn",
                    range_color=[indic.x_min, indic.x_max],
                    hover_data={
                        'ID': False,
                        indic.name: ':.2f',
                        f'{indic.name} legend': False,
                        'lon': False,
                        'lat': False,
                    },
                    hover_name='ID',
                    title=f"{name} Values by Station",
                    map_style="carto-positron",
                    zoom=3
                )
                
                fig.update_layout(
                    map_style="carto-positron",
                    height=600,
                    margin={"r":0,"t":30,"l":0,"b":0}
                )
                
                st.plotly_chart(fig, use_container_width=True)


            quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]  # Default quantiles to highlight
            select = st.selectbox(
                "Select Indicator to Plot",
                options=idcts.METRICS_dict.keys(),
                key="indicator_plot"
            )
            if st.checkbox("Choose quantiles to highlight"):
                q = st.multiselect(
                    "Select Quantiles",
                    options=[np.round(0.05* i, 2) for i in range(1, 21)],
                    default=[0.1, 0.25, 0.5, 0.75, 0.9]
                )
                if st.button("Select Quantiles", key="reset_quantiles"):
                    q.sort()
                    quantiles = q

            indic = idcts.METRICS_dict[select]
            fig = pl.fig_single_indicator(indic, st.session_state.models[0], color="#e41818", interesting=quantiles)

            st.pyplot(fig)

        

            
        else:
            st.info("Multiple models are selected. You can compare here their perfomances." \
            "Please select only one model to analyze its performance in detail.")
            st.write("**Select Indicators to compare:**")
            indicators_names = st.multiselect(
                "Select Indicators",
                options=idcts.METRICS_dict.keys(),
                default=list(idcts.METRICS_dict.keys())  # Default to all indicators
            )

            if len(indicators_names) == 0:
                st.warning("Please select at least one indicator to compute.")
            else:
                if st.button("Plot results"):
                    for names in indicators_names:
                        indic = idcts.METRICS_dict[names]
                        fig, ax = plt.subplots(figsize=(10, 6))

                        fig.suptitle(f"{indic.name}", fontsize=16)
                    
                        ax.set_xlabel(f"{indic.name} value ({indic.unit})")
                        ax.set_xlim(indic.x_min, indic.x_max)
                        ax.set_xticks(np.linspace(indic.x_min, indic.x_max, 5))
                        ax.set_ylim(0, 100)
                        ax.set_yticks(np.linspace(0, 100, 6))
                        if indic.anti:
                            ax.set_title(f"Inverted Cumulative Distribution Function for {indic.name}")
                            ax.set_ylabel(f"% of basins with {indic.name} >= x")
                        else:
                            ax.set_title(f"Cumulative Distribution Function for {indic.name}")
                            ax.set_ylabel(f"% of basins with {indic.name} <= x")

                        ax.grid(True, alpha=0.5)

                        colors = plt.get_cmap('cividis', len(st.session_state.models))

                        for i, model in enumerate(st.session_state.models):
                            pl.add_cumulative_indicator(
                                indic,
                                model.basin_metrics,
                                model.name,
                                ax=ax,
                                color =colors(i), 
                                anti = indic.anti)
                        
                        ax.legend(loc='upper left', fontsize=10)
                        st.pyplot(fig)





        back = st.button("Back to Computation")
        if back:
            st.session_state.step = "COMPUTATION_PAGE"
            st.rerun()

elif st.session_state.mode == "Prediction":
    with st.sidebar:
        st.write("You are in Prediction Mode. You can use the models to predict discharge based on your data.")

    if st.session_state.step == "DATA_LOAD_PAGE":
        st.title("**Predicting Discharge Data**")

        st.title("Data Load and Preparation")
    
        data_source = st.radio(
                "Select Data for the model to predict dicharge on",
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
            
            show_coords = st.checkbox("Provide Basin Coordinate Columns (", value=False)
            if show_coords:
                check_ask_for_column(train_df, 'lon')
                check_ask_for_column(train_df, 'lat')
                st.session_state.show_coords = True
            else:
                st.session_state.show_coords = False


        else:
            if temporal_step == 'Yearly':
                train_df = pd.read_csv('DATA/DF/df_grdc.csv')
                st.session_state.temp_step = 'YEAR'
            elif temporal_step == 'Monthly':
                train_df = pd.read_csv('DATA/DF/df_grdc_month.csv')
                st.session_state.temp_step = 'MONTH'
            elif temporal_step == 'Seasonal':
                train_df = pd.read_csv('DATA/DF/df_grdc_season.csv')
                st.session_state.temp_step = 'SEASON'
            st.session_state.show_coords = True  # Default to True for built-in dataset
            
            
        if data_source == "Use Baltic Sea GRDC dataset" or\
                    (uploaded_file is not None and all(col in train_df.columns for col in ['ID', 'Q', 'A', 'YEAR'] if temporal_step == 'Yearly') and \
                    all(col in train_df.columns for col in ['ID', 'Q', 'A', 'YEAR', 'MONTH'] if temporal_step == 'Monthly') and \
                    all(col in train_df.columns for col in ['ID', 'Q', 'A', 'YEAR', 'SEASON'] if temporal_step == 'Seasonal') and\
                    (not st.session_state.show_coords or (st.session_state.show_coords and all(col in train_df.columns for col in ['lat', 'lon'])))):
            st.success("Data validation successful! You can proceed with the data to use for prediction")
            
            source_ready = True
          

        else:
            source_ready = False
        ### prediction data loading
        if source_ready:

            data_source = st.radio(
                    "Select Data for the model to train on",
                    ("Use Baltic Sea BSDB dataset", "Upload your own data")
                )

            if data_source == "Use Baltic Sea BSDB dataset":
                st.info("Using the built-in Baltic Sea BSDB dataset.")
                st.session_state.data_from_user_source = False
                predictor_file = None  # Disable upload
            else:
                predictor_file = st.file_uploader("Upload your data file (.csv)", type=["csv"],
                                                  key="predictor_file")
                if predictor_file is not None:
                    st.success("File uploaded successfully!")
                    pred_df = pd.read_csv(predictor_file)
                    st.session_state.data_from_user_source = True
                    st.write("Data Preview:")
                    st.dataframe(pred_df.head())


            if data_source == "Upload your own data" and predictor_file is not None:
                st.write("Basin and Temporal Identification")
                if st.session_state.temp_step == 'YEAR':
                    check_ask_for_column(pred_df, 'YEAR')
                    
                
                elif st.session_state.temp_step == 'MONTH':
                    check_ask_for_column(pred_df, 'MONTH')
                    check_ask_for_column(pred_df, 'YEAR')
                    
                elif st.session_state.temp_step == 'SEASON':
                    check_ask_for_column(pred_df, 'SEASON')
                    check_ask_for_column(pred_df, 'YEAR')
                    
                
                check_ask_for_column(pred_df, 'ID')

                
                st.write("Drainage Area Column (Models are using specific discharge)")
                check_ask_for_column(pred_df, 'A')

                st.write("Optional: Basin Centroid Coordinates")
                st.info("Providing 'lat' and 'lon' columns (latitude and longitude) is optional, but required if you want to use geographical neighboring or to plot results on a geographical map.")
                
                if st.session_state.show_coords:
                    st.info("Coordinates have been provided in the training data. Please ensure that they are also present in the prediction data.")
                    check_ask_for_column(pred_df, 'lon')
                    check_ask_for_column(pred_df, 'lat')


                #Keeping only the columns that are needed for the prediction AND warning if some columns are not in the training data
                col_train = train_df.columns.tolist()
                col_pred = pred_df.columns.tolist()
                predictors_train = [col for col in col_train if col not in ['ID', 'Q', 'A', 'YEAR', 'MONTH', 'SEASON']]
                predictors_pred = [col for col in col_pred if col not in ['ID', 'A', 'YEAR', 'MONTH', 'SEASON']]

                in_train_not_in_pred = [col for col in predictors_train if col not in predictors_pred]
                in_pred_not_in_train = [col for col in predictors_pred if col not in predictors_train] 

                if len(in_train_not_in_pred) > 0:
                    st.warning(f"The following predictors are in the training data but not in the prediction data: {', '.join(in_train_not_in_pred)} (These predictors will be dropped and cannot be used for prediction)")
                if len(in_pred_not_in_train) > 0:
                    st.warning(f"The following predictors are in the prediction data but not in the training data: {', '.join(in_pred_not_in_train)} (These predictors will be dropped and cannot be used for prediction)")
                st.info("Common predictors for training and prediction data will be used for the prediction. If you want to use all predictors, please ensure that they are present in both datasets under the same names." \
                " The following predictors will be available for the prediction: " + \
                ", ".join([col for col in predictors_train if col in predictors_pred]))

            else:
                if st.session_state.temp_step == 'YEAR':
                    pred_df = pd.read_csv('DATA/DF/df_bsdb.csv')                   
                elif st.session_state.temp_step == 'MONTH':
                    pred_df = pd.read_csv('DATA/DF/df_bsdb_month.csv')                   
                elif st.session_state.temp_step == 'SEASON':
                    pred_df = pd.read_csv('DATA/DF/df_bsdb_season.csv')

            if data_source == "Use Baltic Sea BSDB dataset" or\
                        (predictor_file is not None and all(col in pred_df.columns for col in ['ID', 'A', 'YEAR'] if temporal_step == 'Yearly') and \
                        all(col in pred_df.columns for col in ['ID', 'A', 'YEAR', 'MONTH'] if temporal_step == 'Monthly') and \
                        all(col in pred_df.columns for col in ['ID', 'A', 'YEAR', 'SEASON'] if temporal_step == 'Seasonal') and\
                        (not st.session_state.show_coords or (st.session_state.show_coords and all(col in pred_df.columns for col in ['lat', 'lon'])))):
                st.success("Data validation successful! You can proceed with the data to use for prediction")
                
                predictions_ready = True
            else:
                predictions_ready = False

            



        if source_ready and predictions_ready:
            if st.button("Next step"):
                col1, col2 = st.columns([0.9, 0.15])
                with col2:
                    st.session_state.step = "MODEL_SELECTION_PAGE"

                    st.session_state.train_df = train_df
                    st.session_state.pred_df = pred_df
                    st.rerun()
                    
        else:
            st.warning("Please ensure all requirments are fulfilled.")
    
    if st.session_state.step == "MODEL_SELECTION_PAGE":
        st.title("Model Selection")
        
        
        if len(st.session_state.models) == 0:
            st.info("No model added yet. Please add your model to proceed.")
        for i, model in enumerate(st.session_state.models):
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.write(f"Model: {model.name}")
            with col2:
                remove = st.button(f"Remove",
                                key=f"remove_{i}",)
            if remove:
                st.session_state.models.remove(model)
                st.rerun()

        if len(st.session_state.models) == 0:
            add = st.button("Add Model")
            if add:
                st.session_state.step = "ADD_MODEL_PAGE"
                st.rerun()
        elif len(st.session_state.models) > 1:
            st.error("You have more than one model.Please select only one model to proceed with the prediction or change to testing mode to compare several models.")
        elif len(st.session_state.models) == 1:
            st.success("You have one model selected. You can proceed with the prediction.")
        col1, col2 = st.columns([0.9, 0.15])

        with col1:
            if st.button("Back to Data Load"):
                st.session_state.step = "DATA_LOAD_PAGE"
                st.rerun()
        with col2:
            if len(st.session_state.models) == 1:    
                next = st.button("Next step")
                if next:
                    if len(st.session_state.models) < 1:
                        st.warning("Please add at least one model.")
                    else:
                        st.session_state.step = "COMPUTATION_PAGE"
                        st.rerun()

    if st.session_state.step == "ADD_MODEL_PAGE":
        st.title("Model Selection")

        st.write("**Select the predictors to use for the model:**")
        pred = [col for col in st.session_state.train_df.columns if col not in ['ID', 'Q', 'YEAR', 'MONTH', 'SEASON']]
        predictors = st.multiselect(
            "Select Predictors",
            options=pred,
                 default=pred if st.session_state.default_pred is None else st.session_state.default_pred
            )

        st.write("**Select the type of regression to use:**")
        regression_type = st.selectbox(
            "Regression Type",
            ["Multiple Linear Regression", "Random Forest"],
            index=0 if st.session_state.default_model_type == "Multiple Linear Regression" else 1
        )

        if regression_type == "Multiple Linear Regression":
            reg = bm.OlsLogMLR(predictors=predictors)

        elif regression_type == "Random Forest":
            max_depth = st.number_input("Max Depth", min_value=1, max_value=50, value=st.session_state.default_max_depth, step=5)
            n_trees = st.number_input("Number of Trees", min_value=5, max_value=200, value=st.session_state.default_nb_tress, step=5)
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
                    ids_train = st.session_state.train_df[['ID', 'lon', 'lat']].drop_duplicates(subset='ID')
                    
                # The following lines will be executed while the spinner is shown
                    gdf_train = gpd.GeoDataFrame(
                        ids_train['ID'],
                        geometry=gpd.points_from_xy(ids_train['lon'], ids_train['lat']),
                        crs="EPSG:4326"
                    )
                    # convert to a metric CRS for distance calculations
                    gdf_train.to_crs(epsg=3395, inplace=True)

                    ids_pred = st.session_state.pred_df[['ID', 'lon', 'lat']].drop_duplicates(subset='ID')
                    gdf_pred = gpd.GeoDataFrame(
                        ids_pred['ID'],
                        geometry=gpd.points_from_xy(ids_pred['lon'], ids_pred['lat']),
                        crs="EPSG:4326"
                    )
                    # convert to a metric CRS for distance calculations
                    gdf_pred.to_crs(epsg=3395, inplace=True)

                n_neighbors = st.number_input(
                    "Number of Neighbors",
                    min_value=1, max_value=100, value=5, step=1,
                    key="geo_n_neighbors"
                )
                st.session_state.neighboring_strategy = ns.SpatialNeighboring(gdf_train, gdf_pred, n_neighbors=n_neighbors)



        if len(st.session_state.grouping_strategies) == 0:
            st.session_state.grouping_strategies = None
    
        model = am.GeneralModel(
                time_step=st.session_state.temp_step,
                reg_model=reg,
                grouping_strategy=st.session_state.grouping_strategies,
                neighboring_strategy=st.session_state.neighboring_strategy
            )

        if st.checkbox("Set a personalized name for this model"):
            name = st.text_input("Model Name", value=model.name)
            if name:
                model.name = name

        if st.button("Add this model"):
                   # Adding default values
            st.session_state.default_pred = predictors  # Store selected predictors in session state
            st.session_state.default_model_type = regression_type  # Store model type in session state
            st.session_state.default_max_depth = max_depth if regression_type == "Random Forest" else st.session_state.default_max_depth
            st.session_state.default_nb_tress = n_trees if regression_type == "Random Forest" else st.session_state.default_nb_tress 

            st.session_state.models.append(model)
            st.session_state.step = "MODEL_SELECTION_PAGE"
            st.rerun()
    
    if st.session_state.step == "COMPUTATION_PAGE":
        st.title("Computation Page")
        
        st.write("**Selected Model:**")
        st.write(st.session_state.models[0].name)

        if st.button("Train model and predict discharge"):
            st.session_state.prediction_in_progress = True
            st.rerun()
            
        # Initialize prediction state if not exists
        if "prediction_in_progress" not in st.session_state:
            st.session_state.prediction_in_progress = False
            
        if st.session_state.prediction_in_progress:
            # Show abort button when prediction is in progress
            col1, col2 = st.columns([0.8, 0.2])
            with col1:
                st.info("Training and prediction in progress...")
            with col2:
                if st.button("Abort"):
                    st.session_state.prediction_in_progress = False
                    st.rerun()
            
            # Run the actual training and prediction
            try:
                with st.spinner("Running training and prediction (this might take a while) ..."):
                    st.session_state.models[0].fit(st.session_state.train_df)
                    st.success("Training completed! Now predicting discharge...")
                    st.session_state.res_df = st.session_state.models[0].predict(st.session_state.pred_df, prediction_set = True)

                st.session_state.prediction_in_progress = False
                st.session_state.step = "RESULTS_PAGE"
                st.rerun()
            except Exception as e:
                st.error(f"Training/Prediction failed: {e}")
                st.session_state.prediction_in_progress = False
    if st.session_state.step == "RESULTS_PAGE":
        df = st.session_state.res_df
        st.title("Results")
        st.success("Model sucessfully predicted discharge")

        st.write("**Predicted Discharge Data:**")
            
        with st.expander("Show Predicted Data", expanded=False):    
            st.dataframe(df)
        st.write("Download the predicted discharge data as a CSV file:")
        csv = df.to_csv().encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='predicted_discharge.csv',
            mime='text/csv'
        )

        if st.session_state.show_coords:
            if st.session_state.temp_step == 'YEAR':
                years = st.session_state.res_df.index.get_level_values('YEAR').unique()

                selected_year = st.selectbox(
                    "Select Year to Plot",
                    options=years,
                    key="year_map"
                )

                plot_df = st.session_state.res_df[st.session_state.res_df.index.get_level_values('YEAR') == selected_year].copy()

            elif st.session_state.temp_step == 'SEASON':
                years = st.session_state.res_df.index.get_level_values('YEAR').unique()
                season = st.session_state.res_df.index.get_level_values('SEASON').unique()


                selected_year = st.selectbox(
                    "Select Year to Plot",
                    options=years,
                    key="year_map"
                )
                selected_season = st.selectbox(
                    "Select Season to Plot",
                    options=season,
                    key="season_map"
                )

                plot_df = st.session_state.res_df[(st.session_state.res_df.index.get_level_values('YEAR') == selected_year) & 
                                                  (st.session_state.res_df.index.get_level_values('SEASON') == selected_season)].copy()
            
            elif st.session_state.temp_step == 'MONTH':
                years = st.session_state.res_df.index.get_level_values('YEAR').unique()
                months = st.session_state.res_df.index.get_level_values('MONTH').unique()

                selected_year = st.selectbox(
                    "Select Year to Plot",
                    options=years,
                    key="year_map"
                )
                selected_month = st.selectbox(
                    "Select Month to Plot",
                    options=months,
                    key="month_map"
                )

                plot_df = st.session_state.res_df[(st.session_state.res_df.index.get_level_values('YEAR') == selected_year) &
                                                  (st.session_state.res_df.index.get_level_values('MONTH') == selected_month)].copy()
                

            cmap = plt.get_cmap('RdYlGn')
            
            # Unpack MultiIndex to columns if present
            plot_df = plot_df.reset_index()

            plot_df = plot_df.merge(
                st.session_state.pred_df[['ID', 'lon', 'lat', 'A']].drop_duplicates(subset='ID'),
                on='ID',
                how='left',            )
            plot_df['Q_sim/A (m3/s /m2)'] = plot_df['Q_sim']/ plot_df['A']
        

            fig = px.scatter_map(
                plot_df, 
                lat="lat", 
                lon="lon", 
                color='Q_sim/A (m3/s /m2)',
                range_color=[0, 0.05],
                color_continuous_scale="tropic",
                hover_data={
                    'ID': False,
                    'Q_sim': ':.2f',
                    'lon': False,
                    'lat': False},
                hover_name='ID',
                title=f"Discharge Values by Station",
                map_style="carto-positron",
                zoom=3
                )
            
            fig.update_layout(
                map_style="carto-positron",
                height=600,
                margin={"r":0,"t":30,"l":0,"b":0}
            )
            
            st.plotly_chart(fig, use_container_width=True)

        if st.button("Back to Model Selection"):
            st.session_state.step = "MODEL_SELECTION_PAGE"
            st.rerun()

