import copy
from typing import List, Type, Optional
import logging
from tqdm import tqdm

import pandas as pd
import geopandas as gpd
import numpy as np

import indicators as idcts
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class BaseModel:
    """
    Base class for regressions models
    This class should be inherited by regressions implementations.
    """

    def __init__(self, name:str, predictors:list = []):
        """
        Initialize the model with a name.
        """
        self.name = f'{name}-{predictors}'
        self._is_fitted = False
        self.predictors = predictors

    def fit(self, train_df: pd.DataFrame ):
        """
        Fit the model to the training data.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, test_df: pd.DataFrame):
        """
        Predict using the model.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")

class GroupingStrategy:
    """
    Base class for clustering.
    Should be inherited by different strategies
    """

    def __init__(self, name: str, attributes: List[str]):
        self.name = name
        self.groups_done = False
        self.attributes = attributes

    def create_groups(self, train_df: pd.DataFrame):
        """
        Creates the groups in the training data. Should be inherited by clustering strategies.
        Should return a pd.Dataframe with added columns corresponding to the groupment, and the name of that columns.
        The content of the column should be an integer starting from 0.
        Use pd.Categorical.code if needed to create integers from categories.
        """
        raise NotImplementedError("Subclasses should implement this method.")

    def predict_group(self, test_df: pd.DataFrame):
        """
        Predict the group for the test data.
        Should add a column to the test DataFrame with the group name.
        Groups should be created before calling this method.
        """
        raise NotImplementedError("Subclasses should implement this method.")

class NeighboringStrategy:
    """
    Base class for neighboring strategies.
    Should be inherited by different strategies.
    Will be eventually used to train the model only on K nearest neighbors of the target station within the groups.
    """
    def __init__(self, name: str, k: int, basin_wise:bool = False): #If basin is True, the strategy only use basin characteritics and neighbors are the same for all lines from the same station.
        self.name = name
        self.k = k  #Number of neighbors to consider
        self.basin_wise = basin_wise

    def find_neighbors(self, group_df: pd.DataFrame, target_id: str):
        """
        Find neighbors for the target station in the training data.
        Should return a list of neighboring station IDs.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    
class GeneralModel:
    """
    A general model class that can implement a base model with a several grouping strategies and an enventual final neighboring strategy.
    If there is no neighboring strategy, the model will be trained on all groups and all models will be stored in the `models` dictionary.
    If there is a neighboring strategy, the model will be trained only on the neighbors of the target station within the groups.
    """
    def __init__(self, time_step: str, reg_model: BaseModel, grouping_strategy: List[GroupingStrategy]= [], neighboring_strategy: Optional[NeighboringStrategy] = None):
        """
        Initialize the model with a name and a grouping strategy.
        """

        strategy_names = "_".join([g.name for g in grouping_strategy]) if grouping_strategy else ""
        neighbor_name = f"_{neighboring_strategy.name}_{neighboring_strategy.k}" if neighboring_strategy else ""
        self.name = f'{reg_model.name}_{strategy_names}{neighbor_name}'


        self.reg_model                          = copy.deepcopy(reg_model)
        self.grouping_strategy                  = copy.deepcopy(grouping_strategy)
        self.neighboring_strategy               = neighboring_strategy
        self.models: dict                       = {}
        self._is_fitted : bool                  = False
        self.all_groups: Optional[dict]         = None
        self.all_test_groups: Optional[dict]    = None
        self.final_groups: Optional[dict] = None
        self.final_test_groups: Optional[dict] = None

        self.holdout_df: Optional[pd.DataFrame] = None
        self.global_metrics: Optional[dict] = None
        self.basin_metrics: Optional[pd.DataFrame] = None

        self.loo_df: Optional[pd.DataFrame] = None


        if time_step not in ['MONTH', 'YEAR', 'SEASON']:
            raise ValueError("temporal_step must be one of 'MONTH', 'YEAR', or 'SEASON'")
        self.time_step = time_step
        
        if time_step == 'MONTH':
            self.data_index = ['ID', 'YEAR', 'MONTH']
        elif time_step == 'YEAR':
            self.data_index = ['ID', 'YEAR']
        elif time_step == 'SEASON':
            self.data_index = ['ID', 'YEAR', 'SEASON']


    def _clean(self, df: pd.DataFrame):
        """
        Clean the DataFrame by removing rows with NaN and 0 for 'Q' values in the target variable.
        This method is called every time a dataframe is passed to the model, before fitting or predicting.
        It ensures that the DataFrame is clean and ready for modeling.
        """
        
        
        if 'A' not in df.columns or 'Q' not in df.columns:
            raise ValueError("Target variables 'A' and 'Q' must be present in the DataFrame.")
        
        for p in self.reg_model.predictors:
            if p not in df.columns:
                raise ValueError(f"Predictor '{p}' must be present in the DataFrame.")
        for gr_str in self.grouping_strategy:
            for attr in gr_str.attributes:
                if attr not in df.columns:
                    raise ValueError(f"Grouping attribute '{attr}' must be present in the DataFrame.")


        logger.info("Cleaning DataFrame...")
        rm_df = df[df.isna().any(axis=1) | (df['A'] == 0) | (df['Q'] == 0)]
        logger.info(f"Removing {len(rm_df)} rows...")
        
        if len(rm_df) > 0:
            logger.info("Following rows will be removed:")
            logger.info(rm_df[self.data_index])

        df_clean = df.drop(rm_df.index)
        #Using the data_index as multi index to preserve the structure of the DataFrame, only set it if not already set
        if not isinstance(df_clean.index, pd.MultiIndex):
            df_clean = df_clean.set_index(self.data_index)

        return df_clean

    def _apply_grouping_strategy(self, train_df: pd.DataFrame, remaining_strategies: list[GroupingStrategy], group_path="", father_strat: Optional[GroupingStrategy]=None):
        """Apply recursive grouping strategies to the training DataFrame
        This method will recursively apply the grouping strategies to the training DataFrame.
        It will return a dictionary where keys are group paths and values are tuples of DataFrames with
        the grouped data and the grouping strategy used to create the group."""
        if len(remaining_strategies) == 0:
            return {group_path: (train_df, father_strat)}

        # Apply the current grouping strategy
        current_strat = copy.deepcopy(remaining_strategies[0])
        remaining_strat = remaining_strategies[1:]

        all_groups = {}
        current_strat = copy.deepcopy(remaining_strategies[0])
        remaining_strat = remaining_strategies[1:]
        logger.info(f"Applying grouping strategy: {current_strat.name}")
        all_groups = {}

        grouped_df, group_col = current_strat.create_groups(train_df)

        for group_value in grouped_df[group_col].unique():
            logger.info(f"{current_strat.name}:Processing group: {group_value}")
            new_path = group_path + f";{current_strat.name}:{group_value}"
            
            group_data = grouped_df[grouped_df[group_col]==group_value].drop(columns=[group_col]) #To avoid several columns with the same name in the DataFrame
            
            all_groups[new_path] = (group_data, current_strat)

            subgroups = self._apply_grouping_strategy(group_data, remaining_strat, father_strat=current_strat, group_path=new_path)
            all_groups.update(subgroups)

        return all_groups
    
    def _get_final_groups_only(self, groups_dict: dict):
        """
        Get the final groups only from the groups dictionary.
        This is used to get the final groups after applying all grouping strategies.
        """
        
        logger.info("Getting final groups only...")

        expected = len(self.grouping_strategy)
        final_groups = {}
        for group_path, group_df in groups_dict.items():
            if group_path.count(';') == expected:
                logger.info(f"Final group found: {group_path}")
                final_groups[group_path] = group_df

        return final_groups

    def fit(self, train_df: pd.DataFrame):
        """
        Fit the model to the training data.
        This method will create groups using the grouping strategy and then fit the regression model on each group.
        If a neighboring strategy is provided, it will not fit models. Fitting will be done when predicting.
        If no neighboring strategy is provided, it will fit the model on all groups.
        """
        

        train_df = self._clean(train_df)
        # Recursively create groups using the grouping strategy
        
        logger.info("Creating groups using the grouping strategy...")
        
        self.all_groups = self._apply_grouping_strategy(train_df, self.grouping_strategy)
        self.final_groups = self._get_final_groups_only(self.all_groups)

        logger.info(f"Total groups created: {len(self.all_groups)}")
        self.group_done = True
       
        if self.neighboring_strategy is None:
            logger.info("No neighboring strategy provided, fitting models on all groups...")
            for group_path, (group_df, father_strat) in self.final_groups.items():
                logger.info(f"Fitting model for group: {group_path}")
                # Fit the regression model on the group DataFrame
                model = copy.deepcopy(self.reg_model)
                model.fit(group_df)
                self.models[group_path] = model
        else:
            logger.info("Neighboring strategy provided, models will be fitted during prediction.")
        self._is_fitted = True

    def _predict_groups(self, test_df: pd.DataFrame, remaining_strategies: List[GroupingStrategy], group_path=""):
        """
        Recursively predicts the groups for the test DataFrame using the grouping strategy.
        Return a dictionary where keys are group paths and values are DataFrames with the grouped data.
        """
        if self.all_groups is None:
            raise RuntimeError("Grouping strategy must be applied before prediction.")
        

        logger.info("Predicting groups for the test DataFrame...")
        all_test_groups = {}


        if len(remaining_strategies) == 0:
            # If no remaining strategies, return the current group path and DataFrame
            all_test_groups[group_path] = test_df
            return all_test_groups
        
        remaining = remaining_strategies[1:]
        current_strat = copy.deepcopy(remaining_strategies[0])
        logger.info(f"Predicting using grouping strategy: {current_strat.name}")
        
        path = group_path + f";{current_strat.name}"

        data, father_strat = self.all_groups[f'{path}:0']
        grouped_test_data, group_col = father_strat.predict_group(test_df)
       
        for group_value in grouped_test_data[group_col].unique():
            logger.info(f"Processing group: {group_value}")
            new_path = path + f":{group_value}"
            group_data = grouped_test_data[grouped_test_data[group_col] == group_value].drop(columns=[group_col])
            subgroups = self._predict_groups(group_data, remaining, group_path=new_path)
            all_test_groups.update(subgroups)
        
        return all_test_groups

    def predict(self, test_df: pd.DataFrame):
        """
        Predict using the model.
        This method will apply the grouping strategy to the test DataFrame and then use the fitted models to make predictions.
        If a neighboring strategy is provided, it will use the neighbors of the target station within the groups.
        """
        

        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        test_df = self._clean(test_df)
        
        # Apply grouping strategy to the test DataFrame
        logger.info("Applying grouping strategy to the test DataFrame...")
        
        self.all_test_groups = self._predict_groups(test_df, self.grouping_strategy)
        self.final_test_groups = self._get_final_groups_only(self.all_test_groups)

        # Prepare results DataFrame
        res_df = test_df.reset_index()[self.data_index + ['Q']].copy()  # Reset index first
        res_df = res_df.set_index(self.data_index)  # Set it back as index
        res_df['Q_sim'] = np.nan

        # Predict using the models for each group
        for group_path, group_test_df in self.final_test_groups.items():
            logger.info(f"Predicting for group: {group_path}")
            
            if self.neighboring_strategy is None:
                if group_path in self.models:
                    model = self.models[group_path]
                    group_predictions = model.predict(group_test_df)
                    res_df.loc[group_test_df.index, 'Q_sim'] = group_predictions['Q_sim'].values
                else:
                    logger.info(f"No model found for group: {group_path}, skipping prediction.")
            else:
                # If a neighboring strategy is provided, find neighbors and predict
                # Get the corresponding training group for finding neighbors
                train_group_df = self.final_groups[group_path][0]  # Get the training DataFrame for this group
                
                if self.neighboring_strategy.basin_wise:
                    for id in group_test_df.index.get_level_values('ID').unique():
                        neighbors = self.neighboring_strategy.find_neighbors(train_group_df, id)
                        if len(neighbors)==0:
                            logger.info(f"No neighbors found for target ID: {id}, skipping prediction.")
                            continue
                        
                        # Create a DataFrame with the neighbors from TRAINING data
                        neighbor_df = train_group_df[train_group_df.index.get_level_values('ID').isin(neighbors)]

                        if not neighbor_df.empty:
                            model = copy.deepcopy(self.reg_model)
                            model.fit(neighbor_df)
                            # Predict only for this specific test ID
                            test_id_df = group_test_df[group_test_df.index.get_level_values('ID') == id]
                            group_predictions = model.predict(test_id_df)
                            res_df.loc[test_id_df.index, 'Q_sim'] = group_predictions['Q_sim'].values

                else:# not basin wise, so we can use multiindex
                    for data_id in group_test_df.index:
                        neighbors = self.neighboring_strategy.find_neighbors(train_group_df, data_id)
                        if len(neighbors) == 0:
                            logger.info(f"No neighbors found for target ID: {data_id}, skipping prediction.")
                            continue
                        
                        # Create a DataFrame with the neighbors from TRAINING data
                        neighbor_df = train_group_df[train_group_df.index.isin(neighbors)]

                        if not neighbor_df.empty:
                            model = copy.deepcopy(self.reg_model)
                            model.fit(neighbor_df)
                            # Predict only for this specific data point
                            single_test_df = group_test_df.loc[[data_id]]
                            group_predictions = model.predict(single_test_df)
                            res_df.loc[data_id, 'Q_sim'] = group_predictions['Q_sim'].values[0]
                        
        return res_df

    def _compute_metrics(self, df: pd.DataFrame):
        if df is None:
            raise RuntimeError("No predictions made yet.")
        pass
        if 'Q' not in df.columns or 'Q_sim' not in df.columns:
            raise ValueError("The results DataFrame must contain 'Q' and 'Q_sim' columns for metric computation.")

        metric_df = pd.DataFrame(index=df.index)
        metric_df['Q'] = df['Q']
        metric_df['Q_sim'] = df['Q_sim']
        metric_df['error'] = metric_df['Q'] - metric_df['Q_sim']
        metric_df['abs_error'] = np.abs(metric_df['error'])
        metric_df['rel_error'] = metric_df['abs_error'] / metric_df['Q']

        global_metrics = {}
        global_metrics['mean_Q'] = metric_df['Q'].mean()
        global_metrics['mean_Q_sim'] = metric_df['Q_sim']
        global_metrics['mean_error'] = metric_df['error'].mean()
        global_metrics['mean_absolute_error'] = metric_df['abs_error'].mean()
        global_metrics['nse'] = idcts.nse(metric_df['Q'], metric_df['Q_sim'])
        global_metrics['pbias'] = idcts.pbias(metric_df['Q'], metric_df['Q_sim'])
        global_metrics['rmse'] = idcts.rmse(metric_df['Q'], metric_df['Q_sim'])
        global_metrics['nrmse'] = idcts.nrmse(metric_df['Q'], metric_df['Q_sim'])
        global_metrics['medape'] = idcts.medape(metric_df['Q'], metric_df['Q_sim'])
        global_metrics['smape'] = idcts.smape(metric_df['Q'], metric_df['Q_sim'])
        global_metrics['medsmape'] = idcts.medsmape(metric_df['Q'], metric_df['Q_sim'])
        global_metrics['flow_deciles_nse'] = idcts.flow_deciles_nse(metric_df['Q'], metric_df['Q_sim'])
        global_metrics['flow_deciles_smape'] = idcts.flow_deciles_smape(metric_df['Q'], metric_df['Q_sim'])

        self.global_metrics = global_metrics

        basin_metrics = pd.DataFrame(columns=['ID', 'mean_Q', 'mean_Q_sim', 'mean_error',
                                              'mean_absolute_error',
                                              'nse', 'pbias', 'rmse', 'nrmse',
                                              'medape', 'smape', 'medsmape',
                                              'flow_deciles_nse', 'flow_deciles_smape'])

        self.basin_metrics = basin_metrics.set_index('ID')

        for basin in metric_df.index.get_level_values('ID').unique():
            basin_df = metric_df.xs(basin, level='ID')
            
            basin_metrics.loc[basin, 'mean_Q'] = basin_df['Q'].mean()
            basin_metrics.loc[basin, 'mean_Q_sim'] = basin_df['Q_sim'].mean()
            basin_metrics.loc[basin, 'mean_error'] = basin_df['error'].mean()
            basin_metrics.loc[basin, 'mean_absolute_error'] = basin_df['abs_error'].mean()
            basin_metrics.loc[basin, 'nse'] = idcts.nse(basin_df['Q'], basin_df['Q_sim'])
            basin_metrics.loc[basin, 'pbias'] = idcts.pbias(basin_df['Q'], basin_df['Q_sim'])
            basin_metrics.loc[basin, 'rmse'] = idcts.rmse(basin_df['Q'], basin_df['Q_sim'])
            basin_metrics.loc[basin, 'nrmse'] = idcts.nrmse(basin_df['Q'], basin_df['Q_sim'])
            basin_metrics.loc[basin, 'medape'] = idcts.medape(basin_df['Q'], basin_df['Q_sim'])
            basin_metrics.loc[basin, 'smape'] = idcts.smape(basin_df['Q'], basin_df['Q_sim'])
            basin_metrics.loc[basin, 'medsmape'] = idcts.medsmape(basin_df['Q'], basin_df['Q_sim'])
            basin_metrics.at[basin, 'flow_deciles_nse'] = idcts.flow_deciles_nse(basin_df['Q'], basin_df['Q_sim'])
            basin_metrics.at[basin, 'flow_deciles_smape'] = idcts.flow_deciles_smape(basin_df['Q'], basin_df['Q_sim'])
            basin_metrics.loc[basin, 'mape'] = idcts.mape(basin_df['Q'], basin_df['Q_sim'])

        self.basin_metrics = basin_metrics
        logger.info("Basin metrics calculated successfully.")
        
    def _show_results(self, df, grouped: bool=False, blocked: bool=True):
        """
        Show the results of the predictions.
        This method will plot the predicted vs actual values for each group.
        """
        if self.global_metrics is None:
            self._compute_metrics(df)

        if df is None:
            raise RuntimeError("No predictions made yet. Call predict() first.")

        if self.grouping_strategy==[] or self.grouping_strategy is None:
            grouped = False
        # Create subplot layout
        fig = plt.figure(figsize=(15, 12))
        gs = gridspec.GridSpec(6, 6)


        ax1 = fig.add_subplot(gs[0:2, 0:2])
        ax2 = fig.add_subplot(gs[0:2, 2:4])
        ax3 = fig.add_subplot(gs[0:2, 4:6])
        
        ax4 = fig.add_subplot(gs[2:4, 0:2])
        ax5 = fig.add_subplot(gs[2:4, 2:4])
        ax6 = fig.add_subplot(gs[2:4, 4:6])
        
        ax7 = fig.add_subplot(gs[4:6, 0:2])
        ax8 = fig.add_subplot(gs[4:6, 2:4])
        ax9 = fig.add_subplot(gs[4:6, 4:6])

        fig.suptitle(f"Results for: {self.name}", fontsize=16)
       
        if grouped:
            group_labels = []
            colors = plt.cm.get_cmap('viridis', len(self.final_test_groups))

            for i, (group_path, group_df) in enumerate(self.final_test_groups.items()):
                group_labels.append(group_path)
                group_res_df = df.loc[group_df.index]

                ax1.scatter(group_res_df['Q'], group_res_df['Q_sim'], label=group_path, color=colors(i), alpha=0.5)
                ax1.plot([group_res_df['Q'].min(), group_res_df['Q'].max()], [group_res_df['Q'].min(), group_res_df['Q'].max()], color='black', linestyle='--')

            ax1.set_xlabel(self.data_index[-1])            
           
            ax2.set_title('Relative Error by Group')
            for i, (group_path, group_df) in enumerate(self.final_test_groups.items()):
                group_res_df = df.loc[group_df.index]
                rel_error = (group_res_df['Q_sim'] - group_res_df['Q']) / group_res_df['Q']
                ax2.scatter(group_res_df['Q'], rel_error, label=group_path, color=colors(i), alpha=0.5)
                ax2.axhline(0, color='black', linestyle='--')
            ax3.set_title('Error by Group')
            for i, (group_path, group_df) in enumerate(self.final_test_groups.items()):
                group_res_df = df.loc[group_df.index]
                abs_error = group_res_df['Q_sim'] - group_res_df['Q']
                ax3.scatter(group_res_df['Q'], abs_error, label=group_path, color=colors(i), alpha=0.5)
                ax3.axhline(0, color='black', linestyle='--')



        else:
            ax1.set_title('Actual vs Fitted Q')
            ax1.scatter(df['Q'], df['Q_sim'], color='blue', label='Predicted vs Actual', alpha=0.5)
            ax1.plot([df['Q'].min(), df['Q'].max()], [df['Q'].min(), df['Q'].max()], color='black', linestyle='--')
            
            ax2.set_title('Relative Error')
            ax2.scatter(df['Q'], (df['Q'] - df['Q_sim']) / df['Q'], color='blue', label='Relative Error', alpha=0.5)
            ax2.axhline(0, color='black', linestyle='--')

            ax3.set_title('Error')
            ax3.scatter(df['Q'], df['Q_sim'] - df['Q'], color='blue', label='Absolute Error', alpha=0.5)
            ax3.axhline(0, color='black', linestyle='--')

        ax1.set_xlabel('Actual Q')       
        ax1.set_ylabel('Fitted Q')
        ax1.set_title('Actual vs Fitted Q')
        ax1.set_xscale('log')
        ax1.set_yscale('log')

        ax2.set_xlabel("Actual Q")
        ax2.set_ylabel("Relative Error")
        ax2.set_xscale('log')

        ax3.set_xlabel('Actual Q')
        ax3.set_ylabel('Error (m3/s)')
        ax3.set_xscale('log')
        ax3.set_yscale('linear')

        
        # NSE plot
        nse_values = self.basin_metrics['nse'].dropna()
        len_nse = len(nse_values)

        nb = 200
        x_values = np.linspace(-1, 1, nb)
        y_values = [np.sum(nse_values > x) / len_nse for x in x_values]

        ax4.plot(x_values, y_values, color='grey', linewidth=2)
        ax4.set_xlabel('NSE')
        ax4.set_ylabel("% of basins with NSE > x")
        ax4.set_title('NSE per Basin')
        
        interesting = [0.1, 0.25, 0.5, 0.75, 0.9]  # percentiles of interest
        colors = plt.cm.get_cmap('winter', len(interesting))
        

        for j, percentile in enumerate(interesting):
            nse_value = np.quantile(nse_values, percentile)
            color = colors(j)

            if nse_value > -1:
                # Horizontal line from left to intersection point
                ax4.axhline(y=(1-percentile), xmin=0, xmax=(nse_value + 1) / 2, color=color, linestyle='--', alpha=0.8)
                # Vertical line from bottom to intersection point
                ax4.axvline(x=nse_value, ymin=0, ymax=(1-percentile), color=color, linestyle='--', alpha=0.8)

                # Add text label at the intersection point
                ax4.text(nse_value, 1-percentile, f"{(1-percentile)*100:.0f}%: {nse_value:.2f}", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'),
                        color='black', fontsize=8, ha='left', va='bottom')


        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(-1, 1)
        ax4.set_ylim(0, 1)

        # PBIAS plot
        pbias_values = np.abs(self.basin_metrics['pbias'].dropna())
        len_pbias = len(pbias_values)

        nb = 200
        x_values = np.linspace(0, 100, nb)
        y_values = [np.sum(pbias_values < x) / len_pbias for x in x_values]

        ax5.plot(x_values, y_values, color='red', linewidth=2)
        ax5.set_xlabel('Absolute PBIAS')
        ax5.set_ylabel("% of basins with |PBIAS| < x")
        ax5.set_title('PBIAS per Basin')

        interesting = [0.1, 0.25, 0.5, 0.75, 0.9]  # percentiles of interest
        colors = plt.cm.get_cmap('winter', len(interesting))
        

        for j, percentile in enumerate(interesting):
            pbias_value = np.quantile(pbias_values, percentile)
            color = colors(j)

            if pbias_value > -1:
                # Horizontal line from left to intersection point
                ax5.axhline(y=percentile, xmin=0, xmax=pbias_value/100, color=color, linestyle='--', alpha=0.8)
                # Vertical line from bottom to intersection point
                ax5.axvline(x=pbias_value, ymin=0, ymax=percentile, color=color, linestyle='--', alpha=0.8)

                # Add text label at the intersection point
                ax5.text(pbias_value, percentile, f"{percentile*100:.0f}%: {pbias_value:.2f}", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'),
                        color='black', fontsize=8, ha='left', va='bottom')


        ax5.grid(True, alpha=0.3)
        ax5.set_xlim(0, 100)
        ax5.set_ylim(0, 1)


        # MedAPE plot
        mape_values = self.basin_metrics['medape'].dropna()
        len_mape = len(mape_values)
        nb = 200
        x_values = np.linspace(0, 100, nb)
        y_values = [np.sum(mape_values < x) / len_mape for x
                    in x_values]
        ax6.plot(x_values, y_values, color='green', linewidth=2)
        ax6.set_xlabel('MedAPE (%)')
        ax6.set_ylabel("% of basins with MedAPE < x")
        ax6.set_title('MedAPE per Basin')
        interesting = [0.1, 0.25, 0.5, 0.75, 0.9]
        colors = plt.cm.get_cmap('winter', len(interesting))
        for j, percentile in enumerate(interesting):
            mape_value = np.quantile(mape_values, percentile)
            color = colors(j)

            if mape_value > -1:
                # Horizontal line from left to intersection point
                ax6.axhline(y=percentile, xmin=0, xmax=mape_value/100, color=color, linestyle='--', alpha=0.8)
                # Vertical line from bottom to intersection point
                ax6.axvline(x=mape_value, ymin=0, ymax=percentile, color=color, linestyle='--', alpha=0.8)

                # Add text label at the intersection point
                ax6.text(mape_value, percentile, f"{percentile*100:.0f}%: {mape_value:.2f}", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'),
                        color='black', fontsize=8, ha='left', va='bottom')
        ax6.grid(True, alpha=0.3)
        ax6.set_xlim(0, 100)
        ax6.set_ylim(0, 1)

        # sMAPE
        smape_values = self.basin_metrics['smape'].dropna()
        len_smape = len(smape_values)
        nb = 200
        x_values = np.linspace(0, 100, nb)
        y_values = [np.sum(smape_values < x) / len_smape for x
                    in x_values]
        ax7.plot(x_values, y_values, color='purple', linewidth=2)
        ax7.set_xlabel('sMAPE (%)')
        ax7.set_ylabel("% of basins with sMAPE < x")
        ax7.set_title('sMAPE per Basin')
        interesting = [0.1, 0.25, 0.5, 0.75, 0.9]
        colors = plt.cm.get_cmap('winter', len(interesting))
        for j, percentile in enumerate(interesting):
            smape_value = np.quantile(smape_values, percentile)
            color = colors(j)

            if smape_value > -1:
                # Horizontal line from left to intersection point
                ax7.axhline(y=percentile, xmin=0, xmax=smape_value/100, color=color, linestyle='--', alpha=0.8)
                # Vertical line from bottom to intersection point
                ax7.axvline(x=smape_value, ymin=0, ymax=percentile, color=color, linestyle='--', alpha=0.8)

                # Add text label at the intersection point
                ax7.text(smape_value, percentile, f"{percentile*100:.0f}%: {smape_value:.2f}", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'),
                        color='black', fontsize=8, ha='left', va='bottom')
        ax7.grid(True, alpha=0.3)
        ax7.set_xlim(0, 100)
        ax7.set_ylim(0, 1)

        #APE cumulative plot
        utilisables_values = np.abs(df[['Q', 'Q_sim']].dropna())


        #Calculate relative error
        utilisables_values = 100*np.abs((utilisables_values['Q_sim'] - utilisables_values['Q'])/ utilisables_values['Q'])

        len_utilisables = len(utilisables_values)
        nb = 200
        x_values = np.linspace(0, 100, nb)
        y_values = [np.sum(utilisables_values < x) / len_utilisables for x in x_values]

        ax8.plot(x_values, y_values, color='orange', linewidth=2)
        ax8.set_xlabel('Relative Error')
        ax8.set_ylabel("% of basins with Relative Error < x")
        ax8.set_title('Relative Error (by point)')
        interesting = [0.1, 0.25, 0.5, 0.75, 0.9]
        colors = plt.cm.get_cmap('winter', len(interesting))
        for j, percentile in enumerate(interesting):
            rel_error_value = np.quantile(utilisables_values, percentile)
            color = colors(j)

            if (rel_error_value > 0 and rel_error_value < 100):
                # Horizontal line from left to intersection point
                ax8.axhline(y=percentile, xmin=0, xmax=rel_error_value/100, color=color, linestyle='--', alpha=0.8)
                # Vertical line from bottom to intersection point
                ax8.axvline(x=rel_error_value, ymin=0, ymax=percentile, color=color, linestyle='--', alpha=0.8)

                # Add text label at the intersection point
                ax8.text(rel_error_value, percentile, f"{percentile*100:.0f}%: {rel_error_value:.2f}", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'),
                        color='black', fontsize=8, ha='left', va='bottom')

        ax8.grid(True, alpha=0.3)
        ax8.set_xlim(0, 100)
        ax8.set_ylim(0, 1)
        #displaying in percentage

        # per basin MAPE plot
        mape_values = self.basin_metrics['mape'].dropna()
        len_mape = len(mape_values)
        nb = 200
        x_values = np.linspace(0, 100, nb)
        y_values = [np.sum(mape_values < x) / len_mape for x in x_values]
        ax9.plot(x_values, y_values, color='brown', linewidth=2)
        ax9.set_xlabel('MAPE')
        ax9.set_ylabel("% of basins with MAPE < x")
        ax9.set_title('MAPE per Basin')
        interesting = [0.1, 0.25, 0.5,  0.75, 0.9]
        colors = plt.cm.get_cmap('winter', len(interesting))
        for j, percentile in enumerate(interesting):
            mape_value = np.quantile(mape_values, percentile)
            color = colors(j)

            if  (mape_value > 0 and  mape_value < 100):
                # Horizontal line from left to intersection point
                ax9.axhline(y=percentile, xmin=0, xmax=mape_value/100, color=color, linestyle='--', alpha=0.8)
                # Vertical line from bottom to intersection point
                ax9.axvline(x=mape_value, ymin=0, ymax=percentile, color=color, linestyle='--', alpha=0.8)

                # Add text label at the intersection point
                ax9.text(mape_value, percentile, f"{percentile*100:.0f}%: {mape_value:.2f}", 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'),
                        color='black', fontsize=8, ha='left', va='bottom')
        
        ax9.grid(True, alpha=0.3)
        ax9.set_xlim(0, 100)
        ax9.set_ylim(0, 1)


        # Adjust layout and show the plot
        plt.subplots_adjust(hspace=0.4, wspace=0.4)


        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=blocked)

    def hold_out_validation(self, df:pd.DataFrame, percent:int =10, random_seed:int =42, show_results:bool =True, grouped:bool =True):
        """
        Performs hold-out validation on the provided DataFrame.
        """
        np.random.seed(random_seed)
        ids = df['ID'].unique()
        test_ids = np.random.choice(ids, size=int(len(ids) * percent / 100), replace=False)
        
        train_df = df[~df['ID'].isin(test_ids)]
        test_df = df[df['ID'].isin(test_ids)]

        #Train the model on the training DataFrame
        self.fit(train_df)
        #Evaluate the model on the test DataFrame
        self.holdout_df = self.predict(test_df)
        #Show the results
        if show_results:
            self._show_results(self.holdout_df, grouped=grouped, blocked=True)

    def leave_one_out_validation(self, df: pd.DataFrame, show_results: bool = True, grouped: bool = False):
        """
        Performs leave-one-out validation on the provided DataFrame.
        Can take a lot of time because it fits a new model for each ID in the Dataframe
        """
        # Silence logger during LOO
        original_level = logging.getLogger().level  # Root logger
        logging.getLogger().setLevel(logging.ERROR)  # This affects all child loggers
        


        try:
            df_loo = df[self.data_index+['Q']].copy()
            df_loo = df_loo.set_index(self.data_index)
            df_loo['Q_sim'] = np.nan

            ids = df['ID'].unique()


            # Use tqdm for a nice progress bar
            for id in tqdm(ids, desc="Leave-one-out validation"):
                train_df = df[df['ID'] != id]
                test_df = df[df['ID'] == id]

                self.fit(train_df)
                res = self.predict(test_df)
                res_index = res.index #We take res index to leave NaNs where clean deleted rows

                df_loo.loc[res_index, 'Q_sim'] = res['Q_sim'].values

        finally:
            logger.setLevel(original_level)
        
        self.loo_df = df_loo
        # Show the results
        if show_results:

            self._show_results(self.loo_df, grouped=grouped, blocked=True)
        logger.info("Leave-one-out validation completed successfully.")