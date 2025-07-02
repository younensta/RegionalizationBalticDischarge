import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from typing import List, Type
from abstract_models import *

import indicators as idcts

    

def add_cumulative_indicator(indicator_class: Type[idcts.Metric] , basin_metric_df: pd.DataFrame,
                              model_name:str, ax: plt.Axes, color, anti:bool=False):
    """
    Adds to ax the cumulative distribution function of the given indicator_class
    """
    values = basin_metric_df[indicator_class.name].dropna()
    len_values = len(values)
    nb = 200
    x_values = np.linspace(indicator_class.x_min, indicator_class.x_max, nb)

    if anti:
        y_values = np.array([100*np.sum(values >= x) for x in x_values]) / len_values
    else:
        y_values = np.array([100*np.sum(values <= x) for x in x_values]) / len_values

    ax.plot(x_values, y_values, label=model_name, color=color)
    interesting = [0.1, 0.25, 0.5, 0.75, 0.9]  # Percentiles to highlight
    colors = plt.cm.get_cmap('winter', len(interesting))
    for j, percentile in enumerate(interesting):
        val = np.quantile(values, percentile)
        color2 = colors(j)

        if  (val > indicator_class.x_min and  val < indicator_class.x_max):
            if anti:
                percentile = 1 - percentile  # Invert the percentile for anti metrics
            # Horizontal line from left to intersection point
            ax.axhline(y=100*percentile, xmin=0, xmax=(val - indicator_class.x_min) / (indicator_class.x_max - indicator_class.x_min),
                        color=color2, linestyle='--', alpha=0.2)
            
            # Vertical line from bottom to intersection point
            ax.axvline(x=val, ymin=0, ymax=percentile, color=color, linestyle='--', alpha=0.8)

            # Add text label at the intersection point
            ax.text(indicator_class.x_min, 100*percentile, f"{percentile*100:.0f}%:", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'),
                    color='black', fontsize=8, ha='left', va='bottom')


def plot_models_results(indicators: List[Type[idcts.Metric]], models: List[GeneralModel],
                        train_df:pd.DataFrame, how:str='holdout', percent:int=10 ):
    
    for m in indicators:
        if m.name not in idcts.METRICS_names:
            raise ValueError(f"Indicator {m.name} is not a valid metric. Should be added to idcts.METRICS first.")
    
    if how not in ['holdout', 'leave-one-out']:
        raise ValueError("Invalid value for 'how'. Should be either 'holdout' or 'leave-one-out'.")
    
    fig, axs = plt.subplots(len(indicators), 1, figsize=(10, 5 * len(indicators)))
    fig.suptitle(f"Model Performance Metrics - {how.capitalize()} Validation", fontsize=16)
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, len(models)))

    for i, indicator in enumerate(indicators):
        ax = axs[i]
        
        ax.set_xlabel(f"{indicator.name} value ({indicator.unit})" )
        if indicator.anti:
            ax.set_title(f"Inverted Cumulative Distribution Function for {indicator.name}")
            ax.set_ylabel(f"% ob basins with {indicator.name} >= x")
        else:
            ax.set_title(f"Cumulative Distribution Function for {indicator.name}")
            ax.set_ylabel(f"% ob basins with {indicator.name} <= x")

        ax.set_xlim(indicator.x_min, indicator.x_max)
        ax.set_xticks(np.linspace(indicator.x_min, indicator.x_max, 5))
        ax.set_ylim(0, 100)
        ax.set_yticks(np.linspace(0, 100, 6))
        ax.grid(True, alpha=0.5)

        
    for model, color in zip(models, colors):
        if how == 'holdout':
            model.hold_out_validation(train_df, percent=percent, show_results=False)

        elif how == 'leave-one-out':
            model.leave_one_out_validation(train_df, show_results=False)

        for indicator in indicators:
            if indicator.name not in model.basin_metrics.columns:
                raise ValueError(f"Indicator {indicator.name} not found in model {model.name} basin metrics.")
            add_cumulative_indicator(indicator, model.basin_metrics, model.name, axs[indicators.index(indicator)], color, anti=indicator.anti)
            ax.grid(True)
        ax.legend(loc='lower right', fontsize='small')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def fig_single_indicator(indicator: Type[idcts.Metric], model: GeneralModel,
                        color: str = "#ee8133", interesting = [0.1, 0.25, 0.5, 0.75, 0.9]) -> plt.Figure:
    """
    Plot the cumulative distribution function of a single indicator for a given model.
    """

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle(f"{indicator.name} - {model.name}", fontsize=16)
 
    ax.set_xlabel(f"{indicator.name} value ({indicator.unit})")
    ax.set_xlim(indicator.x_min, indicator.x_max)
    ax.set_xticks(np.linspace(indicator.x_min, indicator.x_max, 5))
    ax.set_ylim(0, 100)
    ax.set_yticks(np.linspace(0, 100, 6))
    if indicator.anti:
        ax.set_title(f"Inverted Cumulative Distribution Function for {indicator.name}")
        ax.set_ylabel(f"% of basins with {indicator.name} >= x")
    else:
        ax.set_title(f"Cumulative Distribution Function for {indicator.name}")
        ax.set_ylabel(f"% of basins with {indicator.name} <= x")

    ax.grid(True, alpha=0.5)
    
    if model.basin_metrics is None:
        raise ValueError(f"Model {model.name} has no basin metrics. Please run validation first.")

    values = model.basin_metrics[indicator.name].dropna()
    len_values = len(values)
    nb = 200
    x_values = np.linspace(indicator.x_min, indicator.x_max, nb)

    if indicator.anti:
        y_values = np.array([100*np.sum(values >= x) for x in x_values]) / len_values
    else:
        y_values= np.array([100*np.sum(values <= x) for x in x_values]) / len_values

    ax.plot(x_values, y_values, label=model.name, color=color)

    colors = plt.cm.get_cmap('winter', len(interesting))
    for j, percentile in enumerate(interesting):
        val = np.quantile(values, percentile)
        color2 = colors(j)

        if  (val > indicator.x_min and  val < indicator.x_max):
            if indicator.anti:
                percentile = 1 - percentile  # Invert the percentile for anti metrics
            # Horizontal line from left to intersection point
            ax.axhline(y=100*percentile, xmin=0, xmax=(val - indicator.x_min) / (indicator.x_max - indicator.x_min),
                        color=color2, linestyle='--', alpha=0.8)
            
            # Vertical line from bottom to intersection point
            ax.axvline(x=val, ymin=0, ymax=percentile, color=color2, linestyle='--', alpha=0.8)

            # Add text label at the intersection point
            ax.text(val, 100*percentile, f"{percentile*100:.0f}%:{val:.2f}:", 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor='none'),
                    color='black', fontsize=8, ha='left', va='bottom')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    return fig 
