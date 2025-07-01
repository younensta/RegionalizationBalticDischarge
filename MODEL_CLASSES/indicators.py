# This files contains the metrics for evaluating model performance

import numpy as np

class Metric:
    """
    Base class for metrics
    """
    name: str
    #Plotting window
    x_min: float
    x_max: float
    unit: str
    anti: bool #If true, the cumulative distribution function is inverted

    def __call__(self, actual, fitted):
        """
        Calculate the metric value between actual and fitted values.
        This method should be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses should implement this method.")
    

class NSE(Metric):
    """
    Nash-Sutcliffe Efficiency (NSE) metric.
    """
    name = "NSE"
    x_min = -1.0
    x_max = 1.0
    unit = "unit-less"
    anti = True  # Higher values are better, so we invert the CDF
    
    def __call__(self, actual, fitted):
        numerator = np.sum((actual - fitted) ** 2)
        denominator = np.sum((actual - np.mean(actual)) ** 2)

        if denominator == 0:
            return np.nan  # Avoid division by zero

        return 1 - (numerator / denominator)

class PBIAS(Metric):
    """
    Percent Bias (PBIAS) metric.
    """
    name = "PBIAS"
    x_min = -100.0
    x_max = 100.0
    unit = "%"
    anti = False

    def __call__(self, actual, fitted):
        return 100 * np.sum(fitted - actual) / np.sum(actual)

class APBIAS(Metric):
    """
    Absolute Percent Bias (APBIAS) metric.
    """
    name = "APBIAS"
    x_min = 0.0
    x_max = 100.0
    unit = "%"
    anti = False
    
    def __call__(self, actual, fitted):
        return np.mean(np.abs((fitted - actual) / actual)) * 100 if np.any(actual) else np.nan

class RSME(Metric):
    """
    Root Mean Square Error (RMSE) metric.
    """
    name = "RMSE"
    x_min = 0.0
    x_max = 10
    unit = "m3/s"
    anti = False

    def __call__(self, actual, fitted):
        return np.sqrt(np.mean((actual - fitted) ** 2))

class NRSME(Metric):
    """
    Normalized Root Mean Square Error (NRMSE) metric.
    Normalization is done using the interquartile range (IQR).
    """
    name = "NRSME"
    x_min = 0.0
    x_max = 1
    unit = "unit-less"
    anti = False

    def __call__(self, actual, fitted):
        rmse_value = np.sqrt(np.mean((actual - fitted) ** 2))
        iqr = np.percentile(actual, 75) - np.percentile(actual, 25)
        return rmse_value / iqr if iqr != 0 else np.nan


class MAPE(Metric):
    """
    Mean Absolute Percentage Error (MAPE) metric.
    """
    name = "MAPE"
    x_min = 0.0
    x_max = 100.0
    unit = "%"
    anti = False

    def __call__(self, actual, fitted):
        return np.mean(np.abs((fitted - actual) / actual)) * 100 if np.any(actual) else np.nan

class MedAPE(Metric):
    """
    Median Absolute Percentage Error (MedAPE) metric.
    """
    name = "MedAPE"
    x_min = 0.0
    x_max = 100.0
    unit = "%"
    anti = False


    def __call__(self, actual, fitted):
        return np.median(np.abs((actual - fitted) / actual)) * 100 if np.any(actual) else np.nan

class SMAPE(Metric):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) metric.
    """
    name = "SMAPE"
    x_min = 0.0
    x_max = 100.0
    unit = "%"
    anti = False

    def __call__(self, actual, fitted):
        denominator = (np.abs(actual) + np.abs(fitted)) / 2
        return np.mean(np.abs((actual - fitted) / denominator)) * 100 if np.any(denominator) else np.nan

class SMedAPE(Metric):
    """
    Symmetric Median Absolute Percentage Error (MedSMAPE) metric.
    """
    name = "SMedAPE"
    x_min = 0.0
    x_max = 100.
    unit = "%"
    anti = False

    def __call__(self, actual, fitted):
        denominator = (np.abs(actual) + np.abs(fitted)) / 2
        return np.median(np.abs((actual - fitted) / denominator)) * 100 if np.any(denominator) else np.nan


METRICS = [NSE(), PBIAS(), APBIAS(), RSME(), NRSME(), MAPE(), MedAPE(), SMAPE(), SMedAPE()]
METRICS_names = [m.name for m in METRICS]

METRICS_dict = {m.name: m for m in METRICS}