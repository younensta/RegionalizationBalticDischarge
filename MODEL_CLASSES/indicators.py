# This files contains the metrics for evaluating model performance

import numpy as np

def nse(actual, fitted) -> float:
    """
    Calculate the Nash-Sutcliffe Efficiency (NSE) between actual and fitted values.
    """
    numerator = np.sum((actual - fitted) ** 2)
    denominator = np.sum((actual - np.mean(actual)) ** 2)

    if denominator == 0:
        return np.nan  # Avoid division by zero

    return 1 - (numerator / denominator)

def pbias(actual, fitted) -> float:
    """
    Calculate the Percent Bias (PBIAS) between actual and fitted values.
    """
    return 100 * np.sum(fitted - actual) / np.sum(actual)

def rmse(actual, fitted) -> float:
    """
    Calculate the Root Mean Square Error (RMSE) between actual and fitted values.
    """
    return np.sqrt(np.mean((actual - fitted) ** 2))

def nrmse(actual, fitted) -> float:
    """
    Calculate the Normalized Root Mean Square Error (NRMSE) between actual and fitted values.
    Normalization is done using the interquartile range (IQR).
    """
    return rmse(actual, fitted) / (np.percentile(actual, 75) - np.percentile(actual, 25))

def medape(actual, fitted) -> float:
    """
    Calculate the Median Absolute Percentage Error (MedAPE) between actual and fitted values.
    """
    return np.median(np.abs((actual - fitted) / actual)) * 100

def smape(actual, fitted) -> float:
    """
    Calculate the Symmetric Mean Absolute Percentage Error (sMAPE) between actual and fitted values.
    """
    return np.mean(np.abs((actual - fitted) / ((np.abs(actual) + np.abs(fitted)) / 2))) * 100

def medsmape(actual, fitted) -> float:
    """
    Calculate the Median Symmetric Mean Absolute Percentage Error (MedSMAPE) between actual and fitted values.
    """
    return np.median(np.abs((actual - fitted) / ((np.abs(actual) + np.abs(fitted)) / 2))) * 100


def flow_deciles_smape(actual, fitted) -> dict:
    """
    Calculate the sMAPE for each decile of the actual values.
    Returns a dictionary with decile ranges as keys and sMAPE values for different flow deciles as values.
    """
    deciles = np.percentile(actual, np.arange(0, 101, 10))
    smape_values = {}

    for i in range(len(deciles) - 1):
        mask = (actual >= deciles[i]) & (actual < deciles[i + 1])
        if np.any(mask):
            smape_values[f"{deciles[i]:.2f} - {deciles[i + 1]:.2f}"] = smape(actual[mask], fitted[mask])

    return smape_values

def flow_deciles_nse(actual, fitted) -> dict:
    """
    Calculate the NSE for each decile of the actual values.
    Returns a dictionary with decile ranges as keys and NSE values for different flow deciles as values.
    """
    deciles = np.percentile(actual, np.arange(0, 101, 10))
    nse_values = {}

    for i in range(len(deciles) - 1):
        mask = (actual >= deciles[i]) & (actual < deciles[i + 1])
        if np.any(mask):
            nse_values[f"{deciles[i]:.2f} - {deciles[i + 1]:.2f}"] = nse(actual[mask], fitted[mask])

    return nse_values

if __name__ == '__main__':
    # Example usage
    actual = np.array([1, 2, 3, 4, 5])
    fitted = np.array([1.1, 1.9, 3.2, 4.1, 5.0])


    print("NSE:", nse(actual, fitted))
    print("PBIAS:", pbias(actual, fitted))
    print("RMSE:", rmse(actual, fitted))
    print("NRMSE:", nrmse(actual, fitted))
    print("MedAPE:", medape(actual, fitted))
    print("sMAPE:", smape(actual, fitted))
    print("MedSMAPE:", medsmape(actual, fitted))
    print("Flow Deciles sMAPE:", flow_deciles_smape(actual, fitted))
    print("Flow Deciles NSE:", flow_deciles_nse(actual, fitted))

    