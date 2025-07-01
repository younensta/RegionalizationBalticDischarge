from abstract_models import BaseModel

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd

from typing import List

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class OlsLogMLR(BaseModel):
    """
    A model that performs ordinary least squares (OLS) multiple linear regression.
    This class inherits from the Model base class.
    """

    def __init__(self, predictors:List[str] = None):
        super().__init__('OLS_Log_MLR:', predictors)
        self.model = None


    def fit(self, df_train):
        """
        Fit the model to the training data.
        Uses normalization and logarithmic transformation on the target variable.
        """        

        logger.info("Fitting the model...")

        # Log-transform the target variable 'Q' (specific discharge)
        y = np.log(df_train['Q']/df_train['A'])
        
        # Add a constant for the intercept
        X = df_train[self.predictors]
        X = StandardScaler().fit_transform(X)  # Normalize the predictors

        # Fit the model using OLS
        self.model = LinearRegression(y, X).fit()
        self._is_fitted = True

        logger.info("Model fitted successfully.")
        logger.info(self.model.summary())


    def predict(self, df_test):
        """
        Predict using the model.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        # Clean the test DataFrame
        logger.info("Predicting with OLS Log MLR model...")

        # Prepare the test data
        
        X_test = df_test[self.predictors]
        X_test = StandardScaler().fit_transform(X_test)

        # Make predictions
        predictions = self.model.predict(X_test)
        # Inverse log transformation to get the original scale
        predictions = np.exp(predictions) * df_test['A']
        
        res_df = df_test[['Q']].copy()
        res_df.loc[:, 'Q_sim'] = predictions

        return res_df


class LogRF(BaseModel):
    """
    A model that performs random forest regression.
    This class inherits from the Model base class.
    """

    def __init__(self, predictors:List[str], n_trees:int=30, max_depth:int=10, random_state:int=42):
        super().__init__('RF:', predictors)
        
        self.model = RandomForestRegressor(n_estimators=n_trees, max_depth=max_depth, random_state=random_state)
        self._is_fitted = False

    def fit(self, df_train: pd.DataFrame):
        """
        Fit the model to the training data.
        """
        logger.info("Fitting the Random Forest model...")

        # Prepare the training data
        X_train = df_train[self.predictors]
        X_train = StandardScaler().fit_transform(X_train)  # Normalize the predictors

        y_train = np.log(df_train['Q']/df_train['A'])  # Log-transform the target variable

        # Fit the model
        self.model.fit(X_train, y_train)
        self._is_fitted = True

        logger.info("Random Forest model fitted successfully.")

    def predict(self, df_test: pd.DataFrame):
        """
        Predict using the model.
        """
        if not self._is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        logger.info("Predicting with the Random Forest model...")

        # Prepare the test data
        X_test = df_test[self.predictors]
        X_test = StandardScaler().fit_transform(X_test)

        # Make predictions
        predictions = np.exp(self.model.predict(X_test))* df_test['A']  # Inverse log transformation to get the original scale

        res_df = df_test[['Q']].copy()
        res_df.loc[:, 'Q_sim'] = predictions

        return res_df
