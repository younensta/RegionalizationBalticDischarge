from abstract_models import BaseModel

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler


class OlsLogMLR(BaseModel):
    """
    A model that performs ordinary least squares (OLS) multiple linear regression.
    This class inherits from the Model base class.
    """

    def __init__(self, predictors:list = None):
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
        X = sm.add_constant(X)

        # Fit the model using OLS
        self.model = sm.OLS(y, X).fit()
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
        logger.info("Preparing test data for prediction...")

        # Prepare the test data
        
        X_test = df_test[self.predictors]
        X_test = StandardScaler().fit_transform(X_test)
        X_test = sm.add_constant(X_test)

        # Make predictions
        predictions = self.model.predict(X_test)
        # Inverse log transformation to get the original scale
        predictions = np.exp(predictions) * df_test['A']
        
        res_df = df_test[['Q']].copy()
        res_df.loc[:, 'Q_sim'] = predictions

        return res_df
