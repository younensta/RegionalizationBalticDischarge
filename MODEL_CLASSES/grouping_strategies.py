from abstract_models import GroupingStrategy
import pandas as pd

class TemporalGrouping(GroupingStrategy):
    """
    A grouping strategy that groups data by time.
    This class inherits from the GroupingStrategy base class.
    """

    def __init__(self, temporal_column: str):
        self.temporal_column = temporal_column
        super().__init__(name=f"Temporal:{temporal_column}", attributes=[temporal_column])

    def create_groups(self, train_df):
        # Check if temporal column is in regular columns
        if self.temporal_column in train_df.columns:
            temporal_data = train_df[self.temporal_column]
        # Check if temporal column is in the MultiIndex
        elif isinstance(train_df.index, pd.MultiIndex) and self.temporal_column in train_df.index.names:
            temporal_data = train_df.index.get_level_values(self.temporal_column)
        else:
            raise ValueError(f"The DataFrame must contain a '{self.temporal_column}' column or index level for temporal grouping.")

        # Create a copy to avoid modifying the original
        train_df_copy = train_df.copy()
        
        # Create a new column for the group
        train_df_copy['TEMPORAL_GROUP'] = pd.Categorical(temporal_data).codes
        return train_df_copy, 'TEMPORAL_GROUP'

    def predict_group(self, test_df):
        # Check if temporal column is in regular columns
        if self.temporal_column in test_df.columns:
            temporal_data = test_df[self.temporal_column]
        # Check if temporal column is in the MultiIndex
        elif isinstance(test_df.index, pd.MultiIndex) and self.temporal_column in test_df.index.names:
            temporal_data = test_df.index.get_level_values(self.temporal_column)
        else:
            raise ValueError(f"The DataFrame must contain a '{self.temporal_column}' column or index level for temporal grouping.")

        # Create a copy to avoid modifying the original
        test_df_copy = test_df.copy()
        
        # Create a new column for the group
        test_df_copy['TEMPORAL_GROUP'] = pd.Categorical(temporal_data).codes
        return test_df_copy, 'TEMPORAL_GROUP'