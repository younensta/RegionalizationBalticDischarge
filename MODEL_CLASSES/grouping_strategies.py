from abstract_models import GroupingStrategy
import pandas as pd
from k_means_constrained import KMeansConstrained
from typing import List

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
        self.groups_done = True
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
   
class KmeansClustering(GroupingStrategy):
    """
    A grouping strategy that uses K-means clustering.
    This class inherits from the GroupingStrategy base class.
    """
    def __init__(self, cluster_column: List[str], n_clusters: int = 3, min_members: int = 10, random_state: int = 42):
        self.cluster_column = cluster_column
        self.min_members = min_members
        self.n_clusters = n_clusters
        self.random_state = random_state
        
        super().__init__(name=f"Kmeans:{cluster_column}", attributes=cluster_column)
        self.model = KMeansConstrained(n_clusters=self.n_clusters, size_min=self.min_members, random_state=self.random_state)

    def create_groups(self, train_df):
        
        train_df_copy = train_df.copy()
        train_df_for_clustering = train_df_copy[self.cluster_column]

        # Fit the KMeans model
        self.model.fit(train_df_for_clustering)
        # Predict clusters
        train_df_copy[f'KMEANS_{self.cluster_column}_GROUP'] = self.model.predict(train_df_for_clustering)
        self.groups_done = True


        return train_df_copy, f'KMEANS_{self.cluster_column}_GROUP'

    def predict_group(self, test_df):
        test_df_copy = test_df.copy()
        test_df_for_clustering = test_df_copy[self.cluster_column]

        # Predict clusters
        test_df_copy[f'KMEANS_{self.cluster_column}_GROUP'] = self.model.predict(test_df_for_clustering)
        return test_df_copy, f'KMEANS_{self.cluster_column}_GROUP'
    