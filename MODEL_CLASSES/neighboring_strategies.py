from abstract_models import NeighboringStrategy

import pandas as pd
import geopandas as gpd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances


import warnings
from typing import Optional, List

class SpatialNeighboring(NeighboringStrategy):
    """
    A neighboring strategy that uses spatial data to find neighboring stations.
    A gdf containing the centroids of the training and prediction basins is required.
    This class inherits from the NeighboringStrategy base class.
    The geodataframe must contain 'ID' and 'geometry' columns.
    The 'ID' column is used to identify the stations, and the 'geometry' column
    The CRS Has to be the same for both and a valid CRS for distance calculations in meters. (ESPG:3395 recommended)
    """

    def __init__(self, train_centroids_gdf: gpd.GeoDataFrame, pred_centroids_gdf: gpd.GeoDataFrame, n_neighbors: int = 5):
        super().__init__('Spatial_Neighboring', n_neighbors, basin_wise=True)
        self.train_centroids_gdf = train_centroids_gdf
        self.pred_centroids_gdf = pred_centroids_gdf
        
        #check if the GeoDataFrames are valid, contains 'ID' and 'geometry' columns
        if not all(col in self.train_centroids_gdf.columns for col in ['ID', 'geometry']):
            raise ValueError("Training basins centroids GeoDataFrame must contain 'ID' and 'geometry' columns.")
        if not all(col in self.pred_centroids_gdf.columns for col in ['ID', 'geometry']):
            raise ValueError("Prediction basins centroids GeoDataFrame must contain 'ID' and 'geometry' columns.")
        # Check if the CRS is valid and the same for both GeoDataFrames
        if self.train_centroids_gdf.crs is None or self.pred_centroids_gdf.crs is None:
            raise ValueError("Both train and test GeoDataFrames must have a valid CRS.")
        if self.train_centroids_gdf.crs != self.pred_centroids_gdf.crs:
            raise ValueError("Train and test GeoDataFrames must have the same CRS.")
        # Ensure the GeoDataFrames are in a valid CRS for distance calculations
        if not self.train_centroids_gdf.crs.is_geographic and not self.train_centroids_gdf.crs.is_projected:
            warnings.warn("The CRS is not geographic or projected. Distance calculations may not be accurate. Use EPSG:3395 for instance.")

        self.distances: Optional[pd.DataFrame] = None

        self.distances = self._compute_distance_matrix()
        
    def _compute_distance_matrix(self):
        """
        Calculate the distance matrix between training and test centroids.
        """
        if self.train_centroids_gdf.empty or self.pred_centroids_gdf.empty:
            raise ValueError("Training or prediction basins centroids GeoDataFrame is empty.")
        
        # Ensure both GeoDataFrames have the same coordinate reference system (CRS)
        if self.train_centroids_gdf.crs != self.pred_centroids_gdf.crs:
            self.pred_centroids_gdf = self.pred_centroids_gdf.to_crs(self.train_centroids_gdf.crs)
        
        matrix = self.pred_centroids_gdf.geometry.apply(lambda x: self.train_centroids_gdf.geometry.distance(x))
        
        distance_matrix = pd.DataFrame(matrix.values, index=self.pred_centroids_gdf['ID'], columns=self.train_centroids_gdf['ID'])
        distance_matrix.index.name = 'Test_ID'
        distance_matrix.columns.name = 'Train_ID'

        return distance_matrix
    
    def find_neighbors(self, group_df: pd.DataFrame, test_id: str, test_df: pd.DataFrame):
        """
        Get the nearest neighbors for a given test station ID within the training data.
        """
        if self.distances is None:
            raise RuntimeError("Distance matrix must be computed before getting neighbors.")
        

        if test_id not in self.distances.index:
            raise ValueError(f"Test ID '{test_id}' not found in distance matrix.")
        
        #find the ids in the index of group_df
        train_ids = group_df.index.get_level_values('ID').unique()
        
        if train_ids.empty:
            warnings.warn("No training IDs found in the group DataFrame.")
            return []

        for id in train_ids:
            if id not in self.distances.columns:
                raise ValueError(f"Train ID '{id}' not found in distance matrix.")

        distances = self.distances.loc[test_id, train_ids]

        
        # Sort by distance and get the n_neighbors closest training stations
        neighbors = distances.nsmallest(self.k).index

        return neighbors
    
class EuclidianNeighbors(NeighboringStrategy):
    """
    A neighboring strategy that uses Euclidean distance on standardized prodictors to find neighboring stations.
    """

    def __init__(self, columns: List[str], n_neighbors: int=30):
        super().__init__(f'Euclidian_Neighbors:{columns}', n_neighbors, basin_wise=False, col_to_check=columns)
        self.columns = columns
        self.scaler = StandardScaler()


    def find_neighbors(self, group_df:pd.DataFrame, target_id, test_df:pd.DataFrame):
        # Select the target row as a DataFrame to maintain 2D shape
        row = test_df.loc[[target_id], self.columns]
        
        group_df_features = group_df[self.columns]
        
        # Scale the data
        scaled_group_df = self.scaler.fit_transform(group_df_features)
        scaled_row = self.scaler.transform(row)

        # Calculate distances using scikit-learn's optimized function
        dist_array = euclidean_distances(scaled_row, scaled_group_df).flatten()

        distances = pd.Series(dist_array, index=group_df.index)
        neighbors = distances.nsmallest(self.k).index

        return neighbors
