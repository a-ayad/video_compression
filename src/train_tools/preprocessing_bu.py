import pandas as pd
import numpy as np
import re
import sys
import os
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

# Make sure we can import from parent directory if needed
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Custom transformer for resolution conversion
class ResolutionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.resolution_mapping = {}
        self.resolution_values = []  # Store actual numerical values for nearest-neighbor matching
        
    def fit(self, X, y=None):
        if 'metrics_resolution' in X.columns:
            # Extract resolution values (pixel counts)
            resolution_values = X['metrics_resolution'].apply(self._extract_resolution)
            
            # Save unique resolutions and their numerical values
            unique_resolutions = sorted(resolution_values.unique())
            self.resolution_values = unique_resolutions
            
            # Create mapping from resolution to ordinal value
            self.resolution_mapping = {res: i+1 for i, res in enumerate(unique_resolutions)}
            print(f"Resolution mapping: {self.resolution_mapping}")
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        if 'metrics_resolution' in X_copy.columns:
            # Extract resolution values
            resolution_values = X_copy['metrics_resolution'].apply(self._extract_resolution)
            
            # Create a new column for the ordinal values, using nearest-neighbor for new values
            X_copy['resolution_ordinal'] = resolution_values.apply(self._get_ordinal_value)
            
            # Drop the original column
            X_copy.drop('metrics_resolution', axis=1, inplace=True)
        return X_copy
    
    def _extract_resolution(self, res_str):
        """Extract total pixel count from resolution string"""
        if not isinstance(res_str, str):
            return res_str
            
        # Check if in format "(width, height)"
        if '(' in res_str and ')' in res_str:
            numbers = re.findall(r'\d+', res_str)
            if len(numbers) >= 2:
                return int(numbers[0]) * int(numbers[1])  # width * height
        
        # Check if in format "widthxheight"
        elif 'x' in res_str:
            parts = res_str.split('x')
            if len(parts) >= 2:
                return int(parts[0]) * int(parts[1])
            
        return res_str
    
    def _get_ordinal_value(self, resolution):
        """Get ordinal value for a resolution, using nearest neighbor for new values"""
        # If resolution exists in mapping, use the predefined value
        if resolution in self.resolution_mapping:
            return self.resolution_mapping[resolution]
        
        # If it's a new value and we have known values to compare with
        if isinstance(resolution, (int, float)) and self.resolution_values:
            # Find the closest resolution value
            closest_resolution = min(self.resolution_values, 
                                    key=lambda x: abs(x - resolution) if isinstance(x, (int, float)) else float('inf'))
            
            # Use the ordinal value of the closest resolution
            closest_value = self.resolution_mapping[closest_resolution]
            print(f"New resolution value {resolution} mapped to closest known value {closest_resolution} (ordinal: {closest_value})")
            return closest_value
        
        # If all else fails, use median value
        median_value = np.median(list(self.resolution_mapping.values()))
        return median_value

# Custom transformer for frame rate (treating it as a numerical feature)
class FrameRateTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique_values = []
        
    def fit(self, X, y=None):
        if 'metrics_frame_rate' in X.columns:
            # Store unique frame rate values for nearest-neighbor matching
            self.unique_values = sorted([x for x in X['metrics_frame_rate'].unique() if isinstance(x, (int, float))])
            print(f"Frame rate unique values: {self.unique_values}")
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        if 'metrics_frame_rate' in X_copy.columns:
            # Handle new values using nearest-neighbor approach
            X_copy['frame_rate_numeric'] = X_copy['metrics_frame_rate'].apply(self._get_nearest_value)
            
            # Drop the original column
            X_copy.drop('metrics_frame_rate', axis=1, inplace=True)
        return X_copy
    
    def _get_nearest_value(self, value):
        """Find the nearest known frame rate value for new values"""
        # If it's already a numeric value, use it directly
        if isinstance(value, (int, float)):
            # If it's a known value, just return it
            if value in self.unique_values:
                return value
                
            # If it's a new value, find the closest known value
            if self.unique_values:
                closest_value = min(self.unique_values, key=lambda x: abs(x - value))
                print(f"New frame rate value {value} mapped to closest known value {closest_value}")
                return closest_value
        
        # If we can't process it, use the median
        if self.unique_values:
            return np.median(self.unique_values)
        return 30.0  # Fallback default

# Custom transformer for CQ (treating it as a numerical feature)
class CQTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.unique_values = []
        
    def fit(self, X, y=None):
        if 'cq' in X.columns:
            # Store unique CQ values for nearest-neighbor matching
            self.unique_values = sorted([x for x in X['cq'].unique() if isinstance(x, (int, float))])
            print(f"CQ unique values: {self.unique_values}")
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        if 'cq' in X_copy.columns:
            # Handle new values using nearest-neighbor approach
            X_copy['cq_numeric'] = X_copy['cq'].apply(self._get_nearest_value)
            
            # Drop the original column
            X_copy.drop('cq', axis=1, inplace=True)
        return X_copy
    
    def _get_nearest_value(self, value):
        """Find the nearest known CQ value for new values"""
        # If it's already a numeric value, use it directly
        if isinstance(value, (int, float)):
            # If it's a known value, just return it
            if value in self.unique_values:
                return value
                
            # If it's a new value, find the closest known value
            if self.unique_values:
                closest_value = min(self.unique_values, key=lambda x: abs(x - value))
                print(f"New CQ value {value} mapped to closest known value {closest_value}")
                return closest_value
        
        # If we can't process it, use the median
        if self.unique_values:
            return np.median(self.unique_values)
        return 30.0  # Fallback default

# Custom VMAF scaler
class VMAFScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.min_val = None
        self.max_val = None
        
    def fit(self, X, y=None):
        if 'vmaf' in X.columns:
            self.min_val = X['vmaf'].min()
            self.max_val = X['vmaf'].max()
            print(f"VMAF min: {self.min_val}, max: {self.max_val}")
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        if 'vmaf' in X_copy.columns and self.min_val is not None and self.max_val is not None:
            X_copy['vmaf'] = (X_copy['vmaf'] - self.min_val) / (self.max_val - self.min_val)
            X_copy['vmaf'] = X_copy['vmaf'].clip(0, 1)
        return X_copy

# Custom transformer to select features for min-max scaling
class CustomMinMaxScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale
        self.scaler = MinMaxScaler()
        
    def fit(self, X, y=None):
        # Get columns to scale that exist in the dataframe
        existing_columns = [col for col in self.columns_to_scale if col in X.columns]
        if existing_columns:
            self.scaler.fit(X[existing_columns])
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        existing_columns = [col for col in self.columns_to_scale if col in X_copy.columns]
        
        if existing_columns:
            X_copy[existing_columns] = self.scaler.transform(X_copy[existing_columns])
        return X_copy

# Custom transformer to select features for standard scaling
class CustomStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale):
        self.columns_to_scale = columns_to_scale
        self.scaler = StandardScaler()
        
    def fit(self, X, y=None):
        # Get columns to scale that exist in the dataframe
        existing_columns = [col for col in self.columns_to_scale if col in X.columns]
        if existing_columns:
            self.scaler.fit(X[existing_columns])
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        existing_columns = [col for col in self.columns_to_scale if col in X_copy.columns]
        
        if existing_columns:
            X_copy[existing_columns] = self.scaler.transform(X_copy[existing_columns])
        return X_copy

# Transformer to drop unwanted columns
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns_to_drop:
            if col in X_copy.columns:
                X_copy.drop(col, axis=1, inplace=True)
        return X_copy

# Main function to create and fit the preprocessing pipeline
def create_preprocessing_pipeline(input_file=None, output_file=None, pipeline_file=None):
    """
    Create and optionally fit a preprocessing pipeline.
    
    Args:
        input_file: CSV file to fit the pipeline on (None for creating an unfitted pipeline)
        output_file: File to save the preprocessed data to
        pipeline_file: File to save the fitted pipeline to
    
    Returns:
        The preprocessing pipeline (fitted if input_file was provided)
    """
    # Columns to drop from the dataset
    columns_to_drop = ['metrics_scene_change_count', 'video_name', 'output_size', 'bitrate']
    
    # Columns to apply min-max scaling
    minmax_columns = ['cq_numeric', 'frame_rate_numeric', 'resolution_ordinal']
    
    # Define the preprocessing steps
    pipeline = Pipeline([
        ('column_dropper', ColumnDropper(columns_to_drop)),
        ('resolution_transformer', ResolutionTransformer()),
        ('frame_rate_transformer', FrameRateTransformer()),
        ('cq_transformer', CQTransformer()),  # Keep CQ as numeric
        ('vmaf_scaler', VMAFScaler()),
        ('minmax_scaler', CustomMinMaxScaler(minmax_columns)),
        # Note: We don't apply standard scaling to any columns since CQ is now numerically scaled
    ])
    
    # If input file is provided, fit the pipeline
    if input_file:
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Fit the pipeline
        print("Fitting preprocessing pipeline...")
        pipeline.fit(df)
        
        # If output file is provided, save the preprocessed data
        if output_file:
            print(f"Applying pipeline and saving preprocessed data to {output_file}...")
            transformed_df = pipeline.transform(df)
            transformed_df.to_csv(output_file, index=False)
        
        # Save the pipeline if a filename is provided
        if pipeline_file:
            print(f"Saving preprocessing pipeline to {pipeline_file}")
            with open(pipeline_file, 'wb') as f:
                pickle.dump(pipeline, f)
    
    return pipeline

# Function to load a saved pipeline
def load_preprocessing_pipeline(pipeline_file):
    """
    Load a saved preprocessing pipeline.
    
    Args:
        pipeline_file: File containing the saved pipeline
    
    Returns:
        The loaded preprocessing pipeline
    """
    print(f"Loading preprocessing pipeline from {pipeline_file}")
    with open(pipeline_file, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

# Function to preprocess new data using a saved pipeline
def preprocess_new_data(df, pipeline=None, pipeline_file=None):
    """
    Preprocess new data using a fitted pipeline.
    
    Args:
        df: DataFrame containing new data to preprocess
        pipeline: Fitted preprocessing pipeline (if provided)
        pipeline_file: File containing saved pipeline (used if pipeline is None)
    
    Returns:
        Preprocessed DataFrame
    """
    # Load pipeline from file if not provided directly
    if pipeline is None and pipeline_file:
        pipeline = load_preprocessing_pipeline(pipeline_file)
    
    if pipeline is None:
        raise ValueError("Either pipeline or pipeline_file must be provided")
    
    # Apply the pipeline
    return pipeline.transform(df)

# Example usage
if __name__ == "__main__":
    cwd = os.getcwd()
    path = os.path.join(cwd, 'src', 'data')
    input_file = os.path.join(path, 'merged_results.csv')
    output_file = os.path.join(path, 'preprocessed_data.csv')
    pipeline_file = os.path.join(path, 'preprocessing_pipeline.pkl')
    
    # Create and fit the pipeline, save preprocessed data and pipeline
    pipeline = create_preprocessing_pipeline(input_file, output_file, pipeline_file)
    
    # Example of how this would be used for inference later:
    print("\n--- Example: Preprocessing New Data for Inference ---")
    # Load the pipeline
    loaded_pipeline = load_preprocessing_pipeline(pipeline_file)
    
    # Create sample new data (here we're just using the first row of our original data as an example)
    print("Loading sample data for inference...")
    sample_df = pd.read_csv(input_file, nrows=1)
    print("Sample data:")
    print(sample_df)
    
    # Preprocess the sample data
    print("\nApplying preprocessing for inference...")
    preprocessed_sample = preprocess_new_data(sample_df, loaded_pipeline)
    print("Preprocessed sample data:")
    print(preprocessed_sample)
    
    print("\nPreprocessing pipeline successfully created and tested!")