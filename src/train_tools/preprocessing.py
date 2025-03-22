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

# Fixed ResolutionTransformer with better parsing
class ResolutionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.resolution_mapping = {}
        self.resolution_values = []
        self.default_resolutions = {
            "720p": 1280*720,
            "1080p": 1920*1080,
            "1440p": 2560*1440,
            "4K": 3840*2160
        }
        
    def fit(self, X, y=None):
        if 'metrics_resolution' in X.columns:
            # First, print some sample values to understand the format
            sample_values = X['metrics_resolution'].head(5).tolist()
            print(f"Sample resolution values: {sample_values}")
            
            # Extract resolution values
            resolution_values = []
            for val in X['metrics_resolution']:
                res = self._extract_resolution(val)
                if isinstance(res, (int, float)) and res > 0:
                    resolution_values.append(res)
            
            # If we found valid resolutions, create mapping
            if resolution_values:
                unique_resolutions = sorted(set(resolution_values))
                self.resolution_values = unique_resolutions
                self.resolution_mapping = {res: i+1 for i, res in enumerate(unique_resolutions)}
                print(f"Found {len(unique_resolutions)} unique resolutions")
                print(f"Resolution mapping: {self.resolution_mapping}")
            else:
                # If no valid resolutions found, use default mapping
                print("No valid resolutions found, using default mapping")
                self.resolution_values = sorted(self.default_resolutions.values())
                self.resolution_mapping = {res: i+1 for i, res in enumerate(self.resolution_values)}
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        if 'metrics_resolution' in X_copy.columns:
            # Print some sample resolution values before transformation
            sample_before = X_copy['metrics_resolution'].head(3).tolist()
            print(f"Sample resolutions before transformation: {sample_before}")
            
            # Extract resolution values and convert to ordinal 
            extracted_values = []
            ordinal_values = []
            
            for val in X_copy['metrics_resolution']:
                res = self._extract_resolution(val)
                extracted_values.append(res)
                ord_val = self._get_ordinal_value(res)
                ordinal_values.append(ord_val)
            
            # Print some sample extracted and ordinal values
            print(f"Sample extracted resolutions: {extracted_values[:3]}")
            print(f"Sample ordinal values: {ordinal_values[:3]}")
            
            # Create the ordinal values to the resolution column
            X_copy['metrics_resolution'] = ordinal_values
            
        return X_copy
    
    def _extract_resolution(self, res_str):
        """Extract total pixel count from resolution string"""
        if not isinstance(res_str, str):
            return 0
        
        # Try to match patterns like (1280, 720)
        match = re.search(r'\((\d+),\s*(\d+)\)', res_str)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width * height
        
        # Try to match patterns like 1280x720
        match = re.search(r'(\d+)[xX](\d+)', res_str)
        if match:
            width = int(match.group(1))
            height = int(match.group(2))
            return width * height
            
        # Try to extract 720p, 1080p, etc.
        for name, pixels in self.default_resolutions.items():
            if name in res_str:
                return pixels
        
        # If no pattern matches, return default
        return 0
    
    def _get_ordinal_value(self, resolution):
        """Get ordinal value for a resolution, using nearest neighbor for new values"""
        # If resolution is invalid, use the smallest valid value
        if not isinstance(resolution, (int, float)) or resolution <= 0:
            if self.resolution_values:
                return 1  # First ordinal value
            return 1  # Default if no mapping exists
        
        # If resolution exists in mapping, use the predefined value
        if resolution in self.resolution_mapping:
            return self.resolution_mapping[resolution]
        
        # If it's a new value and we have known values to compare with
        if self.resolution_values:
            # Find the closest resolution value
            closest_resolution = min(self.resolution_values, 
                                   key=lambda x: abs(x - resolution))
            
            # Use the ordinal value of the closest resolution
            closest_value = self.resolution_mapping[closest_resolution]
            print(f"New resolution value {resolution} mapped to closest known value {closest_resolution} (ordinal: {closest_value})")
            return closest_value
        
        # If we have no mapping at all, return 1 as a default
        return 1

# NumericFeatureTransformer (for frame rate and other numeric features)
class NumericFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, feature_name, output_column=None):
        self.feature_name = feature_name
        self.output_column = output_column or f"{feature_name}"
        self.unique_values = []
        
    def fit(self, X, y=None):
        if self.feature_name in X.columns:
            # First make sure values are numeric (force conversion)
            numeric_values = pd.to_numeric(X[self.feature_name], errors='coerce')
            
            # Store all unique values, not just float/int types
            unique_vals = sorted([float(x) for x in numeric_values.dropna().unique()])
            self.unique_values = unique_vals
            
            print(f"{self.feature_name} unique values: {self.unique_values}")
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        if self.feature_name in X_copy.columns:
            # First, convert to numeric (force conversion)
            X_copy[self.feature_name] = pd.to_numeric(X_copy[self.feature_name], errors='coerce')
            
            # Debug: original values before transformation
            print(f"Original {self.feature_name}: {X_copy[self.feature_name].head(3).tolist()}")
            
            # Apply the nearest-neighbor transformation
            X_copy[self.output_column] = X_copy[self.feature_name].apply(self._get_numeric_value)
            
            # Debug: transformed values
            print(f"Transformed {self.feature_name} to {self.output_column}: {X_copy[self.output_column].head(3).tolist()}")
            
        return X_copy
    
    def _get_numeric_value(self, value):
        """Get or calculate numeric value for the feature"""
        # If we don't have any valid values from fit, just return the value itself
        if not self.unique_values:
            # Handle the case where fit() didn't store any valid values
            if pd.isna(value):
                return 30.0  # Default fallback for NaN
            return float(value)
        
        # Normal case - we have valid unique values
        if pd.isna(value):
            # For missing/NaN values, use median of known values
            return float(np.median(self.unique_values))
            
        # Convert value to float for comparison
        try:
            float_val = float(value)
            
            # If value exists in our known values, return its ordinal index
            if float_val in self.unique_values:
                ordinal_index = self.unique_values.index(float_val) + 1
                #print(f"Value {float_val} mapped to ordinal index {ordinal_index}")
                return ordinal_index
                
            # Otherwise find nearest known value
            closest_val = min(self.unique_values, key=lambda x: abs(x - float_val))
            ordinal_index = self.unique_values.index(closest_val) + 1
            print(f"New {self.feature_name} value {float_val} mapped to closest known value {closest_val} (ordinal index: {ordinal_index})")
            return ordinal_index
            
        except (ValueError, TypeError):
            # If conversion fails, use median
            return float(np.median(self.unique_values))

# NEW: VMAFScaler specifically for scaling the target column
class VMAFScaler(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='vmaf'):
        self.target_column = target_column
        self.min_val = None
        self.max_val = None
        
    def fit(self, X, y=None):
        if self.target_column in X.columns:
            # Convert to numeric if needed
            X_target = pd.to_numeric(X[self.target_column], errors='coerce')
            self.min_val = X_target.min()
            self.max_val = X_target.max()
            print(f"VMAF scaling range: min={self.min_val}, max={self.max_val}")
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        if self.target_column in X_copy.columns and self.min_val is not None and self.max_val is not None:
            # Convert to numeric if needed
            X_copy[self.target_column] = pd.to_numeric(X_copy[self.target_column], errors='coerce')
            
            # Print original values for debugging
            print(f"Original {self.target_column} values (first 3): {X_copy[self.target_column].head(3).tolist()}")
            
            # Apply min-max scaling to VMAF
            X_copy[self.target_column] = (X_copy[self.target_column] - self.min_val) / (self.max_val - self.min_val)
            
            # Print scaled values for debugging
            print(f"Scaled {self.target_column} values (first 3): {X_copy[self.target_column].head(3).tolist()}")
        return X_copy
    
    def inverse_transform(self, X):
        """Convert scaled VMAF back to original scale"""
        X_copy = X.copy()
        if self.target_column in X_copy.columns and self.min_val is not None and self.max_val is not None:
            X_copy[self.target_column] = X_copy[self.target_column] * (self.max_val - self.min_val) + self.min_val
        return X_copy


# Target Extractor - Only for extracting VMAF as target for model training
class TargetExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, target_column='vmaf', vmaf_scaler=None):
        self.target_column = target_column
        self.vmaf_scaler = vmaf_scaler
        
    def fit(self, X, y=None):
        return self
        
    def transform(self, X):
        # Return X unchanged - this only extracts target
        return X
        
    def get_target(self, X):
        """Extract the target variable (assumes VMAF has already been scaled)"""
        if self.target_column in X.columns:
            y = pd.to_numeric(X[self.target_column], errors='coerce')
            return y
        return None

# Feature scaler for multiple features - EXCLUDES the target variable
class FeatureScaler(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_scale, scaling_type='minmax', excluded_columns=None):
        self.columns_to_scale = columns_to_scale
        self.scaling_type = scaling_type
        self.excluded_columns = excluded_columns or []
        
        if scaling_type == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_type == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown scaling type: {scaling_type}")
            
        # Store fit parameters
        self.feature_min = {}
        self.feature_max = {}
        self.feature_mean = {}
        self.feature_std = {}
        
    def fit(self, X, y=None):
        # Get columns to scale that exist in the dataframe
        existing_columns = [col for col in self.columns_to_scale 
                           if col in X.columns and col not in self.excluded_columns]
        
        if existing_columns:
            print(f"Fitting {self.scaling_type} scaler to: {existing_columns}")
            
            # Store stats for each feature independently
            for col in existing_columns:
                # Convert to numeric if needed
                X[col] = pd.to_numeric(X[col], errors='coerce')
                
                if X[col].nunique() > 1:  # Only store if we have multiple unique values
                    self.feature_min[col] = float(X[col].min())
                    self.feature_max[col] = float(X[col].max())
                    self.feature_mean[col] = float(X[col].mean())
                    self.feature_std[col] = float(X[col].std())
                    print(f"  {col}: min={self.feature_min[col]}, max={self.feature_max[col]}, mean={self.feature_mean[col]:.4f}, std={self.feature_std[col]:.4f}")
                else:
                    # For columns with only one unique value, set a range of [0, 1] 
                    # to avoid division by zero in scaling
                    single_value = float(X[col].iloc[0])
                    self.feature_min[col] = single_value - 0.5  # Create artificial range
                    self.feature_max[col] = single_value + 0.5  # around the single value
                    self.feature_mean[col] = single_value
                    self.feature_std[col] = 1.0
                    print(f"  {col}: only one unique value ({single_value}), using artificial range [{self.feature_min[col]}, {self.feature_max[col]}]")
            
            # Fit the sklearn scaler too (not used directly, but keeping for compatibility)
            self.scaler.fit(X[existing_columns])
            
        return self
        
    def transform(self, X):
        X_copy = X.copy()
        existing_columns = [col for col in self.columns_to_scale 
                           if col in X_copy.columns and col not in self.excluded_columns]
        
        if existing_columns:
            # Print values before scaling for debugging
            for col in existing_columns:
                print(f"{col} before {self.scaling_type} scaling: {X_copy[col].head(3).tolist()}")
            
            # Apply manual scaling to ensure it works correctly
            for col in existing_columns:
                # Convert to numeric if needed
                X_copy[col] = pd.to_numeric(X_copy[col], errors='coerce')
                
                # Skip columns where min == max (would cause division by zero)
                if col in self.feature_min and col in self.feature_max:
                    min_val = self.feature_min[col]
                    max_val = self.feature_max[col]
                    
                    # Check if we need to handle out-of-range values
                    out_of_range = (X_copy[col] < min_val).any() or (X_copy[col] > max_val).any()
                    if out_of_range:
                        print(f"Warning: {col} has values outside of training range [{min_val}, {max_val}]")
                        
                    if min_val == max_val:
                        # For columns where all values are the same,
                        # map to middle of range (0.5) instead of 0
                        print(f"Warning: {col} has min=max={min_val}, setting to 0.5")
                        X_copy[col] = 0.5
                    else:
                        # Apply min-max scaling manually
                        X_copy[col] = (X_copy[col] - min_val) / (max_val - min_val)
                        
                        # Handle out-of-range values by clipping to [0, 1]
                        X_copy[col] = X_copy[col].clip(0, 1)
                else:
                    # If we don't have min/max for this column, leave it as is
                    print(f"Warning: No min/max values for {col}, skipping scaling")
            
            # Print values after scaling for debugging
            for col in existing_columns:
                print(f"{col} after {self.scaling_type} scaling: {X_copy[col].head(3).tolist()}")
                
        return X_copy

# Main function to create and fit the preprocessing pipeline for regression
def create_preprocessing_pipeline(input_file=None, output_file=None, pipeline_file=None, target_column='vmaf'):
    # Columns to drop from the dataset
    columns_to_drop = ['metrics_scene_change_count', 'video_name', 'output_size', 'bitrate']
    
    # Features to scale with min-max
    feature_columns = [
        'cq',
        'metrics_frame_rate', 
        'metrics_resolution',
        'metrics_avg_motion', 
        'metrics_avg_edge_density',
        'metrics_avg_texture', 
        'metrics_avg_temporal_information',
        'metrics_avg_spatial_information', 
        'metrics_avg_color_complexity',
        'metrics_avg_motion_variance', 
        'metrics_avg_grain_noise'
    ]
    
    # Create the VMAF scaler first so we can reference it
    vmaf_scaler = VMAFScaler(target_column)
    
    # Define the preprocessing steps
    pipeline = Pipeline([
        ('column_dropper', ColumnDropper(columns_to_drop)),
        ('resolution_transformer', ResolutionTransformer()),
        ('frame_rate_transformer', NumericFeatureTransformer('metrics_frame_rate')),
        # Scale VMAF explicitly
        ('vmaf_scaler', vmaf_scaler),
        # Scale all features EXCEPT vmaf (which is already scaled)
        ('feature_scaler', FeatureScaler(feature_columns, 
                                         scaling_type='minmax',
                                         excluded_columns=[target_column])),
        # Extract target info but don't transform it (it's already scaled)
        ('target_extractor', TargetExtractor(target_column, vmaf_scaler)),
    ])
    
    # Variables to store training data and target
    X_train = None
    y_train = None
    
    # If input file is provided, fit the pipeline
    if input_file:
        print(f"Reading data from {input_file}...")
        df = pd.read_csv(input_file)
        
        # Fit the pipeline
        print("Fitting preprocessing pipeline...")
        pipeline.fit(df)
        
        # Transform the data
        transformed_df = pipeline.transform(df)
        
        # Verify VMAF has been scaled properly
        if target_column in transformed_df.columns:
            vmaf_min = transformed_df[target_column].min()
            vmaf_max = transformed_df[target_column].max()
            print(f"Scaled VMAF range in output: min={vmaf_min:.6f}, max={vmaf_max:.6f}")
            
            if vmaf_min < 0.001 or vmaf_max > 0.999:
                print("Warning: VMAF scaling may not be working correctly!")
        
        # If output file is provided, save the preprocessed data
        if output_file:
            print(f"Saving preprocessed data to {output_file}...")
            transformed_df.to_csv(output_file, index=False)
        
        # Extract the target variable for model training
        target_extractor = pipeline.named_steps['target_extractor']
        y_train = target_extractor.get_target(transformed_df)
        X_train = transformed_df
        
        if pipeline_file:
            print(f"Saving preprocessing pipeline to {pipeline_file}")
            with open(pipeline_file, 'wb') as f:
                pickle.dump(pipeline, f)
    
    return pipeline, X_train, y_train

# Function to preprocess new data for inference
def preprocess_new_data(df, pipeline=None, pipeline_file=None, return_target=False):
    """
    Preprocess new data using a fitted pipeline.
    
    Args:
        df: DataFrame containing new data to preprocess
        pipeline: Fitted preprocessing pipeline
        pipeline_file: Path to saved pipeline file (used if pipeline is None)
        return_target: Whether to return the target variable as well
        
    Returns:
        Preprocessed DataFrame or tuple of (X, y) if return_target is True
    """
    # Load pipeline from file if not provided directly
    if pipeline is None and pipeline_file:
        print(f"Loading preprocessing pipeline from {pipeline_file}")
        with open(pipeline_file, 'rb') as f:
            pipeline = pickle.load(f)
    
    if pipeline is None:
        raise ValueError("Either pipeline or pipeline_file must be provided")
    
    # Apply the pipeline to get preprocessed features
    X = pipeline.transform(df)
    
    # Extract the target if requested
    y = None
    if return_target and 'target_extractor' in pipeline.named_steps:
        target_extractor = pipeline.named_steps['target_extractor']
        y = target_extractor.get_target(X)
    
    if return_target:
        return X, y
    return X




# Function to find optimal CQ for target VMAF 
def find_optimal_cq(trained_model, target_vmaf, video_features, min_cq=10, max_cq=58, 
                   pipeline=None, margin=0.5, verbose=True):
    """
    Find the highest CQ value that meets or exceeds the target VMAF score.
    
    Args:
        trained_model: Model that predicts VMAF from features including CQ
        target_vmaf: Desired VMAF score (0-1 scale if VMAF was scaled)
        video_features: DataFrame containing video features (excluding CQ)
        min_cq: Minimum CQ to consider (typically 10)
        max_cq: Maximum CQ to consider (typically 58)
        pipeline: Preprocessing pipeline used for feature scaling
        margin: Safety margin to ensure we meet the quality threshold
        verbose: Whether to print debugging information
        
    Returns:
        Tuple of (optimal_cq, predicted_vmaf, is_reliable)
    """
    if pipeline is None:
        raise ValueError("Pipeline is required for feature scaling")
    
    # Get scaling parameters for CQ
    if 'feature_scaler' not in pipeline.named_steps:
        raise ValueError("Pipeline must include feature_scaler")
    
    feature_scaler = pipeline.named_steps['feature_scaler']
    
    if 'cq_numeric' not in feature_scaler.feature_min or 'cq_numeric' not in feature_scaler.feature_max:
        raise ValueError("CQ scaling parameters not found in pipeline")
    
    # Get CQ scaling range
    cq_min = feature_scaler.feature_min['cq']
    cq_max = feature_scaler.feature_max['cq']
    
    # Get VMAF scaler to check if we need to scale the target
    vmaf_scaler = None
    if 'vmaf_scaler' in pipeline.named_steps:
        vmaf_scaler = pipeline.named_steps['vmaf_scaler']
    
    # Apply scaling to target VMAF if needed
    scaled_target_vmaf = target_vmaf
    if vmaf_scaler and vmaf_scaler.min_val is not None and vmaf_scaler.max_val is not None:
        # Check if the target is already in [0,1] range (already scaled)
        if target_vmaf > 0 and target_vmaf < 1:
            scaled_target_vmaf = target_vmaf  # Already scaled
            original_target = target_vmaf * (vmaf_scaler.max_val - vmaf_scaler.min_val) + vmaf_scaler.min_val
            if verbose:
                print(f"Target VMAF {target_vmaf} appears to be scaled (original: ~{original_target:.2f})")
        else:
            # Scale from original VMAF range to [0,1]
            scaled_target_vmaf = (target_vmaf - vmaf_scaler.min_val) / (vmaf_scaler.max_val - vmaf_scaler.min_val)
            if verbose:
                print(f"Scaled target VMAF from {target_vmaf} to {scaled_target_vmaf:.4f}")
    
    # Add safety margin to target
    target_with_margin = scaled_target_vmaf + margin
    if verbose:
        print(f"Target VMAF with margin: {target_with_margin:.4f}")
    
    # Binary search variables
    low = min_cq
    high = max_cq
    best_cq = low  # Start with highest quality as fallback
    best_vmaf = None
    all_results = []  # Track all tested points
    
    # Define a function to make and scale predictions
    def predict_vmaf(cq_value):
        # Scale CQ value for the model
        scaled_cq = (cq_value - min_cq) / (max_cq - min_cq)
        
        # Create features with this CQ
        test_features = video_features.copy()
        test_features['cq_numeric'] = scaled_cq
        
        # Convert to array format expected by model
        X = test_features.values.reshape(1, -1)
        
        # Predict VMAF
        predicted = trained_model.predict(X)[0]
        
        # Store result
        all_results.append((cq_value, predicted))
        
        # Unscale prediction if needed to show VMAF in original scale
        unscaled_prediction = predicted
        if vmaf_scaler and vmaf_scaler.min_val is not None and vmaf_scaler.max_val is not None:
            unscaled_prediction = predicted * (vmaf_scaler.max_val - vmaf_scaler.min_val) + vmaf_scaler.min_val
        
        if verbose:
            print(f"CQ {cq_value}: Predicted VMAF = {unscaled_prediction:.2f}")
            
        return predicted, unscaled_prediction
    
    # Binary search for optimal CQ
    iterations = 0
    max_iterations = 20  # Prevent infinite loops
    
    while high - low > 1 and iterations < max_iterations:
        iterations += 1
        mid = (low + high) // 2
        
        # Predict VMAF for this CQ
        scaled_vmaf, unscaled_vmaf = predict_vmaf(mid)
        
        # Check if this meets our target with margin
        if scaled_vmaf >= target_with_margin:
            # Quality is still good enough with this CQ, try more compression
            low = mid
            best_cq = mid  # Update best known good CQ
            best_vmaf = unscaled_vmaf
        else:
            # Quality is too low, try less compression
            high = mid
    
    # Verification step - check that our optimal CQ actually meets the target
    if best_vmaf is None:
        # If we haven't found a valid solution, check the lowest CQ
        scaled_vmaf, unscaled_vmaf = predict_vmaf(low)
        best_cq = low
        best_vmaf = unscaled_vmaf
    
    # Check the next higher CQ value to confirm we found the boundary
    if best_cq < max_cq:
        next_cq = best_cq + 1
        next_scaled_vmaf, next_unscaled_vmaf = predict_vmaf(next_cq)
        
        # If the next CQ still meets our target, something went wrong in our search
        if next_scaled_vmaf >= target_with_margin:
            if verbose:
                print(f"Warning: CQ {next_cq} still meets quality target. Updating best CQ.")
            best_cq = next_cq
            best_vmaf = next_unscaled_vmaf
    
    # Check if our solution meets the target
    # For this comparison, convert everything to original scale
    original_target = target_vmaf
    if vmaf_scaler and vmaf_scaler.min_val is not None and vmaf_scaler.max_val is not None:
        if target_vmaf > 0 and target_vmaf < 1:  # If target was given in scaled form
            original_target = target_vmaf * (vmaf_scaler.max_val - vmaf_scaler.min_val) + vmaf_scaler.min_val
    
    is_reliable = best_vmaf >= original_target
    
    # Sort results by CQ for a clear overview
    all_results.sort(key=lambda x: x[0])
    
    if verbose:
        print("\nAll tested CQ values:")
        for cq, scaled_pred in all_results:
            # Unscale for display
            unscaled_pred = scaled_pred
            if vmaf_scaler and vmaf_scaler.min_val is not None and vmaf_scaler.max_val is not None:
                unscaled_pred = scaled_pred * (vmaf_scaler.max_val - vmaf_scaler.min_val) + vmaf_scaler.min_val
            print(f"CQ {cq}: VMAF = {unscaled_pred:.2f}")
        
        print(f"\nOptimal CQ: {best_cq} (predicted VMAF: {best_vmaf:.2f}, target: {original_target:.2f})")
        
        if not is_reliable:
            print("Warning: Could not find a CQ value that meets the target VMAF.")
    
    return best_cq, best_vmaf, is_reliable



# Example usage
if __name__ == "__main__":
    cwd = os.getcwd()
    path = os.path.join(cwd, 'src')
    input_file = os.path.join(path,'data', 'merged_results.csv')
    output_file = os.path.join(path,'data', 'preprocessed_data.csv')
    pipeline_file = os.path.join(path,'model', 'preprocessing_pipeline.pkl')
    
    # Create and fit the pipeline, save preprocessed data and pipeline
    pipeline, X_train, y_train = create_preprocessing_pipeline(
        input_file, output_file, pipeline_file, target_column='vmaf'
    )
    
    # Example of how this would be used for inference later
    print("\n--- Example: Preprocessing New Data for Inference ---")
    sample_df = pd.read_csv(input_file, nrows=1)
    print("Sample data:")
    print(sample_df)
    
    # Preprocess the sample data
    print("\nApplying preprocessing for inference...")
    X_new, y_new = preprocess_new_data(sample_df, pipeline, return_target=True)
    print("Preprocessed features (X):")
    print(X_new)
    print("\nTarget value (y):")
    print(y_new)