import os
import logging
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import pickle
from sklearn.inspection import permutation_importance
import sys
import tensorflow as tf

# Add the train_tools directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'train_tools'))

# Import the required classes from the original module
from train_tools.preprocessing import (
    ColumnDropper, 
    VMAFScaler, 
    ResolutionTransformer, 
    NumericFeatureTransformer,
    FeatureScaler,
    TargetExtractor
)

# They will store the cached objects
_MODEL = None
_FEATURE_NAMES = None
_VMAF_SCALER = None
_SCALER = None
_PREDICT_VMAF_FN = None

def load_model_if_needed():
    """Load the model if it hasn't been loaded already"""
    global _MODEL
    if _MODEL is None:
        cwd = os.getcwd()
        path = os.path.join(cwd, 'src', 'model')
        _MODEL = tf.keras.models.load_model(os.path.join(path, 'vmaf_prediction_model.keras'))
        print("Model loaded")
    return _MODEL

def load_feature_names_if_needed():
    """Load feature names if they haven't been loaded already"""
    global _FEATURE_NAMES
    if _FEATURE_NAMES is None:
        cwd = os.getcwd()
        path = os.path.join(cwd, 'src', 'model')
        with open(os.path.join(path, 'feature_names.txt'), 'r') as f:
            _FEATURE_NAMES = [line.strip() for line in f]
        print("Feature names loaded")
    return _FEATURE_NAMES

def load_scalers_if_needed():
    """Load scalers if they haven't been loaded already"""
    global _VMAF_SCALER, _SCALER
    if _VMAF_SCALER is None or _SCALER is None:
        cwd = os.getcwd()
        path = os.path.join(cwd, 'src', 'model')
        _VMAF_SCALER, _SCALER = get_scalers_from_pipeline(os.path.join(path, 'preprocessing_pipeline.pkl'))
        print("Scalers loaded")
    return _VMAF_SCALER, _SCALER

def get_prediction_function():
    """Get the prediction function, creating it if needed"""
    global _PREDICT_VMAF_FN
    if _PREDICT_VMAF_FN is None:
        model = load_model_if_needed()
        vmaf_scaler, _ = load_scalers_if_needed()
        _PREDICT_VMAF_FN = create_vmaf_prediction_function(model, vmaf_scaler, _SCALER)
        print("Prediction function created")
    return _PREDICT_VMAF_FN

def create_vmaf_prediction_function(model, vmaf_scaler=None,scaler=None):
    

    """Create a function for predicting VMAF on new data"""
    def predict_vmaf(video_features, cq_value=None):
        """
        Predict VMAF value for new video features
        
        Parameters:
        -----------
        video_features : dict or pandas DataFrame
            Dictionary or DataFrame containing video features
        cq_value : float, optional
            If provided, will override the cq_numeric value in video_features
            
        Returns:
        --------
        dict
            Dictionary containing VMAF prediction in both scaled and original ranges
        """
        # Convert dictionary to DataFrame if needed
        video_features['cq']=cq_value
        
        vmaf_prediction = model.predict(video_features, verbose=0).flatten()[0]
        
        # Ensure prediction is within valid scaled VMAF range [0, 1]
        vmaf_scaled = max(0, min(1, vmaf_prediction))
     
        # Convert to original VMAF range if scaler is provided
        vmaf_original = vmaf_scaled
        if vmaf_scaler and hasattr(vmaf_scaler, 'min_val') and hasattr(vmaf_scaler, 'max_val'):
            vmaf_original = vmaf_scaled * (vmaf_scaler.max_val - vmaf_scaler.min_val) + vmaf_scaler.min_val
            vmaf_original = max(0, min(100, vmaf_original))  # Ensure in valid range
        
        return {
            'vmaf_scaled': float(vmaf_scaled),
            'vmaf_original': float(vmaf_original)
        }
    
    return predict_vmaf


def search_for_cq(predict_vmaf_fn, video_features, target_vmaf, 
                   min_cq=0.0, max_cq=1.0, min_cq_original=10, max_cq_original=63,
                   tolerance=0.01, max_iterations=20, is_scaled=True):
    """
    Find the optimal CQ value to achieve a target VMAF using binary search
    
    Parameters:
    -----------
    predict_vmaf_fn : function
        Function that predicts VMAF from features and CQ
    video_features : dict
        Video features (excluding CQ)
    target_vmaf : float
        Target VMAF value (in scaled [0,1] range if is_scaled=True)
    min_cq : float
        Minimum CQ value to consider (scaled 0-1)
    max_cq : float
        Maximum CQ value to consider (scaled 0-1)
    min_cq_original : int
        Minimum CQ value in original range (typically 10-63 for AV1)
    max_cq_original : int
        Maximum CQ value in original range (typically 10-63 for AV1)
    tolerance : float
        How close VMAF needs to be to target
    max_iterations : int
        Maximum number of binary search iterations
    is_scaled : bool
        Whether target_vmaf is in scaled [0,1] range
        
    Returns:
    --------
    dict
        Contains optimal CQ, predicted VMAF, and search info
    """
    # Binary search variables
    low = min_cq
    high = max_cq
    best_cq = low  # Start with highest quality (lowest CQ)
    best_vmaf = None
    best_vmaf_original = None
    iterations = 0
    all_results = []
    
    # Target key depends on whether we're using scaled values
    target_key = 'vmaf_scaled' if is_scaled else 'vmaf_original'
    


    # Function to convert between scaled and original CQ
    def scaled_to_original_cq(scaled_cq):
        return scaled_cq * (max_cq_original - min_cq_original) + min_cq_original
    


    # Binary search loop
    while high - low > 0.001 and iterations < max_iterations:
        iterations += 1
        mid = (low + high) / 2
        
        # Convert scaled CQ to original CQ for display
        mid_original = scaled_to_original_cq(mid)
        # Override CQ value if provided
        '''
        if 'cq' in feature_names:
            features_df['cq'] = cq_value
        # Predict VMAF for this CQ
        '''
        result = predict_vmaf_fn(video_features, mid)
        predicted_vmaf = result[target_key]
        predicted_vmaf_original = result['vmaf_original']
        predicted_vmaf_scaled = result['vmaf_scaled']
        
        # Store both scaled and original values
        all_results.append((mid, mid_original, predicted_vmaf_scaled, predicted_vmaf_original))
        
        # If predicted VMAF is close enough to target, we're done
        if abs(predicted_vmaf - target_vmaf) <= tolerance:
            best_cq = mid
            best_vmaf = predicted_vmaf
            best_vmaf_original = predicted_vmaf_original
            break
        
        # Adjust search range
        if predicted_vmaf > target_vmaf:
            # Quality too high, increase CQ (reduce quality)
            low = mid
        else:
            # Quality too low, decrease CQ (increase quality)
            high = mid
            best_cq = mid  # Update best known good CQ
            best_vmaf = predicted_vmaf
            best_vmaf_original = predicted_vmaf_original
    
    # Convert best CQ to original range
    best_cq_original = scaled_to_original_cq(best_cq)
    
    # Sort and return results
    all_results.sort(key=lambda x: x[0])
    
    return {
        'optimal_cq': best_cq,
        'optimal_cq_original': best_cq_original,
        'predicted_vmaf': best_vmaf,
        'predicted_vmaf_original': best_vmaf_original,
        'iterations': iterations,
        'all_tested': all_results
    }

def get_scalers_from_pipeline(pipeline_path='src/data/preprocessing_pipeline.pkl'):
    """
    Extract the VMAF scaler from the saved preprocessing pipeline.
    
    Parameters:
    -----------
    pipeline_path : str
        Path to the saved preprocessing pipeline
        
    Returns:
    --------
    VMAFScaler
        The VMAF scaler object from the pipeline
    """
    # Load the pipeline from the pickle file
    with open(pipeline_path, 'rb') as f:
        pipeline = pickle.load(f)
    
    #print(pipeline.named_steps)
    # Check if pipeline has a vmaf_scaler step
    if 'vmaf_scaler' in pipeline.named_steps:
        vmaf_scaler = pipeline.named_steps['vmaf_scaler']
        print(f"Found VMAF scaler with min={vmaf_scaler.min_val} and max={vmaf_scaler.max_val}")
        feature_scaler = pipeline.named_steps['feature_scaler']
        if 'cq' not in feature_scaler.feature_min or 'cq' not in feature_scaler.feature_max:
            raise ValueError("CQ scaling parameters not found in pipeline")
        # Get CQ scaling range
        cq_min = feature_scaler.feature_min['cq']
        cq_max = feature_scaler.feature_max['cq']
        print(f"Found CQ scaler with min={cq_min} and max={cq_max}")
        return vmaf_scaler, pipeline
    else:
        print("No VMAF scaler found in pipeline. Checking for target_extractor...")
        # Some versions of the code might store this in the target_extractor
        if 'target_extractor' in pipeline.named_steps:
            target_extractor = pipeline.named_steps['target_extractor']
            if hasattr(target_extractor, 'vmaf_scaler'):
                vmaf_scaler = pipeline.named_steps['vmaf_scaler']
                print(f"Found VMAF scaler in target_extractor with min={vmaf_scaler.min_val} and max={vmaf_scaler.max_val}")
                return vmaf_scaler,pipeline
            elif hasattr(vmaf_scaler, 'min_val') and hasattr(vmaf_scaler, 'max_val'):
                print(f"Found VMAF scaling parameters in target_extractor: min={vmaf_scaler.min_val}, max={vmaf_scaler.max_val}")
                return vmaf_scaler, pipeline
    
    # If we can't find a proper VMAF scaler, create a simple one with default values
    print("Could not find VMAF scaler in pipeline. Creating a default scaler with typical VMAF range (23.62-100.0)")
    class DefaultVMAFScaler:
        def __init__(self):
            self.min_val = 23.62  # Your observed minimum VMAF
            self.max_val = 100.0  # Maximum VMAF value
    
    return DefaultVMAFScaler()


def find_optimal_cq(video_features,target_vmaf_original=85):
    """Main function to execute the VMAF prediction pipeline"""
    # Load cached resources (will only load from disk if not already loaded)
    model = load_model_if_needed()
    feature_names = load_feature_names_if_needed()
    vmaf_scaler, scaler = load_scalers_if_needed()
    predict_vmaf = get_prediction_function()
    
    # load input video features into a DataFrame
    input_data = pd.DataFrame([video_features])
    

    # Ensure all required features are present
    for feature in feature_names:
        if feature not in input_data.columns.tolist() and feature != 'cq':
            raise ValueError(f"Missing required feature: {feature}")
            
    features_df = input_data.copy()
    

    # Scale features
    
    features_scaled = scaler.transform(features_df) 
    
    #target_extractor = scaler.named_steps['target_extractor']
    #y_train = target_extractor.get_target(features_df)
    #vmaf_value=(features_df['vmaf'] - vmaf_scaler.min_val) / (vmaf_scaler.max_val - vmaf_scaler.min_val)
    features_scaled['cq']=0.5
    features_scaled = features_scaled[feature_names]
    
    # Make prediction
    #prediction= model.predict(features_scaled)
    #print("Predicted VMAF = ",prediction)
    # Convert scaled predictions back to original VMAF range
    
    # Apply inverse scaling formula: original = scaled * (max - min) + min
    #original_vmaf = prediction[0] * (vmaf_scaler.max_val - vmaf_scaler.min_val) + vmaf_scaler.min_val
    # Ensure prediction is within valid VMAF range [0, 100]
    #original_vmaf = max(0, min(100, original_vmaf))
    #print(original_vmaf)    

    # Demonstrate finding optimal CQ for a target VMAF
    print("Finding optimal CQ for target VMAF")
   
    # Convert the given required VMAF to a scaled value for the search function
    target_vmaf_scaled = (target_vmaf_original - vmaf_scaler.min_val) / (vmaf_scaler.max_val - vmaf_scaler.min_val)
    feature_scaler = scaler.named_steps['feature_scaler']
   
    # Get CQ scaling range
    cq_min = feature_scaler.feature_min['cq']
    cq_max = feature_scaler.feature_max['cq']

    # Find optimal CQ for target VMAF
    result = search_for_cq(predict_vmaf, features_scaled, target_vmaf_scaled, 
                                        min_cq=0.0, max_cq=1.0, 
                                        min_cq_original=cq_min, max_cq_original=cq_max)

    # Display the results prioritizing original values
    print(f"Target VMAF (original): {target_vmaf_original}")
    print(f"Target VMAF (scaled): {target_vmaf_scaled:.4f}")
    print(f"Optimal CQ (original): {result['optimal_cq_original']:.1f}")
    print(f"Optimal CQ (scaled): {result['optimal_cq']:.4f}")
    print(f"Predicted VMAF (original): {result['predicted_vmaf_original']:.1f}")
    print(f"Predicted VMAF (scaled): {result['predicted_vmaf']:.4f}")
    print(f"Search iterations: {result['iterations']}")

    print("\nAll tested CQ values:")
    for scaled_cq, original_cq, vmaf_scaled, vmaf_original in result['all_tested']:
        print(f"  CQ {original_cq:.1f} (scaled: {scaled_cq:.4f}): "f"VMAF {vmaf_original:.1f} (scaled: {vmaf_scaled:.4f})")
    
    # Example of how to use the model for VMAF prediction:
    return int(result['optimal_cq_original'])
# Run the main function if the script is executed directly
if __name__ == "__main__":
    video_features = {"metrics_avg_motion": 0.5,
        "metrics_avg_edge_density": 0.5,
        "metrics_avg_texture":0.5,
        "metrics_avg_temporal_information":0.5,
        "metrics_avg_spatial_information":0.5,
        "metrics_avg_color_complexity":0.5,
        "metrics_scene_change_count":0,
        "metrics_avg_motion_variance":0,
        "metrics_avg_saliency":0,
        "metrics_avg_grain_noise":0,
        "metrics_frame_rate":0,
        "metrics_resolution":0}
    video_features['cq']=0.5    
    find_optimal_cq(video_features)
