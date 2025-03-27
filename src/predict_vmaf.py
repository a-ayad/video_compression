import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import load_model
import os
import pickle
from sklearn.inspection import permutation_importance
import sys
import tensorflow as tf
# disable eager execution because we are using a saved model
tf.compat.v1.disable_eager_execution()

# Add the train_tools directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'train_tools'))

# Import the required classes from the original module
from preprocessing import (
    ColumnDropper, 
    VMAFScaler, 
    ResolutionTransformer, 
    NumericFeatureTransformer,
    FeatureScaler,
    TargetExtractor
)




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
        print(video_features['cq'])
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


def find_optimal_cq(predict_vmaf_fn, video_features, target_vmaf, 
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


def main():
    """Main function to execute the VMAF prediction pipeline"""
    cwd = os.getcwd()
    path = os.path.join(cwd, 'src','model')

    # Load model
    model = tf.keras.models.load_model(os.path.join(path,'vmaf_prediction_model.keras'))
    print("model loaded")  

    # Load feature names"
   
    # Load model input feature names"
    with open(os.path.join(path,'feature_names.txt'), 'r') as f:
        feature_names = [line.strip() for line in f]
    print("feature names loaded")

    # Get the VMAF scaler from your saved preprocessing pipeline
    vmaf_scaler,scaler = get_scalers_from_pipeline(os.path.join(path, 'preprocessing_pipeline.pkl'))
    print("scalers loaded")

    # Create a function to predict VMAF from video features
    predict_vmaf = create_vmaf_prediction_function(model, vmaf_scaler, scaler)
    print("predict_vmaf function created")

    # Load test data
    test_file_path= os.path.join(cwd,'src' ,'data','test.csv')
    test_file=pd.read_csv(test_file_path)
    print("Original Test File \n")
    print(test_file)

    
    
    # Ensure all required features are present
    for feature in feature_names:
        if feature not in test_file.columns.tolist() and feature != 'cq':
            raise ValueError(f"Missing required feature: {feature}")
            
    
    features_df = test_file.copy()
        
        
        
    # Ensure features are in the correct order
    
    # Scale features
    features_scaled = scaler.transform(features_df) 
    
    target_extractor = scaler.named_steps['target_extractor']
    y_train = target_extractor.get_target(features_df)
    vmaf_value=(features_df['vmaf'] - vmaf_scaler.min_val) / (vmaf_scaler.max_val - vmaf_scaler.min_val)
    features_scaled = features_scaled[feature_names]
    
    
    # Make prediction

    prediction= model.predict(features_scaled)
    print(prediction)
    # Convert scaled predictions back to original VMAF range
    
    # Apply inverse scaling formula: original = scaled * (max - min) + min
    original_vmaf = prediction[0] * (vmaf_scaler.max_val - vmaf_scaler.min_val) + vmaf_scaler.min_val
        # Ensure prediction is within valid VMAF range [0, 100]
    original_vmaf = max(0, min(100, original_vmaf))
    print(original_vmaf)    


    # Demonstrate finding optimal CQ for a target VMAF
    print("Finding optimal CQ for target VMAF")
    

    # Define target VMAF in original scale (not scaled)
    target_vmaf_original = 85  # Example: target VMAF of 85 out of 100
    # Convert to scaled value for the search function
    target_vmaf_scaled = (target_vmaf_original - vmaf_scaler.min_val) / (vmaf_scaler.max_val - vmaf_scaler.min_val)

    feature_scaler = scaler.named_steps['feature_scaler']
   
    # Get CQ scaling range
    cq_min = feature_scaler.feature_min['cq']
    cq_max = feature_scaler.feature_max['cq']

    
    




    
    # Find optimal CQ for target VMAF
    result = find_optimal_cq(predict_vmaf, features_scaled, target_vmaf_scaled, 
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
    '''   
        print("\nExample of how to use the model for VMAF prediction:")
        print("```python")
        print("import numpy as np")
        print("import tensorflow as tf")
        print("import pickle")
        print("import pandas as pd")
        print("")
        
        print("# Example video features with a specific CQ value")
        print("video_features = {")
        for feature in feature_names:
            if feature != 'cq_numeric':
                print(f"    '{feature}': 0.5,  # Replace with actual values")
        print("}")
        print("")
        print("# Function to predict VMAF for a given CQ")
        print("def predict_vmaf_for_cq(features, cq_value):")
        print("    # Create a copy of features and set CQ")
        print("    features_copy = features.copy()")
        print("    features_copy['cq_numeric'] = cq_value")
        print("")
        print("    # Create DataFrame with the right feature order")
        print("    input_df = pd.DataFrame([features_copy])[feature_names]")
        print("")
        print("    # Scale input")
        print("    input_scaled = scaler.transform(input_df)")
        print("")
        print("    # Make prediction")
        print("    vmaf_scaled = model.predict(input_scaled, verbose=0)[0][0]")
        print("    vmaf_scaled = max(0, min(1, vmaf_scaled))  # Ensure in [0,1] range")
        print("")
        print("    # Convert to original VMAF scale")
        print("    vmaf_original = vmaf_scaled * (vmaf_max - vmaf_min) + vmaf_min")
        print("")
        print("    return {")
        print("        'vmaf_scaled': float(vmaf_scaled),")
        print("        'vmaf_original': float(vmaf_original)")
        print("    }")
        print("")
        print("# Test different CQ values")
        print("cq_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]")
        print("for cq in cq_values:")
        print("    result = predict_vmaf_for_cq(video_features, cq)")
        print("    print(f\"CQ (scaled): {cq:.2f}, VMAF: {result['vmaf_original']:.2f}\")")
        print("```")'
        '''
# Run the main function if the script is executed directly
if __name__ == "__main__":
    main()