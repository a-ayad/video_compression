import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import os
import pickle
from sklearn.inspection import permutation_importance
import sys
# Prevent TensorFlow from using GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# Disable TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=WARNING, 3=ERROR
# Suppress additional TensorFlow warnings
tf.get_logger().setLevel('ERROR')



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

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def get_vmaf_scaler_from_pipeline(pipeline_path='src/model/preprocessing_pipeline.pkl'):
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
    
    # Check if pipeline has a vmaf_scaler step
    if 'vmaf_scaler' in pipeline.named_steps:
        vmaf_scaler = pipeline.named_steps['vmaf_scaler']
        print(f"Found VMAF scaler with min={vmaf_scaler.min_val} and max={vmaf_scaler.max_val}")
        return vmaf_scaler
    else:
        print("No VMAF scaler found in pipeline. Checking for target_extractor...")
        # Some versions of the code might store this in the target_extractor
        if 'target_extractor' in pipeline.named_steps:
            target_extractor = pipeline.named_steps['target_extractor']
            if hasattr(target_extractor, 'vmaf_scaler'):
                vmaf_scaler = target_extractor.vmaf_scaler
                print(f"Found VMAF scaler in target_extractor with min={vmaf_scaler.min_val} and max={vmaf_scaler.max_val}")
                return vmaf_scaler
            elif hasattr(target_extractor, 'min_val') and hasattr(target_extractor, 'max_val'):
                print(f"Found VMAF scaling parameters in target_extractor: min={target_extractor.min_val}, max={target_extractor.max_val}")
                return target_extractor
    
    # If we can't find a proper VMAF scaler, create a simple one with default values
    print("Could not find VMAF scaler in pipeline. Creating a default scaler with typical VMAF range (23.62-100.0)")
    class DefaultVMAFScaler:
        def __init__(self):
            self.min_val = 23.62  # Your observed minimum VMAF
            self.max_val = 100.0  # Maximum VMAF value
    
    return DefaultVMAFScaler()

def load_and_prepare_data(file_path='src/data/preprocessed_data.csv'):
    """Load and prepare the data for training a VMAF prediction model"""
    # Load data
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {df.shape[0]} rows with {df.shape[1]} columns")
    
    # Define features and target
    X = df.drop('vmaf', axis=1)  # Now we drop vmaf instead of cq_ordinal
    y = df['vmaf']  # Now our target is vmaf
    
    feature_names = X.columns.tolist()
    print(f"Features: {feature_names}")
    
    # Get information about the target
    print(f"Target: vmaf with range [{y.min():.4f}, {y.max():.4f}], mean: {y.mean():.4f}")
    
    # Split the data - no need for stratify with continuous target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    # Scale the features that aren't already scaled
    # Note: Our preprocessing already scaled the features, so this might be redundant
    # But keeping it for consistency and in case there are any unscaled features
    
    return X_train, X_test, y_train, y_test, feature_names

def build_vmaf_prediction_model(input_dim):
    """Build and compile a neural network model for VMAF prediction"""
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  # Sigmoid activation for [0,1] output
    ])
    
    model.compile(
        optimizer='adam',
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    model.summary()
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    """Train the VMAF prediction model"""
    # Define callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=20,
        restore_best_weights=True
    )
    
    checkpoint_path = 'best_vmaf_model.keras'
    model_checkpoint = ModelCheckpoint(
        checkpoint_path,
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    # Train model
    history = model.fit(
        X_train, y_train,
        epochs=150,
        batch_size=16,
        validation_data=(X_test, y_test),
        callbacks=[early_stopping, model_checkpoint],
        verbose=1
    )
    
    return model, history

def evaluate_vmaf_model(model, X_test, y_test, vmaf_scaler=None):
    """Evaluate the VMAF prediction model performance"""
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Calculate regression metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate additional metrics for VMAF prediction
    # If VMAF is scaled (0-1), adjust thresholds accordingly
    within_01_pct = np.mean(np.abs(y_pred - y_test) <= 0.01)
    within_05_pct = np.mean(np.abs(y_pred - y_test) <= 0.05)
    
    print("\nModel Performance (Scaled VMAF):")
    print(f"MSE: {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE: {mae:.6f}")
    print(f"R²: {r2:.4f}")
    print(f"Predictions within 0.01 of actual VMAF: {within_01_pct:.2%}")
    print(f"Predictions within 0.05 of actual VMAF: {within_05_pct:.2%}")
    
    # If we have the VMAF scaler, show metrics in original scale too
    if vmaf_scaler and hasattr(vmaf_scaler, 'min_val') and hasattr(vmaf_scaler, 'max_val'):
        # Unscale predictions and actual values
        vmaf_range = vmaf_scaler.max_val - vmaf_scaler.min_val
        y_test_orig = y_test * vmaf_range + vmaf_scaler.min_val
        y_pred_orig = y_pred * vmaf_range + vmaf_scaler.min_val
        
        # Calculate metrics in original scale
        mse_orig = mean_squared_error(y_test_orig, y_pred_orig)
        rmse_orig = np.sqrt(mse_orig)
        mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
        r2_orig = r2_score(y_test_orig, y_pred_orig)
        
        # VMAF points in original scale
        within_1_pt = np.mean(np.abs(y_pred_orig - y_test_orig) <= 1.0)
        within_5_pts = np.mean(np.abs(y_pred_orig - y_test_orig) <= 5.0)
        
        print("\nModel Performance (Original VMAF Scale):")
        print(f"MSE: {mse_orig:.4f}")
        print(f"RMSE: {rmse_orig:.4f}")
        print(f"MAE: {mae_orig:.4f}")
        print(f"R²: {r2_orig:.4f}")
        print(f"Predictions within 1 point of actual VMAF: {within_1_pt:.2%}")
        print(f"Predictions within 5 points of actual VMAF: {within_5_pts:.2%}")
    
    return y_pred, mse, rmse, mae, r2

def plot_vmaf_results(history, y_test, y_pred, cq_values=None):
    """Plot training history and VMAF prediction results"""
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    # Loss history
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # MAE history
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('vmaf_training_history.png')
    plt.close()
    
    # Plot predictions vs actual values
    plt.figure(figsize=(15, 5))
    
    # Scatter plot
    plt.subplot(1, 3, 1)
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 1], [0, 1], 'r--')  # Assuming scaled VMAF in [0,1]
    plt.xlabel('Actual VMAF')
    plt.ylabel('Predicted VMAF')
    plt.title('Actual vs Predicted VMAF')
    plt.grid(True)
    
    # Error distribution
    plt.subplot(1, 3, 2)
    errors = y_pred - y_test
    plt.hist(errors, bins=30, alpha=0.7)
    plt.axvline(0, color='r', linestyle='--')
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title('VMAF Prediction Error Distribution')
    plt.grid(True)
    
    # Error by true VMAF
    plt.subplot(1, 3, 3)
    plt.scatter(y_test, errors, alpha=0.5)
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Actual VMAF')
    plt.ylabel('Prediction Error')
    plt.title('Error vs Actual VMAF')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('vmaf_prediction_results.png')
    plt.close()
    
    # If we have CQ values, create a VMAF vs CQ plot
    if cq_values is not None and len(cq_values) == len(y_test):
        plt.figure(figsize=(10, 6))
        
        # Create a colormap based on CQ values
        plt.scatter(cq_values, y_test, alpha=0.5, label='Actual VMAF', color='blue')
        plt.scatter(cq_values, y_pred, alpha=0.5, label='Predicted VMAF', color='red')
        plt.xlabel('CQ Value (Scaled)')
        plt.ylabel('VMAF')
        plt.title('VMAF vs CQ')
        plt.legend()
        plt.grid(True)
        plt.savefig('vmaf_vs_cq.png')
        plt.close()

def create_vmaf_prediction_function(model, feature_names, vmaf_scaler=None):
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
        if isinstance(video_features, dict):
            # Ensure all required features are present
            for feature in feature_names:
                if feature not in video_features and feature != 'cq':
                    raise ValueError(f"Missing required feature: {feature}")
            
            # Convert to DataFrame
            features_df = pd.DataFrame([video_features])
        else:
            features_df = video_features.copy()
        
        # Override CQ value if provided
        if cq_value is not None and 'cq' in feature_names:
            features_df['cq'] = cq_value
        
        # Ensure features are in the correct order
        features_df = features_df[feature_names]
        
        # Scale features
        features_scaled = vmaf_scaler.transform(features_df)
        
        # Make prediction
        vmaf_prediction = model.predict(features_scaled, verbose=0).flatten()[0]
        
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
        print(video_features)
        # Predict VMAF for this CQ
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


class KerasModelWrapper:
    """Wrapper for Keras model to make it compatible with sklearn's permutation_importance"""
    def __init__(self, keras_model):
        self.keras_model = keras_model
        
    def fit(self, X, y):
        # This is just a dummy method to satisfy the sklearn interface
        # The model is already trained
        return self
        
    def predict(self, X):
        return self.keras_model.predict(X, verbose=0).flatten()


def main():
    """Main function to execute the VMAF prediction pipeline"""
    try:
        # Load and prepare data
        X_train, X_test, y_train, y_test, feature_names = load_and_prepare_data()
        
        # Extract CQ values for visualization
        cq_values = None
        X_test_df = pd.DataFrame(X_test)
        if 'cq' in X_test_df.columns:
            cq_values = X_test_df['cq'].values
        
        # Build model
        input_dim = X_train.shape[1]
        model = build_vmaf_prediction_model(input_dim)
        
        # Train model
        model, history = train_model(model, X_train, y_train, X_test, y_test)
        
        cwd = os.getcwd()
        path = os.path.join(cwd, 'src', 'model')
        # Get the VMAF scaler from your saved preprocessing pipeline
        vmaf_scaler = get_vmaf_scaler_from_pipeline(os.path.join(path, 'preprocessing_pipeline.pkl'))

        # Evaluate model
        y_pred, mse, rmse, mae, r2 = evaluate_vmaf_model(model, X_test, y_test, vmaf_scaler)
        
        # Plot results
        try:
            plot_vmaf_results(history, y_test, y_pred, cq_values)
            print("Saved training history and prediction plots")
        except Exception as e:
            print(f"Could not create plots: {e}")
        
        # Analyze feature importance
        try:
            # Create a proper sklearn-compatible wrapper for the Keras model
            model_wrapper = KerasModelWrapper(model)
            
            # Calculate permutation importance
            perm_importance = permutation_importance(
                model_wrapper, X_test, y_test,
                n_repeats=5, random_state=42,
                scoring='neg_mean_squared_error'
            )
            
            # Sort features by importance
            indices = np.argsort(perm_importance.importances_mean)[::-1]
            
            # Print feature importances
            print("\nFeature Importance:")
            for i in indices:
                if i < len(feature_names):  # Make sure we don't go out of bounds
                    feature_name = feature_names[i]
                    print(f"{feature_name}: {perm_importance.importances_mean[i]:.6f} ± {perm_importance.importances_std[i]:.6f}")
        except Exception as e:
            print(f"Could not analyze feature importance: {e}")
        
        # Save model and artifacts
        try:
            model.save(os.path.join(path,'vmaf_prediction_model.keras'))
            print("Saved model to 'vmaf_prediction_model.keras'")
            
            # Save feature names
            with open(os.path.join(path,'feature_names.txt'), 'w') as f:
                f.write('\n'.join(feature_names))
            print("Saved feature names to 'feature_names.txt'")
            
            # Create prediction function
            predict_vmaf = create_vmaf_prediction_function(model, feature_names, vmaf_scaler)
            
            # Demonstrate finding optimal CQ for a target VMAF
            print("\nDemonstration: Finding optimal CQ for target VMAF")

            # Create sample video features (using middle values for demonstration)
            sample_features = {}
            for feature in feature_names:
                if feature != 'cq':  # Skip CQ as we'll vary this
                    sample_features[feature] = 0.5  # Use middle value for demonstration

            # Define target VMAF in original scale (not scaled)
            target_vmaf_original = 85  # Example: target VMAF of 85 out of 100
            # Convert to scaled value for the search function
            target_vmaf_scaled = (target_vmaf_original - vmaf_scaler.min_val) / (vmaf_scaler.max_val - vmaf_scaler.min_val)

            # Find optimal CQ for target VMAF
            result = find_optimal_cq(predict_vmaf, sample_features, target_vmaf_scaled, 
                                    min_cq=0.0, max_cq=1.0, 
                                    min_cq_original=10, max_cq_original=63)

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
                print(f"  CQ {original_cq:.1f} (scaled: {scaled_cq:.4f}): "
                    f"VMAF {vmaf_original:.1f} (scaled: {vmaf_scaled:.4f})")
        except Exception as e:
            print(f"Could not save model artifacts: {e}")
    except Exception as e:
        print(f"Error in main function: {e}")
        raise  # Re-raise the exception to see the full traceback
        '''
        # Example usage information
        print("\nExample of how to use the model for VMAF prediction:")
        print("```python")
        print("import numpy as np")
        print("import tensorflow as tf")
        print("import pickle")
        print("import pandas as pd")
        print("")
        print("# Load model")
        print("model = tf.keras.models.load_model('vmaf_prediction_model.keras')")
        print("")
        print("# Load scaler")
        print("with open('vmaf_model_scaler.pkl', 'rb') as f:")
        print("    scaler = pickle.load(f)")
        print("")
        print("# Load feature names")
        print("with open('vmaf_feature_names.txt', 'r') as f:")
        print("    feature_names = [line.strip() for line in f]")
        print("")
        print("# Define VMAF scaling parameters")
        print("vmaf_min = 23.62  # Minimum VMAF in your training data")
        print("vmaf_max = 100.0  # Maximum VMAF in your training data")
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