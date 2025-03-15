
import os 
import pandas as pd
import pickle
import sys

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



def test():
    print("Test function in test.py")

    return None



def get_vmaf_scaler_from_pipeline(pipeline_path='data/preprocessing_pipeline.pkl'):
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

def preprocess_data(input_file):

    file=pd.read_csv(input_file)
   
    # For a specific column, you could use:
    column_name = "vmaf"
    print(f"Mean of {column_name}: {file[column_name].mean()}")
    print(f"Min of {column_name}: {file[column_name].min()}")
    print(f"Max of {column_name}: {file[column_name].max()}")
    print(f"Variance of {column_name}: {file[column_name].var()}")

    # Save the preprocessed data


  
    




if __name__ == "__main__":
    cwd=os.getcwd()
    print(cwd)
    path=os.path.join(cwd,'src','data')
    input_file=os.path.join(path,'merged_results.csv')
    output_file=os.path.join(path,'preprocessed_data.csv')
    preprocess_data(output_file)
    vmaf_Scaler=get_vmaf_scaler_from_pipeline(os.path.join(path,'preprocessing_pipeline.pkl'))
