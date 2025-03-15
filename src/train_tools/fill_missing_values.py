import pandas as pd
import os
import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib.pyplot as plt

def fill_missing_vmaf(csv_path, output_path=None):
    """
    Fill in missing VMAF values in a video dataset.
    
    Parameters:
    -----------
    csv_path : str
        Path to the input CSV file with video data
    output_path : str, optional
        Path to save the output CSV file. If None, returns the DataFrame
        
    Returns:
    --------
    pandas.DataFrame if output_path is None, otherwise None
    """
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Convert '(None,)' to NaN for easier processing
    df['vmaf'] = df['vmaf'].replace('(None,)', np.nan)
    
    # Check how many missing values we have
    missing_count = df['vmaf'].isna().sum()
    print(f"Found {missing_count} missing VMAF values out of {len(df)} total rows")
    
    # Convert VMAF to float for calculation
    df['vmaf'] = pd.to_numeric(df['vmaf'], errors='coerce')
    
    # Process each video separately
    video_names = df['video_name'].unique()
    print(f"Processing {len(video_names)} unique videos")
    
    # Track our fill statistics
    filled_count = 0
    
    # Process each video
    for video_name in video_names:
        # Get data for this video
        video_df = df[df['video_name'] == video_name].copy()
        
        # Check if there are any missing values for this video
        if not video_df['vmaf'].isna().any():
            continue
            
        # Sort by CQ
        video_df = video_df.sort_values('cq')
        
        # Identify rows with missing VMAF
        missing_mask = video_df['vmaf'].isna()
        missing_indices = video_df.index[missing_mask]
        
        # If all VMAF values are missing, we can't interpolate
        if missing_mask.all():
            # Use global average values if available, otherwise leave as NaN
            continue
            
        # Get valid CQ-VMAF pairs
        valid_df = video_df[~missing_mask][['cq', 'vmaf']]
        
        # If we only have one valid point, we can't do proper interpolation
        if len(valid_df) == 1:
            # For all missing values, use the only valid value we have
            only_vmaf = valid_df['vmaf'].iloc[0]
            df.loc[missing_indices, 'vmaf'] = only_vmaf
            filled_count += len(missing_indices)
            continue
        
        # If we have multiple valid points, we can interpolate/extrapolate
        
        # For each missing value
        for idx in missing_indices:
            current_cq = df.loc[idx, 'cq']
            
            # Find nearest lower and higher CQ values with valid VMAF
            lower_valid = valid_df[valid_df['cq'] < current_cq]
            higher_valid = valid_df[valid_df['cq'] > current_cq]
            
            # Case 1: We can interpolate (valid points both above and below)
            if not lower_valid.empty and not higher_valid.empty:
                # Get closest points
                lower_cq = lower_valid['cq'].max()
                lower_vmaf = lower_valid.loc[lower_valid['cq'].idxmax(), 'vmaf']
                
                higher_cq = higher_valid['cq'].min()
                higher_vmaf = higher_valid.loc[higher_valid['cq'].idxmin(), 'vmaf']
                
                # Linear interpolation
                interp_vmaf = lower_vmaf + (higher_vmaf - lower_vmaf) * \
                             (current_cq - lower_cq) / (higher_cq - lower_cq)
                
                # Set the interpolated VMAF value
                df.loc[idx, 'vmaf'] = interp_vmaf
                filled_count += 1
                
            # Case 2: We can only extrapolate from lower CQ values
            elif not lower_valid.empty:
                # Get two highest CQ values with valid VMAF
                top_lower = lower_valid.nlargest(2, 'cq')
                
                if len(top_lower) >= 2:
                    # Get points for extrapolation
                    x1, y1 = top_lower.iloc[0]['cq'], top_lower.iloc[0]['vmaf']
                    x2, y2 = top_lower.iloc[1]['cq'], top_lower.iloc[1]['vmaf']
                    
                    # Calculate slope
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Extrapolate
                    extrap_vmaf = y1 + slope * (current_cq - x1)
                    
                    # Cap VMAF between 0 and 100
                    extrap_vmaf = max(0, min(100, extrap_vmaf))
                    
                    # Set the extrapolated VMAF value
                    df.loc[idx, 'vmaf'] = extrap_vmaf
                    filled_count += 1
                else:
                    # If we only have one point, use that value
                    df.loc[idx, 'vmaf'] = lower_valid['vmaf'].iloc[0]
                    filled_count += 1
                    
            # Case 3: We can only extrapolate from higher CQ values
            elif not higher_valid.empty:
                # Get two lowest CQ values with valid VMAF
                bottom_higher = higher_valid.nsmallest(2, 'cq')
                
                if len(bottom_higher) >= 2:
                    # Get points for extrapolation
                    x1, y1 = bottom_higher.iloc[0]['cq'], bottom_higher.iloc[0]['vmaf']
                    x2, y2 = bottom_higher.iloc[1]['cq'], bottom_higher.iloc[1]['vmaf']
                    
                    # Calculate slope
                    slope = (y2 - y1) / (x2 - x1)
                    
                    # Extrapolate
                    extrap_vmaf = y1 + slope * (current_cq - x1)
                    
                    # Cap VMAF between 0 and 100
                    extrap_vmaf = max(0, min(100, extrap_vmaf))
                    
                    # Set the extrapolated VMAF value
                    df.loc[idx, 'vmaf'] = extrap_vmaf
                    filled_count += 1
                else:
                    # If we only have one point, use that value
                    df.loc[idx, 'vmaf'] = higher_valid['vmaf'].iloc[0]
                    filled_count += 1
    
    # Special case for lower CQ values (10-22) which are missing for all videos
    # Global average VMAF trends show higher VMAF at lower CQ, 
    # so we'll use a simple model based on the average trend
    
    # Get average VMAF by CQ level
    avg_by_cq = df.groupby('cq')['vmaf'].mean().reset_index()
    
    # Get the lowest CQ with valid VMAF data (likely CQ 24)
    lowest_valid_cq = avg_by_cq.dropna().nsmallest(1, 'cq').iloc[0]['cq']
    lowest_valid_vmaf = avg_by_cq.dropna().nsmallest(1, 'cq').iloc[0]['vmaf']
    
    # For CQ values below the lowest valid CQ
    low_cq_missing = df[df['cq'] < lowest_valid_cq]['vmaf'].isna()
    low_cq_missing_indices = df[df['cq'] < lowest_valid_cq].index[low_cq_missing]
    
    if len(low_cq_missing_indices) > 0:
        # Simple model: For every 2 units of CQ decrease, VMAF increases by ~0.2
        # This is based on analyzing the general trend of VMAF vs CQ
        for idx in low_cq_missing_indices:
            current_cq = df.loc[idx, 'cq']
            extrapolated_vmaf = lowest_valid_vmaf + 0.1 * (lowest_valid_cq - current_cq)
            
            # Cap at 100 since VMAF can't exceed 100
            extrapolated_vmaf = min(100, extrapolated_vmaf)
            
            df.loc[idx, 'vmaf'] = extrapolated_vmaf
            filled_count += 1
    
    # Print statistics
    print(f"Filled {filled_count} out of {missing_count} missing values ({filled_count/missing_count*100:.2f}%)")
    
    # Round VMAF values to 2 decimal places
    df['vmaf'] = df['vmaf'].round(2)
    
    # Save to output file if specified
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Saved filled data to {output_path}")
        return None
    else:
        return df

def visualize_vmaf_filling(original_csv, filled_csv, sample_videos=3):
    """
    Visualize the filling of VMAF values for sample videos
    
    Parameters:
    -----------
    original_csv : str
        Path to the original CSV file
    filled_csv : str
        Path to the filled CSV file
    sample_videos : int
        Number of sample videos to visualize
    """
    # Read original and filled data
    original_df = pd.read_csv(original_csv)
    filled_df = pd.read_csv(filled_csv)
    
    # Convert '(None,)' to NaN in original data
    original_df['vmaf'] = original_df['vmaf'].replace('(None,)', np.nan)
    original_df['vmaf'] = pd.to_numeric(original_df['vmaf'], errors='coerce')
    
    # Get sample videos
    videos = original_df['video_name'].unique()
    sample_videos = min(sample_videos, len(videos))
    sample_video_names = videos[:sample_videos]
    
    # Create plots
    fig, axes = plt.subplots(sample_videos, 1, figsize=(10, 5*sample_videos))
    if sample_videos == 1:
        axes = [axes]
    
    for i, video_name in enumerate(sample_video_names):
        # Get data for this video
        orig_video = original_df[original_df['video_name'] == video_name].sort_values('cq')
        filled_video = filled_df[filled_df['video_name'] == video_name].sort_values('cq')
        
        # Plot
        ax = axes[i]
        ax.plot(orig_video['cq'], orig_video['vmaf'], 'o', label='Original Data')
        ax.plot(filled_video['cq'], filled_video['vmaf'], '-', label='Filled Data')
        
        ax.set_title(f'VMAF vs CQ for {video_name}')
        ax.set_xlabel('CQ')
        ax.set_ylabel('VMAF')
        ax.legend()
        ax.grid(True)
    
    plt.tight_layout()
    plt.savefig('vmaf_filling_visualization.png')
    plt.close()
    print("Created visualization of VMAF filling")

# Example usage:
# Get the current working directory
cwd=os.getcwd()
path=os.path.join(cwd,'src','videos')
# Read the CSV file
fill_missing_vmaf(os.path.join(path,'videos_vmaf_results.csv'),os.path.join(path,'videos_vmaf_filled.csv'))
visualize_vmaf_filling(os.path.join(path,'videos_vmaf_results.csv'), os.path.join(path,'videos_vmaf_filled.csv'))