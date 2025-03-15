import pandas as pd
import os
import numpy as np
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



def merge_csv_files(vmaf_csv, metrics_csv, output_csv):
    """
    Merge two CSV files based on video names, with the first file as the main one.
    
    Args:
        vmaf_csv (str): Path to the VMAF results CSV file.
        metrics_csv (str): Path to the video metrics CSV file.
        output_csv (str): Path to save the merged CSV file.
    """
    print(f"Reading VMAF results from {vmaf_csv}")
    vmaf_df = pd.read_csv(vmaf_csv)
    
    print(f"Reading video metrics from {metrics_csv}")
    metrics_df = pd.read_csv(metrics_csv)
    
    print(f"VMAF results: {len(vmaf_df)} rows")
    print(f"Video metrics: {len(metrics_df)} rows")
    
    # Check column names to identify video name columns
    print("VMAF columns:", vmaf_df.columns.tolist())
    print("Metrics columns:", metrics_df.columns.tolist())
    
    # Find the video name column in each DataFrame
    vmaf_video_col = [col for col in vmaf_df.columns if "video" in col.lower()][0]
    metrics_video_col = [col for col in metrics_df.columns if "video" in col.lower()][0]
    
    print(f"Using '{vmaf_video_col}' from VMAF file and '{metrics_video_col}' from metrics file")
    
    # Create base video name column
    vmaf_df['base_video_name'] = vmaf_df[vmaf_video_col].apply(
        lambda x: x.split('_cq')[0] if isinstance(x, str) and '_cq' in x else (
            x.split('_segment')[0] if isinstance(x, str) and '_segment' in x else x
        )
    )
    
    # Create a mapping of base names to full metrics
    metrics_dict = {}
    for idx, row in metrics_df.iterrows():
        video_name = row[metrics_video_col]
        if isinstance(video_name, str):
            base_name = video_name.split('_segment')[0] if '_segment' in video_name else video_name
            metrics_dict[base_name] = row.to_dict()
    
    # Add metrics columns to VMAF DataFrame
    metrics_columns = [col for col in metrics_df.columns if col != metrics_video_col]
    

    # Fill in metrics for matching videos
    rows_updated = 0
    for idx, row in vmaf_df.iterrows():
        base_name = row['base_video_name']
        if base_name in metrics_dict:
            for col in metrics_columns:
                vmaf_df.at[idx, f'metrics_{col}'] = metrics_dict[base_name][col]
            rows_updated += 1
    
    print(f"Updated {rows_updated} out of {len(vmaf_df)} rows with metrics data")
    
    # Clean up column names in the final DataFrame
    renamed_columns = {}
    for col in vmaf_df.columns:
        if '+AF8-' in col:
            renamed_columns[col] = col.replace('+AF8-', '_')
    
    if renamed_columns:
        vmaf_df.rename(columns=renamed_columns, inplace=True)
        print(f"Renamed {len(renamed_columns)} columns to use underscores")
    

    # Drop temporary column
    vmaf_df.drop('base_video_name', axis=1, inplace=True)
    
    # Save to output CSV
    vmaf_df.to_csv(output_csv, index=False)
    print(f"Merged data saved to {output_csv}")
    
    # Print sample of merged data
    print("\nSample of merged data:")
    print(vmaf_df.head())

if __name__ == "__main__":
    cwd=os.getcwd()
    path=os.path.join(cwd,'src','data')
    vmaf_csv = os.path.join(path,'videos_vmaf_filled.csv')
    metrics_csv = os.path.join(path,'video_metrics_results.csv')
    output_csv = os.path.join(path,'merged_results.csv')
    
    # Make sure input files exist
    if not os.path.exists(vmaf_csv):
        print(f"Error: VMAF file {vmaf_csv} does not exist.")
    elif not os.path.exists(metrics_csv):
        print(f"Error: Metrics file {metrics_csv} does not exist.")
    else:
        merge_csv_files(vmaf_csv, metrics_csv, output_csv)