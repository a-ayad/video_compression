import os
import pandas as pd
import numpy as np
import time
from tqdm import tqdm
import glob
from analyze_video_fast import analyze_video_fast


def regenerate_dataset_from_csv(existing_csv, videos_base_path, output_csv, sample_interval=3):
    """
    Regenerate a dataset using the names from an existing CSV file but with new optimized metrics.
    
    Args:
        existing_csv: Path to existing CSV file with video information
        videos_base_path: Base path where videos can be found
        output_csv: Path to save the new CSV file
        sample_interval: Sampling interval for frame analysis
        
    Returns:
        DataFrame with the regenerated dataset
    """
   


    # Load existing CSV
    print(f"Loading existing dataset from {existing_csv}")
    existing_df = pd.read_csv(existing_csv)
    
    # Check if video_name column exists
    if 'video_name' not in existing_df.columns:
        raise ValueError("The CSV file must contain a 'video_name' column")
    
    # Get unique video names
    video_names = existing_df['video_name'].unique()
    print(f"Found {len(video_names)} unique video files to process")
    
    # Function to find a video file in the base path
    def find_video_file(base_path, video_name):
        """Search for a video file recursively in the base path"""
        # Try direct match first
        direct_match = os.path.join(base_path, video_name)
        if os.path.exists(direct_match):
            return direct_match
            
        # Try recursive search
        for root, dirs, files in os.walk(base_path):
            if video_name in files:
                return os.path.join(root, video_name)
                
        # Try searching with glob pattern (more flexible)
        pattern = os.path.join(base_path, "**", video_name)
        matching_files = glob.glob(pattern, recursive=True)
        
        if matching_files:
            return matching_files[0]
            
        return None
    
    # Process each video
    results = []
    skipped = []
    
    for video_name in tqdm(video_names):
        # Find the video file
        video_path = find_video_file(videos_base_path, video_name)
        
        if not video_path:
            print(f"Warning: Could not find video {video_name} in {videos_base_path}")
            skipped.append(video_name)
            continue
        
        try:
            # Analyze video with fast method
            metrics = analyze_video_fast(video_path, max_frames=100, sample_interval=sample_interval)
            
            if metrics:
                # Add file information
                metrics['video_name'] = video_name
                metrics['file_path'] = video_path
                
                # Convert resolution tuple to string
                if 'metrics_resolution' in metrics and isinstance(metrics['metrics_resolution'], tuple):
                    resolution = metrics['metrics_resolution']
                    metrics['resolution_width'] = resolution[0]
                    metrics['resolution_height'] = resolution[1]
                    metrics['resolution'] = f"{resolution[0]}x{resolution[1]}"
                
                # Add to results
                results.append(metrics)
        except Exception as e:
            print(f"Error processing {video_name}: {e}")
            skipped.append(video_name)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Rename columns to match original format
    column_mapping = {
        'metrics_avg_motion': 'avg_motion',
        'metrics_avg_edge_density': 'avg_edge_density',
        'metrics_avg_texture': 'avg_texture',
        'metrics_avg_temporal_information': 'avg_temporal_information',
        'metrics_avg_spatial_information': 'avg_spatial_information',
        'metrics_avg_color_complexity': 'avg_color_complexity',
        'metrics_scene_change_count': 'scene_change_count',
        'metrics_avg_motion_variance': 'avg_motion_variance',
        'metrics_avg_grain_noise': 'avg_grain_noise',
        'metrics_frame_rate': 'frame_rate'
    }
    
    # Rename columns
    df = df.rename(columns=column_mapping)
    
    # Select only relevant columns that match original format
    relevant_columns = list(column_mapping.values()) + ['video_name', 'resolution', 'file_path']
    
    # Filter to only include columns that exist
    final_columns = [col for col in relevant_columns if col in df.columns]
    df = df[final_columns]
    
    # Save to CSV
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} entries to {output_csv}")
    print(f"Skipped {len(skipped)} videos that couldn't be found or processed")
    
    if skipped:
        print("First 5 skipped videos:", skipped[:5])
    
    return df


if __name__ == "__main__":
    # Get the current working directory
    cwd = os.getcwd()
    path = os.path.join(cwd, 'src')
    existing_csv = os.path.join(path, 'data', 'video_metrics_results.csv')
    videos_base_path = os.path.join(cwd, 'videos', 'input_videos')
    output_csv = os.path.join(path, 'data', 'regenerated_video_metrics_results.csv')
    
    # Regenerate dataset
    regenerate_dataset_from_csv(existing_csv, videos_base_path, output_csv, sample_interval=3)