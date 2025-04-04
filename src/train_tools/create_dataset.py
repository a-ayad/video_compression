def create_complete_training_dataset(input_folder, output_folder, segment_method='hybrid', 
                                     codec="AV1_NVENC", cq_values=None):
    """
    Complete end-to-end pipeline for creating a training dataset:
    1. Segment videos using the specified method
    2. Analyze each segment with fast metrics
    3. Encode segments at different CQ/CRF values
    4. Calculate VMAF for each encoding
    5. Save the complete dataset
    
    Args:
        input_folder: Folder containing original videos
        output_folder: Folder to save all outputs
        segment_method: 'scene', 'fixed', or 'hybrid'
        codec: Codec to use for encoding tests
        cq_values: CQ values to test (if None, use defaults)
        
    Returns:
        DataFrame with the complete training dataset
    """
    import os
    import pandas as pd
    import numpy as np
    import time
    from tqdm import tqdm
    import shutil
    
    # Set up folders
    segments_folder = os.path.join(output_folder, 'segments')
    encodings_folder = os.path.join(output_folder, 'encodings')
    metrics_folder = os.path.join(output_folder, 'metrics')
    
    os.makedirs(segments_folder, exist_ok=True)
    os.makedirs(encodings_folder, exist_ok=True)
    os.makedirs(metrics_folder, exist_ok=True)
    
    # Step 1: Segment videos
    print("\n==== SEGMENTING VIDEOS ====")
    if segment_method == 'scene':
        segments = create_scene_based_dataset(input_folder, segments_folder)
    elif segment_method == 'fixed':
        segments = create_fixed_length_dataset(input_folder, segments_folder)
    else:  # hybrid
        segments = create_hybrid_dataset(input_folder, segments_folder)
    
    # Get segment files from DataFrame or list
    if isinstance(segments, pd.DataFrame):
        segment_files = segments['segment_path'].tolist()
    else:
        segment_files = [s['segment_path'] for s in segments]
    
    # Step 2: Analyze segments with fast metrics
    print("\n==== ANALYZING SEGMENTS ====")
    all_metrics = []
    
    for segment_file in tqdm(segment_files):
        segment_name = os.path.basename(segment_file)
        print(f"Analyzing {segment_name}")
        
        try:
            # Analyze with fast metrics
            metrics = fast_analyze_video_raw(segment_file, max_frames=50, sample_interval=2)
            
            if metrics:
                # Add segment information
                metrics['segment_file'] = segment_name
                metrics['segment_path'] = segment_file
                
                # Convert resolution tuple to string
                if 'metrics_resolution' in metrics and isinstance(metrics['metrics_resolution'], tuple):
                    resolution = metrics['metrics_resolution']
                    metrics['resolution_width'] = resolution[0]
                    metrics['resolution_height'] = resolution[1]
                    metrics['resolution'] = f"{resolution[0]}x{resolution[1]}"
                
                # Add to results
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Error analyzing {segment_name}: {e}")
    
    # Create and save metrics DataFrame
    metrics_df = pd.DataFrame(all_metrics)
    
    # Rename columns for consistency
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
    
    # Rename columns if they exist
    for old_name, new_name in column_mapping.items():
        if old_name in metrics_df.columns:
            metrics_df = metrics_df.rename(columns={old_name: new_name})
    
    # Save metrics
    metrics_csv = os.path.join(metrics_folder, "segment_metrics.csv")
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved metrics for {len(metrics_df)} segments to {metrics_csv}")
    
    # Step 3: Encode segments at different CQ values
    print("\n==== ENCODING SEGMENTS ====")
    
    # Define CQ values to test if not provided
    if cq_values is None:
        cq_values = list(range(20, 51, 5))  # [20, 25, 30, 35, 40, 45, 50]
    
    # Import encode_video function
    try:
        from encode_video import encode_video
    except ImportError:
        print("Warning: encode_video module not found.")
        print("Make sure it's available in your path.")
        return metrics_df
    
    # Import calculate_vmaf function
    try:
        from calculate_vmaf import calculate_vmaf
    except ImportError:
        print("Warning: calculate_vmaf module not found.")
        print("Make sure it's available in your path.")
        return metrics_df
    
    # Prepare results storage
    encoding_results = []
    
    # Process each segment
    for idx, row in tqdm(metrics_df.iterrows(), total=len(metrics_df)):
        segment_file = row['segment_path']
        segment_name = os.path.basename(segment_file)
        segment_base = os.path.splitext(segment_name)[0]
        
        print(f"\nEncoding segment {idx+1}/{len(metrics_df)}: {segment_name}")
        
        segment_data = row.to_dict()
        
        # Encode at different CQ values
        for cq in cq_values:
            output_file = os.path.join(encodings_folder, f"{segment_base}_cq{cq}.mp4")
            
            # Skip if already encoded
            if os.path.exists(output_file) and os.path.getsize(output_file) > 1000:
                print(f"  CQ {cq}: Already encoded, loading stats")
                
                # Try to load existing results
                try:
                    existing_results = pd.read_csv(os.path.join(output_folder, "training_dataset.csv"))
                    matching_rows = existing_results[
                        (existing_results['segment_file'] == segment_name) & 
                        (existing_results['cq'] == cq)
                    ]
                    
                    if not matching_rows.empty:
                        # Get existing VMAF
                        vmaf = matching_rows['vmaf'].values[0]
                        
                        # Add to results
                        result = segment_data.copy()
                        result.update({
                            'cq': cq,
                            'vmaf': vmaf
                        })
                        encoding_results.append(result)
                        continue
                except Exception as e:
                    print(f"  Error loading existing stats: {e}")
            
            # Encode the segment
            print(f"  CQ {cq}: Encoding...")
            try:
                encode_result, encoding_time = encode_video(segment_file, output_file, codec, rate=cq)
                
                # Check if encoding was successful
                if not os.path.exists(output_file) or os.path.getsize(output_file) < 1000:
                    print(f"  CQ {cq}: Encoding failed or produced invalid file")
                    continue
                
                # Calculate VMAF
                print(f"  CQ {cq}: Calculating VMAF...")
                vmaf = calculate_vmaf(segment_file, output_file)
                
                # Add file size information
                file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
                
                # Add to results
                result = segment_data.copy()
                result.update({
                    'cq': cq,
                    'vmaf': vmaf,
                    'encoding_time': encoding_time,
                    'file_size_mb': file_size_mb
                })
                encoding_results.append(result)
                
                # Save intermediate results
                intermediate_df = pd.DataFrame(encoding_results)
                intermediate_df.to_csv(os.path.join(output_folder, "training_dataset.csv"), index=False)
                
            except Exception as e:
                print(f"  Error in encoding pipeline: {e}")
    
    # Create final DataFrame
    training_df = pd.DataFrame(encoding_results)
    
    # Save final dataset
    training_csv = os.path.join(output_folder, "training_dataset.csv")
    training_df.to_csv(training_csv, index=False)
    print(f"\nComplete training dataset created with {len(training_df)} entries")
    print(f"Dataset saved to {training_csv}")
    
    return training_df


# Example usage:
if __name__ == "__main__":
    # To run the entire pipeline:
    # dataset = create_complete_training_dataset(
    #     input_folder="./videos", 
    #     output_folder="./training_data",
    #     segment_method="hybrid",
    #     codec="AV1_NVENC"
    # )
    
    # To run just the segmentation step:
    # segments = create_hybrid_dataset(
    #     input_folder="./videos",
    #     output_folder="./segments",
    #     target_segment_count=300
    # )
    
    # Or to use scene-based segmentation:
    # segments = create_scene_based_dataset(
    #     input_folder="./videos",
    #     output_folder="./segments"
    # )
    
    # Remember to ensure all helper functions are defined or imported
    print("Import this module and call the functions as needed.")