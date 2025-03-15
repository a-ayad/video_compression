import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from get_video_metrics import analyze_video
from encode_video import encode_video
from calculate_vmaf import calculate_vmaf
import re

def extract_bitrate_from_encoding_results(encoding_results):
    """
    Extract the bitrate from the encoding_results string.
    Example input: "frame=   34 fps=0.0 q=86.0 size=    1536KiB time=00:00:00.68 bitrate=18504.8kbits/s speed=1.32x"
    Returns: Bitrate in Mbps
    """
    # Use regex to find the bitrate value
    match = re.search(r"bitrate=\s*(\d+\.?\d*)kbits/s", encoding_results)
    if match:
        # Extract the value and convert to Mbps (divide by 1000)
        bitrate_value = float(match.group(1))
        return bitrate_value / 1000  # Convert kbits/s to Mbps
    
    # If not found in kbits/s format, try other formats
    match = re.search(r"bitrate=\s*(\d+\.?\d*)bits/s", encoding_results)
    if match:
        # Extract the value and convert to Mbps (divide by 1000000)
        bitrate_value = float(match.group(1))
        return bitrate_value / 1000000  # Convert bits/s to Mbps
    
    # If still not found, return 0
    return 0.0

def reprocess_missing_vmaf(input_folder, output_folder, csv_file):
    """
    Check the CSV file for rows with None/NaN VMAF values and reprocess only those videos.
    
    Args:
        input_folder (str): Path to the folder containing the input video files.
        output_folder (str): Path to the folder where output videos are saved.
        csv_file (str): Path to the CSV file with the results.
    """
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Load existing results
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows from {csv_file}")
    
    # Identify rows with None/NaN VMAF values - check for different representations
    missing_vmaf = df[df['vmaf'].isna() | 
                      (df['vmaf'] == '(None,)') | 
                      (df['vmaf'] == '(None, )') |
                      (df['vmaf'].astype(str).str.contains('None'))]
    
    print(f"Found {len(missing_vmaf)} rows with missing VMAF values")
    
    if len(missing_vmaf) == 0:
        print("No missing VMAF values found. Nothing to reprocess.")
        return
    
    # Process each row with missing VMAF
    for idx, row in missing_vmaf.iterrows():
        video_file = row['video_name']
        cq = row['cq']
        
        video_path = os.path.join(input_folder, video_file)
        if not os.path.exists(video_path):
            print(f"Warning: Video file {video_path} not found. Skipping.")
            continue
            
        print(f"Reprocessing {video_file} with CQ {cq}")
        
        # Generate output file name
        output_file = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_cq{cq}.mp4")
        
        try:
            # Re-encode the video
            encoding_results, encoding_time = encode_video(video_path, output_file, codec="AV1_NVENC", rate=cq)
            
            # Calculate VMAF
            vmaf_score = calculate_vmaf(video_path, output_file)
            print(f"Recalculated VMAF score: {vmaf_score}")
            
            # Update other metrics
            output_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
            bitrate = extract_bitrate_from_encoding_results(encoding_results)
            
            # Update the DataFrame
            df.at[idx, 'vmaf'] = vmaf_score
            df.at[idx, 'output_size'] = output_size
            df.at[idx, 'bitrate'] = bitrate
            
            # Save intermediate results
            df.to_csv(csv_file, index=False)
            print(f"Updated results saved to {csv_file}")
            
            # Remove the encoded file to save space
            os.remove(output_file)
            
        except Exception as e:
            print(f"Error reprocessing {video_file} with CQ {cq}: {e}")
    
    # After processing all missing values, create plots for each video
    create_plots_for_all_videos(df, output_folder)
    
    print("All missing VMAF values have been reprocessed.")

def create_plots_for_all_videos(df, output_folder):
    """
    Create plots for all videos in the DataFrame.
    
    Args:
        df (DataFrame): The DataFrame containing the results.
        output_folder (str): Path to the folder where plots should be saved.
    """
    # Get unique video names
    unique_videos = df['video_name'].unique()
    
    for video_file in unique_videos:
        # Filter data for this video
        video_df = df[df['video_name'] == video_file]
        
        # Check if any vmaf values are still None or problematic strings
        if any(pd.isna(video_df['vmaf']) | 
               (video_df['vmaf'].astype(str).str.contains('None'))):
            print(f"Skipping plot for {video_file}: contains missing VMAF values")
            continue
            
        # Make sure vmaf is numeric
        try:
            video_df['vmaf'] = pd.to_numeric(video_df['vmaf'])
        except Exception as e:
            print(f"Error converting VMAF to numeric for {video_file}: {e}")
            continue
        
        # Skip if there's not enough data
        if len(video_df) < 2:
            print(f"Skipping plot for {video_file}: not enough data points")
            continue
            
        # Check if the plot already exists
        plot_filename = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_plots.png")
        if os.path.exists(plot_filename):
            print(f"Plot already exists for {video_file}, skipping")
            continue
        
        try:
            # Sort by CQ to ensure proper plotting
            video_df = video_df.sort_values('cq')
            
            # Create plot
            fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))
            
            ax1.plot(video_df['cq'], video_df['vmaf'], marker='o')
            ax1.set_title(f'VMAF vs CQ for {video_file}')
            ax1.set_xlabel('CQ')
            ax1.set_ylabel('VMAF')
            
            ax2.plot(video_df['bitrate'], video_df['vmaf'], marker='o')
            ax2.set_title(f'VMAF vs Bitrate for {video_file}')
            ax2.set_xlabel('Bitrate (Mbps)')
            ax2.set_ylabel('VMAF')
            
            ax3.plot(video_df['cq'], video_df['output_size'], marker='o')
            ax3.set_title(f'File Size vs CQ for {video_file}')
            ax3.set_xlabel('CQ')
            ax3.set_ylabel('File Size (MB)')
            
            plt.tight_layout()
            plt.savefig(plot_filename, dpi=300)
            print(f"Plots saved for {video_file}")
            plt.close()
            
        except Exception as e:
            print(f"Error creating plot for {video_file}: {e}")

if __name__ == "__main__":
    input_folder = './videos/output_segments'
    output_folder = './videos/output_videos'
    csv_file = './videos/videos_vmaf_results.csv'
    reprocess_missing_vmaf(input_folder, output_folder, csv_file)