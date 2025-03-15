import os
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
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

def process_videos(input_folder, output_folder, output_csv):
    """
    Process all video files in the input folder, calculate VMAF and output file size for each CQ value,
    and save the results in a pandas DataFrame. Also, plot VMAF vs CQ and file size vs CQ for each video.
    """
    # Check if output CSV already exists and load existing results
    existing_results = []
    if os.path.exists(output_csv):
        existing_df = pd.read_csv(output_csv)
        existing_results = existing_df.to_dict('records')
        print(f"Loaded {len(existing_results)} existing results from {output_csv}")
    
    results_list = existing_results.copy()  # Start with existing results
    
    # Create a set of (video_name, cq) tuples that have already been processed
    processed_combinations = {(item["video_name"], item["cq"]) for item in existing_results}
    
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.avi', '.mov', '.mp4', '.mkv', '.wmv', '.flv', '.y4m'))]
    
    # Check if output folder exists, create if not
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        print(f"Processing video: {video_file}")
        
        video_results = []  # Store results for the current video
        
        # Get existing results for this video
        for result in existing_results:
            if result["video_name"] == video_file:
                video_results.append(result)
                
        for cq in range(10, 60, 2):
            # Skip if this combination has already been processed
            if (video_file, cq) in processed_combinations:
                print(f"Skipping {video_file} with CQ {cq} (already processed)")
                continue
                
            output_file = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_cq{cq}.mp4")
            
            try:
                encoding_results, encoding_time = encode_video(video_path, output_file, codec="AV1_NVENC", rate=cq)
                vmaf_score = calculate_vmaf(video_path, output_file)
                print("VMAF score: ", vmaf_score)
                output_size = os.path.getsize(output_file) / (1024 * 1024)  # Convert to MB
                bitrate = extract_bitrate_from_encoding_results(encoding_results)  # Get bitrate in Mbps
               
                result = {
                    "video_name": video_file,
                    "cq": cq,
                    "vmaf": vmaf_score,
                    "output_size": output_size,
                    "bitrate": bitrate
                }
                results_list.append(result)
                video_results.append(result)
                
                # Save intermediate results after each encoding
                temp_df = pd.DataFrame(results_list)
                temp_df.to_csv(output_csv, index=False)
                print(f"Intermediate results saved to {output_csv}")
                
                # Add to processed combinations
                processed_combinations.add((video_file, cq))
                
                # Remove the encoded file to save space
                os.remove(output_file)
            
            except Exception as e:
                print(f"Error processing {video_file} with CQ {cq}: {e}")
                # Continue with next CQ value

        # Create a DataFrame for the current video
        if video_results:
            # Check if plot already exists
            plot_filename = os.path.join(output_folder, f"{os.path.splitext(video_file)[0]}_plots.png")
            if os.path.exists(plot_filename):
                print(f"Plot already exists for {video_file}, skipping plot creation")
                continue
                
            video_df = pd.DataFrame(video_results)
            
            # Plot VMAF vs CQ, VMAF vs Bitrate, and File Size vs CQ
            try:
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
                print(f"Error creating plots for {video_file}: {e}")

    # Final save of the DataFrame to CSV file
    final_df = pd.DataFrame(results_list)
    final_df.to_csv(output_csv, index=False)
    print(f"Final results saved to {output_csv}")

if __name__ == "__main__":
    input_folder = './videos/output_segments'
    output_folder = './videos/output_videos'
    output_csv = './videos/videos_vmaf_results.csv'
    process_videos(input_folder, output_folder, output_csv)