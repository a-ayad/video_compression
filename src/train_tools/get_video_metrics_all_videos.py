import os
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from get_video_metrics import analyze_video

def process_videos(input_folder, output_csv, max_frames=100, scale_factor=0.5):
    """
    Process all video files in the input folder and record the results in a pandas DataFrame.

    Args:
        input_folder (str): Path to the folder containing the input video files.
        output_csv (str): Path to the output CSV file where results will be saved.
        max_frames (int): Maximum number of frames to process for each video.
        scale_factor (float): Scale factor for resizing the video frames.
    """
    video_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.avi', '.mov', '.mp4', '.mkv', '.wmv', '.flv', '.y4m'))]
    results_list = []

    for video_file in video_files:
        video_path = os.path.join(input_folder, video_file)
        print(f"Processing video: {video_file}")
        results = analyze_video(video_path, max_frames=max_frames, scale_factor=scale_factor)
        if results:
            results['video_name'] = video_file
            results_list.append(results)

    # Create a pandas DataFrame from the results list
    df = pd.DataFrame(results_list)

    # Save the DataFrame to a CSV file
    df.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

if __name__ == "__main__":
    # Get the current working directory
    cwd=os.getcwd()
    path=os.path.join(cwd,'src')
    input_folder = os.path.join(path,'videos','output_segments')
    output_csv = os.path.join(path,'data','video_metrics_results.csv')
    process_videos(input_folder, output_csv, max_frames=250, scale_factor=0.5)