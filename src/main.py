
import os
import re
import ffmpeg
import tempfile
from merge_videos import merge_videos
import json
import subprocess
import sys
import time
import pandas as pd
from calulate_vmaf_adv import calculate_vmaf_advanced,calculate_parallel_vmaf
# Add the correct paths to import from
sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'train_tools'))

from encode_video import encode_video
from calculate_vmaf import calculate_vmaf
from split_video_into_scenes import split_video_into_scenes
from get_video_metrics import analyze_video
from get_suggested_rate_non_ML import lookup_crf
from find_optimal_cq import find_optimal_cq

# Import the required classes from the original module

from train_tools.preprocessing import (
    ColumnDropper, 
    VMAFScaler, 
    ResolutionTransformer, 
    NumericFeatureTransformer,
    FeatureScaler,
    TargetExtractor
)

metrics_list = {
        "metrics_avg_motion",
        "metrics_avg_edge_density",
        "metrics_avg_texture",
        "metrics_avg_temporal_information",
        "metrics_avg_spatial_information",
        "metrics_avg_color_complexity",
        "metrics_scene_change_count",
        "metrics_avg_motion_variance",
        "metrics_avg_saliency",
        "metrics_avg_grain_noise",
        "metrics_frame_rate",
        "metrics_resolution",
    }

def check_scene_sequence_complete(scene_files, max_scene_gap=1):
    """Check if scene sequence has no gaps larger than allowed"""
    scene_numbers = []
    for file_path in scene_files:
        match = re.search(r"scene_(\d+)\.mp4", os.path.basename(file_path))
        if match:
            scene_numbers.append(int(match.group(1)))
    
    if not scene_numbers:
        return False
        
    scene_numbers.sort()
    
    # Check for gaps in scene numbering
    for i in range(len(scene_numbers)-1):
        if scene_numbers[i+1] - scene_numbers[i] > max_scene_gap:
            print(f"Gap detected in scene sequence between {scene_numbers[i]} and {scene_numbers[i+1]}")
            return False
            
    return True

def get_frame_rate(video_path): #TODO : move to own file
    """
    Returns the average frame rate of the video as a float.
    The rate is extracted from ffprobe's output (e.g., "30000/1001").
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-show_entries', 'stream=avg_frame_rate',
        '-of', 'json',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)
    try:
        avg_frame_rate = data['streams'][0]['avg_frame_rate']
        # The frame rate is given as a fraction, e.g., "30000/1001".
        num, den = avg_frame_rate.split('/')
        if float(den) > 0:
            return float(num) / float(den)
    except (KeyError, IndexError, ValueError):
        return None
    
def get_frame_timestamps(video_path, num_frames=50): #TODO : move to own file
    """ 
    Attempts to retrieve the presentation timestamps for the first few frames
    of the given video by trying multiple available fields.
    
    Returns:
        A list of timestamp strings for the first num_frames, or an empty list if none found.
    """
    fields = ['pkt_pts_time', 'best_effort_timestamp_time', 'pts_time']
    for field in fields:
        cmd = [
            'ffprobe', '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', f'frame={field}',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        timestamps = result.stdout.strip().splitlines()
        # Check if we got non-empty, valid timestamps
        if timestamps and all(ts.strip() not in ('N/A', '') for ts in timestamps):
            return timestamps[:num_frames]
    return []

def get_frame_count(video_path): #TODO : move to own file
    """
    Uses FFprobe to get the frame count of a video.
    """
    cmd = [
        'ffprobe', '-v', 'error',
        '-select_streams', 'v:0',
        '-count_frames',
        '-show_entries', 'stream=nb_read_frames',
        '-print_format', 'json',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    data = json.loads(result.stdout)
    try:
        frame_count = int(data['streams'][0]['nb_read_frames'])
    except (KeyError, IndexError):
        frame_count = None
    return frame_count

def sort_scene_files_by_number(file_paths):
    """Sort scene files by their scene number rather than filename lexicographically."""
    def extract_scene_number(file_path):
        import re


        # Extract scene number - handles both input scenes and output scenes
        match = re.search(r"scene_(\d+)\.mp4|output_scene_(\d+)_", file_path)
        if match:
            # Return the first non-None group
            return int(match.group(1) if match.group(1) else match.group(2))
        return 0  # Default if pattern doesn't match
        
    return sorted(file_paths, key=extract_scene_number)

def validate_video_files(scene_videos):
    """Check if each video file can be decoded properly"""
    bad_files = []
    
    for video in scene_videos:
        cmd = ['ffprobe', '-v', 'error', video]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        if result.stderr:
            print(f"Error in file {video}: {result.stderr}")
            bad_files.append(video)
    
    return bad_files

def enhanced_encoding(input_file,output_video_dir='videos/output_videos',
                      temp_directory='videos/temp_scenes',
                      codec='AV1_NVENC',ai_encoding=True, calculate_vmaf_flag=False,optimized_vmaf_flag=False):
    """
    Enhanced video encoding with optional VMAF calculation
    
    Args:
        input_file: Path to input video
        output_video_dir: Directory for output videos
        temp_directory: Directory for temporary scene files
        codec: Video codec to use
        ai_encoding: Whether to use AI-based encoding parameters
        calculate_vmaf_flag: When True, calculate VMAF scores (slower but provides quality metrics)
    """
    # define the output file name
    output_file_full = os.path.join(output_video_dir, f"output_{codec.lower()}.mp4")
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist.")
        return
        
    # Check if the output file already exists
    results_csv = os.path.join(output_video_dir, 'encoding_results.csv') #TODO: delete from final file
    if os.path.exists(output_file_full) and os.path.exists(results_csv):
        print(f"Output file '{output_file_full}' already exists.")
        results_df = pd.read_csv(results_csv)
        if 'vmaf_score' in results_df.columns:
            vmaf_full_video = results_df['vmaf_score'].values[0]
            print(f"VMAF for full video: {vmaf_full_video}")
        else:
            vmaf_full_video = None

        encoding_time_calculated = results_df['encoding_time'].values[0]
        file_size_input_original = results_df['file_size_mb'].values[0]
        print(f"Results loaded from {results_csv}")
    else:
        print(f"Encoding full video using {codec}...")
        # encode the full video
        encoding_results,encoding_time_calculated = encode_video(input_file, output_file_full,codec,rate=30) #check variables
        # calculate the VMAF for the full video
        # calculate VMAF only if flag is True
        vmaf_full_video = None
        if calculate_vmaf_flag:
            if optimized_vmaf_flag:
                # Use optimized VMAF calculation    
                vmaf_full_video = calculate_vmaf_advanced(input_file, output_file_full,
                                             use_downscaling=True, scale_factor=0.5)
            else:
                vmaf_full_video = calculate_vmaf(input_file, output_file_full)

        print(f"VMAF for full video: {vmaf_full_video}")
        
        # Calculate file size
        file_size_input_original = os.path.getsize(output_file_full) / (1024 * 1024)

        # Create results DataFrame (with or without VMAF)
        results_data = {
            'video_name': [os.path.basename(output_file_full)],
            'codec': [codec],
            'encoding_time': [encoding_time_calculated],
            'file_size_mb': [file_size_input_original]
        }
        
        if vmaf_full_video is not None:
            results_data['vmaf_score'] = [vmaf_full_video]

        # Save the DataFrame to a CSV file
    
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(results_csv, index=False)





    # Check if scenes are already split before splitting
    existing_scenes = []
    scene_pattern = re.compile(r"scene_(\d+)\.mp4")
    
    if os.path.exists(temp_directory):
        for file_name in os.listdir(temp_directory):
            match = scene_pattern.match(file_name)
            if match:
                existing_scenes.append(os.path.join(temp_directory, file_name))
    
    if existing_scenes and check_scene_sequence_complete(existing_scenes):
        print(f"Found complete sequence of {len(existing_scenes)} existing scene files")
        scene_files = sort_scene_files_by_number(existing_scenes)
    else:
        print("Scene sequence incomplete or missing, performing scene detection")
        scene_files = split_video_into_scenes(input_file, temp_directory)

    scene_videos    =   []
    scenes_vmaf=[]
    input_files=  []

     # First, scan for existing encoded scene files
    existing_output_scenes = {}
    for file_name in os.listdir(temp_directory):
        if file_name.startswith(f"output_scene_") and file_name.endswith(f"_{codec.lower()}.mp4"):
            match = re.search(r"output_scene_(\d+)_", file_name)
            if match:
                scene_number = int(match.group(1))
                existing_output_scenes[scene_number] = os.path.join(temp_directory, file_name)


    # Process each input scene file
    for file_name in os.listdir(temp_directory):
        if file_name.startswith("scene") and file_name.endswith(".mp4"):
            match = re.search(r"scene_(\d+)\.mp4", file_name)
            if not match:
                print(f"Scene number not found in file name: {file_name}")
                continue
                
            scene_number = int(match.group(1))
            input_scene = os.path.join(temp_directory, file_name)
            output_scene = os.path.join(temp_directory, f"output_scene_{scene_number}_{codec.lower()}.mp4")
            
            # Check if the output scene already exists
            if scene_number in existing_output_scenes:
                print(f"Scene {scene_number} already encoded, using existing file")
                scene_videos.append(existing_output_scenes[scene_number])
                continue
            # If not, analyze and encode it
            print(f"Encoding scene {scene_number}")
            video_metrics = analyze_video(input_scene, max_frames=100, scale_factor=0.50)
                # Convert resolution tuple to string format if it exists
            if "metrics_resolution" in video_metrics and isinstance(video_metrics["metrics_resolution"], tuple):
                width, height = video_metrics["metrics_resolution"]
                video_metrics["metrics_resolution"] = f"({width}, {height})" 

            #print(f"Metrics for scene {file_name}", video_metrics)
            if ai_encoding == True:
                suggested_crf = find_optimal_cq(video_metrics,target_vmaf_original=93)
            else:
                avg_motion, avg_edge_density, avg_texture = video_metrics['metrics_avg_motion'],
                video_metrics['metrics_avg_edge_density'], video_metrics['metrics_avg_texture']
                suggested_crf = lookup_crf(avg_motion, avg_edge_density, avg_texture)
            print(f"Optimal CQ for scene {file_name}", suggested_crf)

            #encode the scene using the calculated optimal rate
            encode_video(input_scene, output_scene,codec,suggested_crf)
            # calculate VMAF for the scene only if flag is True
            '''
            if calculate_vmaf_flag:
                if optimized_vmaf_flag:
                # Use optimized VMAF calculation    
                    scene_vmaf = calculate_vmaf_advanced(input_scene, output_scene,
                                             use_downscaling=True, scale_factor=0.5)
                else:
                    scene_vmaf = calculate_vmaf(input_scene, output_scene)

                print(f"VMAF for scene {file_name}: {scene_vmaf}")
                scenes_vmaf.append(scene_vmaf)'
            '''
            #print(f"VMAF for scene {file_name}: {scene_vmaf}")
            
            scene_videos.append(output_scene)
            input_files.append(input_scene)
            
            
        else:
            print(f"Skipping file: {file_name}")
    



    try:
        print("Scene videos: ", input_files)
        print("Encoded files: ", scene_videos)
        input_files = sort_scene_files_by_number(input_files)
        scene_videos = sort_scene_files_by_number(scene_videos)
        print("Sorted input scenes:", input_files)
        print("Sorted encoded scenes:", scene_videos)
        if calculate_vmaf_flag:
                if optimized_vmaf_flag:
                # Use optimized VMAF calculation    
                    vmaf_results= calculate_parallel_vmaf(input_files, scene_videos,use_downscaling=True,scale_factor=0.5)
                else:
                    vmaf_results= calculate_parallel_vmaf(input_files, scene_videos)
            
        if calculate_vmaf_flag:    
            print("VMAF parallel results: ", vmaf_results)
       
        output_file_concat = os.path.join(output_video_dir, f"output_concat_{codec.lower()}.mp4")
        bad_files = validate_video_files(scene_videos)
        if bad_files:
            print(f"Warning: {len(bad_files)} files may be corrupted")
            # Consider what to do - skip them or use alternative encoding
        merge_videos(scene_videos, output_file_concat)
        #print("scenes vmaf: ", scenes_vmaf)
        #print("scenes duration: ", scenes_duration)
        #full_vmaf_weighted_sum=calculate_weighted_average_vmaf(scenes_duration, scenes_vmaf)
        
        concatenated_video_size = os.path.getsize(output_file_concat) / (1024 * 1024)
        if calculate_vmaf_flag:
            if optimized_vmaf_flag:
            # Use optimized VMAF calculation    
                full_vmaf = calculate_vmaf_advanced(input_file, output_file_concat,
                                             use_downscaling=True, scale_factor=0.5)
            else:
                full_vmaf = calculate_vmaf(input_file, output_file_concat)
            print(f"VMAF for concatenated video: {full_vmaf}")

        if vmaf_full_video is not None:
            print(f"VMAF for full video: {vmaf_full_video}")
        print( f"Full encoded video size: {file_size_input_original} MB")
        print( f"Concatenated encoded video size: {concatenated_video_size} MB")
        # check frame counts on the original and merged video
        original_frames = get_frame_count(input_file)
        merged_frames = get_frame_count(output_file_concat)
        print(f"Original video frame count: {original_frames}")
        print(f"Merged video frame count:   {merged_frames}")
        # check frame rates on the original and merged video
        orig_rate = get_frame_rate(input_file)
        merged_rate = get_frame_rate(output_file_concat)
        print("Original video frame rate: ", orig_rate)
        print("Merged video frame rate:   ", merged_rate)
        #Optional: check timestamps on the original and merged video
        #orig_timestamps = get_frame_timestamps(input_file_full)
        #merged_timestamps = get_frame_timestamps(output_file_concat)
        #print("Original timestamps: ", orig_timestamps)
        #print("Merged timestamps:   ", merged_timestamps)
    except ffmpeg.Error as e:
        print(f"Error merging videos: {e.stderr.decode('utf8')}")
    except ValueError as e:
        print(f"Error: {e}")        
        


           
if __name__ == "__main__":
    input_video_dir= './videos/input_videos'
    input_file = os.path.join(input_video_dir,'combined_videos', f"input_1.y4m")
    output_video_dir= './videos/output_videos'
    temp_directory= './videos/temp_scenes'
    codec = "AV1_NVENC"
    # Import time module to measure execution time

    # Record start time before calling enhanced_encoding
    start_time = time.time()
    enhanced_encoding(input_file,output_video_dir,temp_directory,codec)
    # After function execution, we'll get the elapsed time
    execution_time = time.time() - start_time
    print(f"Enhanced encoding completed in {execution_time:.2f} seconds")
    print(f"({execution_time/60:.2f} minutes)")
    
