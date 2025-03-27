
import os
import re
import ffmpeg
import tempfile
from merge_videos import merge_videos
import json
import subprocess
import sys

# Add the train_tools directory to the path so we can import from it
sys.path.insert(0, os.path.join(os.getcwd(), 'src', 'train_tools'))

from encode_video import encode_video
from calculate_vmaf import calculate_vmaf
from split_video_into_scenes import split_video_into_scenes
from video_metrics import analyze_video

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


def enhanced_encoding(input_file,output_video_dir='videos/output_videos',temp_directory='videos/temp_scenes',codec='AV1_NVENC'):
    # Example usage:
    input_file_full = os.path.join(input_video_dir, f"input_{file_number}.y4m")
    output_file_full = os.path.join(output_video_dir, f"output_{file_number}_{codec.lower()}.mp4")
    encoding_results,encoding_time_calculated = encode_video(input_file_full, output_file_full,codec,rate=30)
    vmaf_full_video= calculate_vmaf(input_file_full, output_file_full)
    print(f"VMAF for full video: {vmaf_full_video}")   
    scene_files= split_video_into_scenes(input_file_full,temp_directory)
    scene_videos    =   []
    scenes_vmaf=[]
    #rates=[32,42,42]
    rates=[32,28,36]
    counter=0
    for file_name in os.listdir(temp_directory):
        if file_name.startswith(f"scene") and file_name.endswith(".mp4"):
            input_scene = os.path.join(temp_directory, file_name)
            
            match = re.search(r"scene_(\d+)\.mp4", file_name) 
            if match:
                scene_number= int(match.group(1))
            else:
                print("Scene number not found in the file name.")
                scene_number= 1
            output_scene = os.path.join(temp_directory, f"output_scene_{scene_number}_{codec.lower()}.mp4")
            file_size_input = os.path.getsize(input_scene) / (1024 * 1024)
            #print("Input scene size: ", round(file_size_input, 2), "MB")
            # Extract file number from file name (assumes format input_XX.y4m)
            avg_motion, avg_edge_density, avg_texture = analyze_video(input_scene, max_frames=150, scale_factor=0.5)
            suggested_crf = lookup_crf(avg_motion, avg_edge_density, avg_texture)
            print(f"Suggested CQ for scene {file_name}", suggested_crf)
            encode_video(input_scene, output_scene,codec,rates[counter])
            scene_vmaf= calculate_vmaf(input_scene, output_scene) 
            print(f"VMAF for scene {file_name}: {scene_vmaf}")
            scenes_vmaf.append(scene_vmaf)
            scene_videos.append(output_scene)
            counter+=1
            
        else:
            print(f"Skipping file: {file_name}")
    try:
        #print("Scene videos: ", scene_videos) 
        output_file_concat = os.path.join(output_video_dir, f"output_{file_number}_concat_{codec.lower()}.mp4")
        merge_videos(scene_videos, output_file_concat)
        print("scenes vmaf: ", scenes_vmaf)
        #print("scenes duration: ", scenes_duration)
        #full_vmaf_weighted_sum=calculate_weighted_average_vmaf(scenes_duration, scenes_vmaf)
        full_vmaf= calculate_vmaf(input_file_full, output_file_concat)
        print(f"VMAF for full video: {vmaf_full_video}")  
        print(f"VMAF for concatenated video: {full_vmaf}")
        #print(f"VMAF for concatenated video (weighted average): {full_vmaf_weighted_sum}")
        # Optionally, check frame counts on the original and merged video
        original_frames = get_frame_count(input_file_full)
        merged_frames = get_frame_count(output_file_concat)
        print(f"Original video frame count: {original_frames}")
        print(f"Merged video frame count:   {merged_frames}")
        orig_rate = get_frame_rate(input_file_full)
        merged_rate = get_frame_rate(output_file_concat)
        print("Original video frame rate: ", orig_rate)
        print("Merged video frame rate:   ", merged_rate)
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
    input_file = os.path.join(input_video_dir, f"input_1.y4m")
    
    output_video_dir= './videos/output_videos'
    temp_directory= './videos/temp_scenes'
    codec = "AV1_NVENC"
    main(input_file,output_video_dir,temp_directory,codec)
