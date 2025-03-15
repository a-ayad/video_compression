import os
import subprocess
from collections import defaultdict

def get_video_resolution(filepath):
    """
    Get the resolution of a video file using ffprobe.

    Args:
        filepath (str): Path to the video file.

    Returns:
        tuple: Width and height of the video.
    """
    ffprobe_command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height",
        "-of", "csv=s=x:p=0",
        filepath
    ]
    result = subprocess.run(ffprobe_command, capture_output=True, text=True)
    resolution = result.stdout.strip()
    width, height = map(int, resolution.split('x'))
    return width, height

def get_video_duration(filepath):
    """
    Get the duration of a video file using ffprobe.

    Args:
        filepath (str): Path to the video file.

    Returns:
        float: Duration of the video in seconds.
    """
    ffprobe_command = [
        "ffprobe",
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        filepath
    ]
    result = subprocess.run(ffprobe_command, capture_output=True, text=True)
    duration = float(result.stdout.strip())
    return duration

def split_videos_into_segments(input_folder, output_folder):
    """
    Splits all video files in the input folder into 5-second segments
    and saves them in the output folder.

    Args:
        input_folder (str): Path to the folder containing the input video files.
        output_folder (str): Path to the folder where output segments will be saved.
    """
    resolution_count = defaultdict(int)
    segment_resolution_count = defaultdict(int)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.avi', '.mov', '.mp4', '.mkv', '.wmv', '.flv', '.y4m')): # Add more video extensions if needed
            input_filepath = os.path.join(input_folder, filename)
            base_filename, ext = os.path.splitext(filename)
            output_segment_pattern = os.path.join(output_folder, f"{base_filename}_segment_%03d.mp4")

            # Get the resolution of the input video
            width, height = get_video_resolution(input_filepath)
            resolution_count[(width, height)] += 1

            ffmpeg_command = [
                "ffmpeg",
                "-i", input_filepath,
                "-c:v", "libx264",  # Use libx264 codec
                "-preset", "veryslow",  # Use veryslow preset for better compression
                "-crf", "0",  # Use CRF 0 for lossless quality
                "-map", "0",
                "-segment_time", "5",
                "-force_key_frames", "expr:gte(t,n_forced*5)",  # Force keyframes at 5-second intervals
                "-f", "segment",
                "-reset_timestamps", "1",
                output_segment_pattern
            ]

            try:
                subprocess.run(ffmpeg_command, check=True, capture_output=True)
                print(f"Successfully split '{filename}' into segments in '{output_folder}'")

                # Count the number of output segments and their resolutions
                segment_files = [f for f in os.listdir(output_folder) if f.startswith(base_filename) and f.endswith('.mp4')]
                for segment_file in segment_files:
                    segment_filepath = os.path.join(output_folder, segment_file)
                    segment_duration = get_video_duration(segment_filepath)
                    if segment_duration < 5:
                        os.remove(segment_filepath)
                        print(f"Removed segment '{segment_file}' as it is shorter than 5 seconds")
                    else:
                        segment_width, segment_height = get_video_resolution(segment_filepath)
                        segment_resolution_count[(segment_width, segment_height)] += 1

            except subprocess.CalledProcessError as e:
                print(f"Error processing '{filename}':")
                print(e.stderr.decode()) # Print FFmpeg error message

    print("\nInput video resolution count:")
    for resolution, count in resolution_count.items():
        print(f"Resolution {resolution}: {count} videos")

    print("\nOutput segment resolution count:")
    for resolution, count in segment_resolution_count.items():
        print(f"Resolution {resolution}: {count} segments")

if __name__ == "__main__":
    current_directory = os.getcwd()
    input_data_folder = os.path.join(current_directory, "videos", "input_videos")
    output_segments_folder = os.path.join(current_directory, "videos", "output_segments")
    if not os.path.exists(output_segments_folder):
        os.makedirs(output_segments_folder)
    if os.path.exists(input_data_folder):
        print(f"Data folder exists: {input_data_folder}")
    else:
        print(f"Data folder does not exist: {input_data_folder}")

    split_videos_into_segments(input_data_folder, output_segments_folder)
    print(f"\nVideo splitting process complete. Segments saved in '{output_segments_folder}'")