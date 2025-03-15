

from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, save_images
import datetime
import time
import os
import ffmpeg
import json
import subprocess
import tempfile


def has_audio(video_path):
    """
    Returns True if the given video file has an audio stream.
    """
    cmd = [
        'ffprobe', '-v', 'error', 
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        video_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return bool(result.stdout.strip())

def merge_videos(scene_videos, output_video):
    """
    Merge the list of re-encoded scene files using FFmpeg's concat filter.
    This function checks whether the scene files have audio or not, and builds the
    concat filter accordingly.
    
    Args:
        scene_files (list): List of file paths to the scene videos.
        output_video (str): Path to the final merged video.
    """
     # Create a temporary file listing the input files
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
        for video in scene_videos:
            # Ensure that the file paths are absolute
            f.write(f"file '{os.path.abspath(video)}'\n")
        concat_file = f.name
    
    # Use the concat demuxer to merge videos without re-encoding
    ffmpeg.input(concat_file, format='concat', safe=0).output(output_video, c='copy').run(overwrite_output=True)
    print(f"Successfully merged videos (without re-encoding) into '{output_video}'")

 