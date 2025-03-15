
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, save_images
import datetime
import time
import os
import ffmpeg
import subprocess


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



def reencode_scene(video_path, start_time, end_time, output_path):
    """
    Re-encode a scene segment from the original video with exact timing.
    
    Args:
        video_path (str): Path to the original video.
        start_time (float): Scene start time in seconds.
        end_time (float): Scene end time in seconds.
        output_path (str): Path to save the re-encoded segment.
    """
    duration = end_time - start_time
    # Create input stream with precise start and duration.
    stream = ffmpeg.input(video_path, ss=start_time, t=duration)
    if has_audio(video_path):
        # Re-encode both video and audio. Using aac for audio encoding.
        out = ffmpeg.output(
            stream.video, stream.audio, output_path,
            vcodec='libx264', crf=0, preset='fast',
            acodec='aac'
        )
    else:
        # Process video only.
        out = ffmpeg.output(
            stream.video, output_path,
            vcodec='libx264', crf=0, preset='fast'
        )
    out.overwrite_output().run()
    print(f"Re-encoded scene: {output_path}")


def split_video_into_scenes(video_path, temp_dir='./videos/temp_scenes'):
    """
    Split and re-encode the original video into scenes using precise timing.
    
    Args:
        video_path (str): Path to the original video.
        scene_boundaries (list): A list of tuples [(start_time, end_time), ...].
        output_dir (str): Directory to store the re-encoded scenes.
    
    Returns:
        list: Paths to the re-encoded scene files.
    """
    scene_boundaries = detect(video_path, AdaptiveDetector())
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    scene_files = []
    for idx, (start, end) in enumerate(scene_boundaries):
        scene_file = os.path.join(temp_dir, f'scene_{idx:03d}.mp4')
        
        reencode_scene(video_path, start, end, scene_file)
        scene_files.append(scene_file)
    return scene_files

    


if __name__ == "__main__":
    
    # Example usage:
    video_path = "./videos/input_videos/input_1.y4m"
    
    split_video_into_scenes(video_path)




