import os
import ffmpeg
import json
import subprocess
import tempfile
from scenedetect import detect, AdaptiveDetector
from calculate_vmaf import calculate_vmaf

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

def get_frame_count(video_path):
    """Return the number of frames in a video using ffprobe."""
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
            vcodec='libx264', crf=0, preset='ultrafast',
            acodec='aac'
        )
    else:
        # Process video only.
        out = ffmpeg.output(
            stream.video, output_path,
            vcodec='libx264', crf=0, preset='ultrafast'
        )
    out.overwrite_output().run()
    print(f"Re-encoded scene: {output_path}")

def split_and_reencode_video(video_path, scene_boundaries, output_dir='./temp_scenes'):
    """
    Split and re-encode the original video into scenes using precise timing.
    
    Args:
        video_path (str): Path to the original video.
        scene_boundaries (list): A list of tuples [(start_time, end_time), ...].
        output_dir (str): Directory to store the re-encoded scenes.
    
    Returns:
        list: Paths to the re-encoded scene files.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    scene_files = []
    for idx, (start, end) in enumerate(scene_boundaries):
        scene_file = os.path.join(output_dir, f'scene_{idx:03d}.mp4')
        reencode_scene(video_path, start, end, scene_file)
        scene_files.append(scene_file)
    return scene_files

def merge_videos_reencode(scene_files, output_video):
    """
    Merge the list of re-encoded scene files using FFmpeg's concat filter.
    This function checks whether the scene files have audio or not, and builds the
    concat filter accordingly.
    
    Args:
        scene_files (list): List of file paths to the scene videos.
        output_video (str): Path to the final merged video.
    """
    # Check if the first scene file has audio. Assume all files are consistent.
    audio_present = has_audio(scene_files[0])
    if audio_present:
        inputs = [ffmpeg.input(scene) for scene in scene_files]
        joined = ffmpeg.concat(*inputs, v=1, a=1).node
        v = joined[0]
        a = joined[1]
        ffmpeg.output(
            v, a, output_video,
            vcodec='libx264', crf=0, preset='ultrafast'
        ).overwrite_output().run()
    else:
        # For audio-less videos, create inputs for video only.
        video_inputs = [ffmpeg.input(scene).video for scene in scene_files]
        joined = ffmpeg.concat(*video_inputs, v=1, a=0)
        ffmpeg.output(
            joined, output_video,
            vcodec='libx264', crf=0, preset='ultrafast'
        ).overwrite_output().run()
    print(f"Merged video created: {output_video}")

# --- Main Processing ---

# Path to your input video
video_path = './videos/input_videos/input_1.y4m'  # Adjust the path and container as needed

# Detect scenes.
# For this example, we assume the `detect` function returns a list of tuples (start_time, end_time) in seconds.
scene_boundaries = detect(video_path, AdaptiveDetector())
# If your scene detection returns objects, convert them accordingly, e.g.:
# scene_boundaries = [(scene.start_time, scene.end_time) for scene in detected_scenes]

# Re-encode each scene with frame-accurate trimming
scene_files = split_and_reencode_video(video_path, scene_boundaries)

# Merge the precisely trimmed scene files
output_file_concat = os.path.join('./videos/output_videos/', "output_1_concat_accurate.mp4")
merge_videos_reencode(scene_files, output_file_concat)

# Check frame counts to verify alignment
original_frames = get_frame_count(video_path)
merged_frames = get_frame_count(output_file_concat)
print(f"Original video frame count: {original_frames}")
print(f"Merged video frame count:   {merged_frames}")

# Calculate VMAF
full_vmaf = calculate_vmaf(video_path, output_file_concat)
print(f"VMAF for full video: {full_vmaf}")
