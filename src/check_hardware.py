import tempfile
import os
import subprocess
import time
import ffmpeg


def test_encoder_works(encoder_name, test_duration=1):
    """
    Actually test if an encoder works by encoding a small test clip.
    Returns True if encoding succeeds, False otherwise.
    """
   
    
    # Create temp files for test
    input_file = tempfile.NamedTemporaryFile(suffix='.yuv', delete=False).name
    output_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False).name
    
    try:
        # Generate a test input (solid color video)
        subprocess.run([
            'ffmpeg', '-y', '-f', 'lavfi', '-i', 
            f'color=c=blue:s=1280x720:d={test_duration}', 
            '-pix_fmt', 'yuv420p', input_file
        ], check=False, capture_output=True)
        
        # Try encoding with the specified encoder
        result = subprocess.run([
            'ffmpeg', '-y', '-f', 'rawvideo', '-pix_fmt', 'yuv420p',
            '-s', '1280x720', '-r', '30', '-i', input_file,
            '-c:v', encoder_name, '-preset', 'fast', '-t', str(test_duration),
            output_file
        ], capture_output=True)
        
        # Check if encoding was successful
        return result.returncode == 0 and os.path.exists(output_file) and os.path.getsize(output_file) > 0
    
    except Exception as e:
        print(f"Error testing {encoder_name}: {e}")
        return False
    finally:
        # Clean up temp files
        for file in [input_file, output_file]:
            if os.path.exists(file):
                os.unlink(file)

def get_best_working_codec():
    """Select the best actually working codec"""
    # Hardware encoders in preference order
    hw_encoders = ['av1_nvenc', 'h264_nvenc', 'hevc_nvenc' ]
    
    # Software encoders in preference order
    sw_encoders = ['libx264', 'libx265', 'libsvtav1']
    
    print("Testing hardware encoders...")
    for encoder in hw_encoders:
        print(f"Testing {encoder}...", end=" ")
        if test_encoder_works(encoder):
            print("✅ Working!")
            return encoder
        print("❌ Failed")
    
    print("Testing software encoders...")
    for encoder in sw_encoders:
        print(f"Testing {encoder}...", end=" ")
        if test_encoder_works(encoder):
            print("✅ Working!")
            return encoder
        print("❌ Failed")
    
    # Ultimate fallback
    print("Falling back to libx264")
    return 'libx264'

# Usage in your main code
if __name__ == "__main__":
    # Test and select the best working codec
    codec = get_best_working_codec()
    print(f"Selected codec: {codec}")
    
    input_video_dir = './videos/input_videos'
    input_file = os.path.join(input_video_dir, 'combined_videos', f"input_8.y4m")
    output_video_dir = './videos/output_videos'
    temp_directory = './videos/temp_scenes'
    
    #enhanced_encoding(input_file, output_video_dir, temp_directory, codec)