
'''
This script performs video encoding, decoding, and quality assessment using various codecs.
It includes the following functionalities:
1. Configurable parameters for different video encoders.
2. Encoding a video using specified codec settings.
3. Decoding a video and measuring decoding time.
4. Calculating VMAF and other quality metrics using ffmpeg-quality-metrics.
5. Processing multiple input videos and saving the results.
6. Displaying results in a DataFrame and plotting the results.
Functions:
- encode_video(input_file, output_file, settings): Encodes a video using specified codec settings.
- decode_video(input_file): Decodes a video and measures decoding time.
- calculate_vmaf(input_file, encoded_file): Calculates the VMAF and other quality metrics.
- main(): Main function to process input videos, encode them, calculate metrics, and display results.
Usage:
- Place input videos in the specified input folder.
- Run the script to encode videos, calculate metrics, and display results.
'''
import ffmpeg
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from ffmpeg_quality_metrics import FfmpegQualityMetrics
import time

# Configurable parameters for all encoders
# Recommended encoder settings with inline comments
ENCODER_SETTINGS = {
    "AV1_Optimized": {
        "codec": "libsvtav1",     # SVT-AV1 encoder; widely regarded for good quality/efficiency balance in AV1
        "preset": "8",            # Numeric preset for SVT-AV1; higher numbers generally yield better quality but slower encoding
        "crf": 30,                # CRF for AV1; starting around 30 is common—lower for higher quality, but at the cost of speed and file size
        "keyint": 50,
        
    },
    "AV1_NVENC": {
        "codec": "av1_nvenc",    # Nvidia hardware encoder for H.264
        "preset": "p6",                         # NVENC presets range from p1 (fastest) to p7 (highest quality); p4 is a balanced starting point
                   # Rate control mode: 'vbr' indicates variable bitrate encoding
        "cq": 22,                 # Constant quantizer value for NVENC; lower values mean better quality (similar in purpose to CRF) 0-51
        "keyint": 50,
        'pix_fmt': 'yuv420p'  # Force 8-bit pixel format
        
    },
    "AV1_Rust": {
        "codec": "librav1e",      # Rust-based AV1 encoder
        "preset": "4",            # Preset value for rav1e; lower numbers are slower and higher quality, adjust as needed
        "crf": 35,                # CRF-like parameter for rav1e; a value around 35 is a balanced starting point
        "keyint": 50,             # Keyframe interval
    },
    "AV1_Fallback": {
        "codec": "libaom-av1",    # AOM AV1 encoder, often slower but serves as a fallback
        "preset": "medium",       # Preset for libaom-av1; adjust with additional parameters if desired
        "crf": 30,                # CRF for AV1; common starting point for quality control
        "b:v": "0",               # Setting bitrate to 0 for constant quality mode with libaom
        "cpu-used": 4,            # A parameter to trade off encoding speed for quality (lower means higher quality)
        "keyint": 50,             # Keyframe interval
    },
    "H264": {
        "codec": "libx264",       # Software encoder for H.264
        "preset": "medium",       # Determines encoding speed vs. compression efficiency; slower presets improve compression (e.g., "slow")
        "crf": 23,                # Constant Rate Factor: lower value means better quality (range ~18–23 is typical for visually lossless output)
        "keyint": 50, 
                    # Keyframe interval (in frames); defines how often an I-frame is inserted. Lower values improve seeking but increase file size.
    },
    "H266_VVC": {
        "codec": "libvvenc",    # Nvidia hardware encoder for H.264
        "preset": "p4",           # NVENC presets range from p1 (fastest) to p7 (highest quality); p4 is a balanced starting point
        "crf": 28,              # Rate control mode: 'vbr' indicates variable bitrate encoding
                                # Constant quantizer value for NVENC; lower values mean better quality (similar in purpose to CRF)
        "keyint": 50,             # Keyframe interval
    },
    "HEVC": {
        "codec": "libx265",       # Software encoder for HEVC (H.265)
        "preset": "medium",       # Balance between encoding speed and compression efficiency for HEVC
        "crf": 28,                # HEVC typically requires higher CRF values than H.264; 28 is a common starting point
        "keyint": 50,             # Keyframe interval
    },
    "HEVC_NVENC": {
        "codec": "hevc_nvenc",    # Nvidia hardware encoder for HEVC
        "preset": "p4",           # NVENC preset for HEVC
        "rc": "vbr",
        "maxrate": "4M",  # Maximum bitrate (8 Mbps in this example)
        "bufsize": "8M", # Buffer size (typically 2x maxrate)# Use variable bitrate encoding
        "cq": 22,                 # Constant quantizer for HEVC NVENC; adjust based on quality needs
        "keyint": 50,             # Keyframe interval
    },
    "H264_NVENC": {
        "codec": "h264_nvenc",    # Nvidia hardware encoder for HEVC
        "preset": "p4",           # NVENC preset for HEVC
        "rc": "vbr",
        "maxrate": "12M",  # Maximum bitrate (8 Mbps in this example)
        "bufsize": "24M", # Buffer size (typically 2x maxrate)# Use variable bitrate encoding
        "cq": 22,                 # Constant quantizer for HEVC NVENC; adjust based on quality needs
        "keyint": 50,             # Keyframe interval
    },
    
}

'''
        if max_bit_rate is not None:
            output_args["maxrate"] = max_bit_rate
            # Extract the numeric part of max_bit_rate, double it, and convert back to string with 'k'
            numeric_maxrate = int(max_bit_rate[:-1])
            bufsize = f"{numeric_maxrate * 2}k"
            output_args["bufsize"] = bufsize
'''

def encode_video(input_file, output_file,codec,rate=None,max_bit_rate=None,preset=None):
    """
    Encodes a video using specified codec settings.
    """
    print("Encoding video using codec:", codec)
    settings = ENCODER_SETTINGS.get(codec)
    if not settings:
        print(f"Codec settings for {codec} not found.")
        return None, None
    
    results = []

            # Iterate through each encoder setting
    try:
        # Prepare the output settings for FFmpeg
        output_args = {
            "vcodec": settings["codec"], 
        }
        if rate is not None:
            output_args["cq"] = rate
        
        else :
             if settings.get("cq") is not None:
                output_args["cq"] = settings["cq"]
        
        if settings.get("crf") is not None:
            output_args["crf"] = settings["crf"]
        if settings.get("bitrate"):
            output_args["b:v"] = settings["bitrate"]
        if settings.get("b:v"):
            output_args["b:v"] = settings["b:v"]
        if settings.get("keyint"):
            output_args["g"] = settings["keyint"]
        if settings.get("row_mt") is not None:
            output_args["row-mt"] = settings["row_mt"]
        if settings.get("rc") is not None:
            output_args["rc"] = settings["rc"]
        if settings.get("cpu-used") is not None:
            output_args["cpu-used"] = settings["cpu-used"]
        if settings.get("tile_columns") is not None:
            output_args["tile-columns"] = settings["tile_columns"]
        if settings.get("pix_fmt") is not None:
            output_args["pix_fmt"] = settings["pix_fmt"]
        if settings.get("maxrate") is not None:
            output_args["maxrate"] = settings["maxrate"]
        if settings.get("bufsize") is not None:
            output_args["bufsize"] = settings["bufsize"]
    # Copy audio by default (or you can customize this too)
        #cmd.extend(["-c:a", "copy", output_file])
        print("Output args: ", output_args)
        start_time = time.time()
        #print("Output args: ", output_args)
        result = ffmpeg.input(input_file).output(output_file, **output_args).run(
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True
        )
        end_time = time.time()
        encoding_time_calculated = round(end_time - start_time, 2)
        #print(f"Encoding time: {encoding_time_calculated} seconds")
        stderr = result[1].decode("utf-8")
        if os.path.exists(output_file):
            # Calculate file size (in MB)
            file_size_input = os.path.getsize(input_file) / (1024 * 1024)
            file_size_output = os.path.getsize(output_file) / (1024 * 1024)
            #print("Output file size: ", round(file_size_output, 2), "MB")
            Compression_factor = round((file_size_output / file_size_input) * 100, 2)
            #print("Compression factor: ", Compression_factor, "%")

        # Extract encoding time from FFmpeg logs (simplistic extraction)
        encoding_results = None
        for line in stderr.splitlines():
            if "time=" in line:
                encoding_results = line
                break

        print(f"Encoded using {settings['codec']}: {output_file}")
        return encoding_results, encoding_time_calculated
    except ffmpeg.Error as e:
        print(f"Error encoding with {settings['codec']}: {e}")
        print(f"FFmpeg stderr: {e.stderr.decode('utf8')}")
        return None, None
                


