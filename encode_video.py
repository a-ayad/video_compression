
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




def encode_video(input_file, output_file,codec,settings):
    """
    Encodes a video using specified codec settings.
    """
    
    results = []

            # Iterate through each encoder setting
    try:
        # Prepare the output settings for FFmpeg
        output_args = {
            "vcodec": settings["codec"],
        }
        if settings.get("crf") is not None:
            output_args["crf"] = settings["crf"]
        if settings.get("preset") is not None:
            output_args["preset"] = settings["preset"]
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
        if settings.get("cq") is not None:
            output_args["cq"] = settings["cq"]
        if settings.get("cpu-used") is not None:
            output_args["cpu-used"] = settings["cpu-used"]
        if settings.get("tile_columns") is not None:
            output_args["tile-columns"] = settings["tile_columns"]
    
    # Copy audio by default (or you can customize this too)
        #cmd.extend(["-c:a", "copy", output_file])

        start_time = time.time()
        result = ffmpeg.input(input_file).output(output_file, **output_args).run(
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True
        )
        end_time = time.time()
        encoding_time_calculated = round(end_time - start_time,2)
        print(f"Encoding time: {encoding_time_calculated} seconds")    
        stderr = result[1].decode("utf-8")
        if os.path.exists(output_file):
            # Calculate file size (in MB)
            file_size_input = os.path.getsize(input_file) / (1024 * 1024)
            file_size_output = os.path.getsize(output_file) / (1024 * 1024)
            print("Output file size: ", round(file_size_output, 2), "MB")
            Compression_factor = round((file_size_output / file_size_input) * 100, 2)
            print("Compression factor: ", Compression_factor, "%")

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
        return None, None
                


