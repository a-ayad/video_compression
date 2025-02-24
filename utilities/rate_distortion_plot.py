
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
        "codec": "libsvtav1",
        "preset": "8",
        # For bitrate testing, remove or comment out constant quality settings:
        # "crf": 30,
        "b:v": "1000k",     # Use this as a placeholder value (will be overridden in tests)
        "keyint": 50,
    },
    "AV1_NVENC": {
        "codec": "av1_nvenc",
        "preset": "p4",
        "rc": "vbr",
        # "cq": 19,         # Comment out if testing explicit bitrate control
        "b:v": "1000k",     # Placeholder bitrate
        "keyint": 50,
    },
    "H266_VVC": {
        "codec": "libvvenc",    # Nvidia hardware encoder for H.264
        "preset": "p4",           # NVENC presets range from p1 (fastest) to p7 (highest quality); p4 is a balanced starting point
        #"crf": 28,              # Rate control mode: 'vbr' indicates variable bitrate encoding
        "b:v": "1000k",                        # Constant quantizer value for NVENC; lower values mean better quality (similar in purpose to CRF)
        "keyint": 50,             # Keyframe interval
    },
    "H264": {
        "codec": "libx264",
        "preset": "medium",
        # "crf": 23,         # Remove CRF for bitrate-controlled testing
        "b:v": "1000k",     # Placeholder bitrate value
        "keyint": 50,
    },
    "H264_NVENC": {
        "codec": "av1_nvenc",  # (Make sure this is the correct codec for H264 NVENC; otherwise, it might be "h264_nvenc")
        "preset": "p4",
        "rc": "vbr",
        # "cq": 19,
        "b:v": "1000k",     # Placeholder bitrate value
        "keyint": 50,
    },
    "HEVC": {
        "codec": "libx265",
        "preset": "medium",
        # "crf": 28,
        "b:v": "1000k",     # Placeholder bitrate value
        "keyint": 50,
    },
    "HEVC_NVENC": {
        "codec": "hevc_nvenc",
        "preset": "p4",
        "rc": "vbr",
        # "cq": 22,
        "b:v": "1000k",     # Placeholder bitrate value
        "keyint": 50,
    },
    "VVC_H266" : {
        "codec": "libvvenc",
        "preset": "p4",
        "b:v": "1000k",
        "keyint": 50,
    },

}


def encode_video(input_file, output_file, settings):
    """
    Encodes a video using specified codec settings.
    """
    try:
        # Prepare the output settings for FFmpeg
        output_args = {
            "vcodec": settings["codec"],
        }
        if settings.get("crf") is not None:
            output_args["crf"] = settings["crf"]
        if settings.get("preset") is not None:
            output_args["preset"] = settings["preset"]
        if settings.get("keyint"):
            output_args["g"] = settings["keyint"]
        if settings.get("row_mt") is not None:
            output_args["row-mt"] = settings["row_mt"]
        if settings.get("b:v") is not None:
            output_args["b:v"] = settings["b:v"]
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
        stderr = result[1].decode("utf-8")

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

def decode_video(input_file):
    """
    Decodes a video and measures decoding time.
    """
    try:
        start_time = time.time()
        ffmpeg.input(input_file).output("null", f="null").run(overwrite_output=True, quiet=True)
        end_time = time.time()
        decoding_time = end_time - start_time
        return round(decoding_time, 2)
    except ffmpeg.Error as e:
        print(f"Error decoding video: {e}")
        return None

def calculate_vmaf(input_file, encoded_file):
    """
    Calculates the VMAF and other quality metrics using ffmpeg-quality-metrics.
    """
    try:
        # Example: downscale frames and sample every 10th frame.
        ffqm = FfmpegQualityMetrics(input_file, encoded_file)
        metrics = ffqm.calculate(["vmaf"])
        # Average the VMAF values over all frames
        avgvmaf = sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"])
        print(f"VMAF: {round(avgvmaf, 2)}")
        return round(avgvmaf, 2)
    except Exception as e:
        print(f"Error calculating quality metrics: {e}")
        return None,

def main():
    """
    Main function to perform video compression and analysis.
    Processes input videos, compresses them using different codecs,
    calculates file sizes, VMAF scores, encoding and decoding times,
    and displays the results in a DataFrame and charts.
    """
    target_bitrates = [ 1000,2000, 4000,8000,16000,32000, 64000]  # in kbps

    input_folder = "videos"  # Folder containing input videos
    output_folder = "videos" # Folder to save output videos
    os.makedirs(output_folder, exist_ok=True)

    results = []

    for file_name in os.listdir(input_folder):
        if file_name.startswith("input_") and file_name.endswith(".y4m"):
            input_file = os.path.join(input_folder, file_name)
            file_size_input = os.path.getsize(input_file) / (1024 * 1024)
            print("Input file size: ", round(file_size_input, 2), "MB")
            # Extract file number from file name (assumes format input_XX.y4m)
            file_number = file_name.split("_")[1].split(".")[0]
            for bitrate in target_bitrates:
                for codec_name, settings in ENCODER_SETTINGS.items():
                    output_file = os.path.join(output_folder, f"output_{file_number}_{codec_name.lower()}_{bitrate}k.mp4")

                    # Update the bitrate setting for the encoder
                    if settings.get("b:v"):
                        settings["b:v"] = f"{bitrate}k"
                    
                    # Encode the video
                    encoding_results,encoding_time_calculated = encode_video(input_file, output_file, settings)
                    if encoding_results==None:
                        continue
                    if os.path.exists(output_file):
                        # Calculate file size (in MB)
                        file_size = os.path.getsize(output_file) / (1024 * 1024)
                        print("Output file size: ", round(file_size, 2), "MB")
                        Compression_factor = round((file_size / file_size_input) * 100, 2)
                        print("Compression factor: ", Compression_factor, "%")
                        # Calculate VMAF score
                        vmaf_score = calculate_vmaf(input_file, output_file)
                        # Extract additional encoding metrics from the encoding log
                        encoding_metrics = {
                            "frame": None,
                            "fps": None,
                            "bitrate": None,
                            "speed": None
                        }
                        #print("Encoding Time Log: ", encoding_results)
                        if encoding_results:
                            frame_match = re.search(r"frame=\s*(\d+)", encoding_results)
                            fps_match = re.search(r"fps=\s*([\d\.]+)", encoding_results)
                            bitrate_match = re.search(r"bitrate=\s*([\d\.]+kbits/s)", encoding_results)
                            speed_match = re.search(r"speed=\s*([\d\.]+x)", encoding_results)
                            if frame_match:
                                encoding_metrics["frame"] = frame_match.group(1)
                            if fps_match:
                                encoding_metrics["fps"] = fps_match.group(1)
                            if bitrate_match:
                                encoding_metrics["bitrate"] = bitrate_match.group(1)
                            if speed_match:
                                encoding_metrics["speed"] = speed_match.group(1)

                        print("Encoding Metrics: ", encoding_metrics)
                        print("calculated encoding time: ", encoding_time_calculated)

                        # Measure decoding time
                        decoding_time = decode_video(output_file)

                       
                        results.append({
                        "Input File": file_name,
                        "Codec": codec_name,
                        "Target Bitrate (kbps)": bitrate,
                        "Input File Size (MB)": round(file_size_input, 2),
                        "Output File Size (MB)": round(file_size, 2),
                        "Compression Rate": Compression_factor,
                        "VMAF": vmaf_score if vmaf_score else "Error",
                        "Encoding Time": encoding_results,
                        "Encoding Time Calculated": encoding_time_calculated,
                        "Decoding Time": decoding_time,
                        "Frame": encoding_metrics["frame"],
                        "FPS": encoding_metrics["fps"],
                        "Bitrate": encoding_metrics["bitrate"],
                        "Speed": encoding_metrics["speed"]
                        })

    print("Results:", results)

    if not results:
        print("No results to process. Please check your input files and encoding process.")
        return

    # Display results in a DataFrame and save to CSV
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("results.csv", index=False)
  
   
if __name__ == "__main__":
    main()