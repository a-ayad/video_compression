
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
        "keyint": 50,             # Keyframe interval
    },
    "AV1_NVENC": {
        "codec": "av1_nvenc",    # Nvidia hardware encoder for H.264
        "preset": "p4",           # NVENC presets range from p1 (fastest) to p7 (highest quality); p4 is a balanced starting point
        "rc": "vbr",              # Rate control mode: 'vbr' indicates variable bitrate encoding
        "cq": 19,                 # Constant quantizer value for NVENC; lower values mean better quality (similar in purpose to CRF)
        "keyint": 50,             # Keyframe interval
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
        "keyint": 50,             # Keyframe interval (in frames); defines how often an I-frame is inserted. Lower values improve seeking but increase file size.
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
        "rc": "vbr",              # Use variable bitrate encoding
        "cq": 22,                 # Constant quantizer for HEVC NVENC; adjust based on quality needs
        "keyint": 50,             # Keyframe interval
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
        '''
        ffqm = FfmpegQualityMetrics(input_file, encoded_file)
        metrics = ffqm.calculate(["vmaf"])
        '''
        # Average the VMAF values over all frames
        avgvmaf = sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"])
        print(f"VMAF: {round(avgvmaf, 2)}")
        return round(avgvmaf, 2)
    except Exception as e:
        print(f"Error calculating quality metrics: {e}")
        return None

def main():
    """
    Main function to perform video compression and analysis.
    Processes input videos, compresses them using different codecs,
    calculates file sizes, VMAF scores, encoding and decoding times,
    and displays the results in a DataFrame and charts.
    """
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

            # Iterate through each encoder setting
            for codec_name, settings in ENCODER_SETTINGS.items():
                output_file = os.path.join(output_folder, f"output_{file_number}_{codec_name.lower()}.mp4")

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
                    print("Encoding Time Log: ", encoding_results)
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

    # Plot the results
    plt.figure(figsize=(10, 6))
    for codec_name in ENCODER_SETTINGS.keys():
        codec_results = df[df["Codec"] == codec_name]
        plt.bar(codec_results["Input File"], codec_results["Output File Size (MB)"],
                label=f"{codec_name} File Size", alpha=0.7)
        plt.plot(codec_results["Input File"], codec_results["VMAF"], marker="o",
                 label=f"{codec_name} VMAF Score")
        plt.plot(codec_results["Input File"], codec_results["Decoding Time"], marker="x",
                 label=f"{codec_name} Decoding Time")

    plt.xlabel("Input File")
    plt.ylabel("Value")
    plt.title("Codec Comparison: File Size, VMAF, Encoding Time, and Decoding Time")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()