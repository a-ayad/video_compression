
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
ENCODER_SETTINGS = {
    "h264_qsv": {
        "codec": "h264_qsv",
        "crf": 23,
        "preset": "faster",
        "keyint": 30,
        "row_mt": 1,
        "tile_columns": 6
    },
}

def encode_video(input_file, output_file, settings):
    """
    Encodes a video using specified codec settings.
    """
    try:
        # Prepare the FFmpeg command with provided settings
        output_args = {
            "vcodec": settings["codec"],
            "crf": settings["crf"],
            "preset": settings["preset"],
        }

        if settings.get("bitrate"):
            output_args["b:v"] = settings["bitrate"]

        if settings.get("keyint"):
            output_args["g"] = settings["keyint"]

        if settings.get("row_mt") is not None:
            output_args["row-mt"] = settings["row_mt"]

        if settings.get("tile_columns") is not None:
            output_args["tile-columns"] = settings["tile_columns"]
        
        start_time = time.time()
        result = ffmpeg.input(input_file).output(output_file, **output_args).run(overwrite_output=True, capture_stdout=True, capture_stderr=True)
        stderr = result[1].decode("utf-8")
        end_time = time.time()
        encoding_time_measured = end_time - start_time
        
        # Extract encoding time from FFmpeg logs
        encoding_time = None
        for line in stderr.splitlines():
            if "time=" in line:
                encoding_time = line
        print(f"Encoded using {settings['codec']}: {output_file}")
        
        return encoding_time, round(encoding_time_measured, 2)
    except ffmpeg.Error as e:
        print(f"Error encoding with {settings['codec']}: {e}")
        return None,None

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
        ffqm = FfmpegQualityMetrics(input_file, encoded_file)
        metrics = ffqm.calculate(["vmaf"])
        # average the ssim_y values over all frames
        avgvmaf= sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"])
        print(f"VMAF: {round(avgvmaf, 2)}")
        return round(avgvmaf, 2)
    
    except Exception as e:
        print(f"Error calculating quality metrics: {e}")
        return None

def main():
    """
    Main function to perform video compression and analysis.
    This function processes input videos from a specified folder, compresses them using different codecs,
    and evaluates the results based on file size, VMAF score, encoding time, and decoding time. The results
    are saved to a CSV file and visualized using bar and line charts.
    Steps:
    1. Create the output folder if it doesn't exist.
    2. Iterate through input videos in the input folder.
    3. For each video, compress it using different codecs and settings.
    4. Calculate and print the input and output file sizes.
    5. Calculate the VMAF score for the compressed video.
    6. Measure the decoding time for the compressed video.
    7. Append the results to a list.
    8. Print the results list for debugging.
    9. If no results are found, print a message and return.
    10. Display the results in a DataFrame and save it to a CSV file.
    11. Plot the results using bar and line charts.
    Note:
    - The function assumes the existence of the following global variables and functions:
      - ENCODER_SETTINGS: A dictionary containing codec names and their settings.
      - encode_video(input_file, output_file, settings): A function to encode the video.
      - calculate_vmaf(input_file, output_file): A function to calculate the VMAF score.
      - decode_video(output_file): A function to measure the decoding time.
    - The function uses the pandas and matplotlib libraries for data handling and visualization.
    Raises:
    - Any exceptions raised by the os, pandas, or matplotlib libraries.
    """
    input_folder = "videos"  # Folder containing input videos
    output_folder = "videos"  # Folder to save output videos
    os.makedirs(output_folder, exist_ok=True)

    results = []

    for file_name in os.listdir(input_folder):
        if file_name.startswith("input_") and file_name.endswith(".y4m"):
            input_file = os.path.join(input_folder, file_name)
            file_size_input = os.path.getsize(input_file) / (1024 * 1024)
            print("Input file size: ", round(file_size_input, 2), "MB")
            # Extract number from input file name
            file_number = file_name.split("_")[1].split(".")[0]

            # Iterate through each encoder and its settings
            for codec_name, settings in ENCODER_SETTINGS.items():
                output_file = os.path.join(output_folder, f"output_{file_number}_{codec_name.lower()}.mp4")

                # Encode the video
                encoding_time,encoding_time_measured = encode_video(input_file, output_file, settings)

                # Check if the output file exists
                if os.path.exists(output_file):
                    # Calculate file size (in MB)
                    file_size = os.path.getsize(output_file) / (1024 * 1024)
                    print("output file size: ", round(file_size, 2), "MB")
                    print("Compression factor ", round((file_size/file_size_input), 2)*100, "%")
                    # Calculate VMAF score
                    vmaf_score = calculate_vmaf(input_file, output_file)
                    # Extract additional encoding metrics from the encoding log
                    encoding_metrics = {
                        "frame": None,
                        "fps": None,
                        "bitrate": None,
                        "speed": None
                    }
                    print("Encoding metrics: ", encoding_time)
                    print("Encoding Time: ", encoding_time_measured)
                    if encoding_time:
                        frame_match = re.search(r"frame=\s*(\d+)", encoding_time)
                        fps_match = re.search(r"fps=\s*([\d\.]+)", encoding_time)
                        bitrate_match = re.search(r"bitrate=\s*([\d\.]+kbits/s)", encoding_time)
                        speed_match = re.search(r"speed=\s*([\d\.]+x)", encoding_time)

                        if frame_match:
                            encoding_metrics["frame"] = frame_match.group(1)
                        if fps_match:
                            encoding_metrics["fps"] = fps_match.group(1)
                        if bitrate_match:
                            encoding_metrics["bitrate"] = bitrate_match.group(1)
                        if speed_match:
                            encoding_metrics["speed"] = speed_match.group(1)

                    print(encoding_metrics)
                   
                    # Append results with additional encoding metrics
                    
                    # Measure decoding time
                    decoding_time = decode_video(output_file)


                    results.append({
                        "Input File": file_name,
                        "Codec": codec_name,
                        "File Size (MB)": round(file_size, 2),
                        "VMAF": vmaf_score if vmaf_score else "Error",
                        "Encoding Time": encoding_time_measured,
                        "Decoding Time": decoding_time,
                        "Frame": encoding_metrics["frame"],
                        "FPS": encoding_metrics["fps"],
                        "Bitrate": encoding_metrics["bitrate"],
                        "Speed": encoding_metrics["speed"]
                    })
    # Debug: Print the results list
    print("Results:", results)

    # Check if results are empty
    if not results:
        print("No results to process. Please check your input files and encoding process.")
        return

    # Display results in a DataFrame
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("results.csv", index=False)
    # Plot the results
    plt.figure(figsize=(10, 6))

    # Bar chart for file sizes
    for codec_name in ENCODER_SETTINGS.keys():
        codec_results = df[df["Codec"] == codec_name]
        plt.bar(codec_results["Input File"], codec_results["File Size (MB)"], label=f"{codec_name} File Size", alpha=0.7)

    # Line chart for VMAF scores
    for codec_name in ENCODER_SETTINGS.keys():
        codec_results = df[df["Codec"] == codec_name]
        plt.plot(codec_results["Input File"], codec_results["VMAF"], marker="o", label=f"{codec_name} VMAF Score")

    # Line chart for decoding time
    for codec_name in ENCODER_SETTINGS.keys():
        codec_results = df[df["Codec"] == codec_name]
        plt.plot(codec_results["Input File"], codec_results["Decoding Time"], marker="x", label=f"{codec_name} Decoding Time")

    plt.xlabel("Input File")
    plt.ylabel("Value")
    plt.title("Codec Comparison: File Size, VMAF, Encoding Time, and Decoding Time")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()