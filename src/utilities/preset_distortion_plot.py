import ffmpeg
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from ffmpeg_quality_metrics import FfmpegQualityMetrics
import time

# Example encoder settings for H264 (modify as needed)
# Here we remove the CRF value to focus on the effect of presets.
ENCODER_SETTINGS = {
     "AV1_NVENC": {
        "codec": "av1_nvenc",
        #"crf": 30,  # Comment out if testing explicit bitrate control
        "preset": "medium",       # This will be updated in the loop
        # "cq": 19,         # Comment out if testing explicit bitrate control
        "keyint": 50,
    },
     "H264_NVENC": {
        "codec": "av1_nvenc",  # (Make sure this is the correct codec for H264 NVENC; otherwise, it might be "h264_nvenc")
        #"crf": 30,
        # "cq": 19,
        "preset": "medium",       # This will be updated in the loop
        "keyint": 50,
    },
    "HEVC_NVENC": {
        "codec": "hevc_nvenc",
        #"crf": 30,
        # "cq": 22,
        "preset": "medium",       # This will be updated in the loop
        "keyint": 50,
    },
}

def encode_video(input_file, output_file, settings):
    """
    Encodes a video using specified codec settings.
    """
    try:
        # Build output arguments from settings
        output_args = {"vcodec": settings["codec"]}
        if settings.get("b:v"):
            output_args["b:v"] = settings["b:v"]
        if settings.get("preset"):
            output_args["preset"] = settings["preset"]
            
        if settings.get("keyint"):
            output_args["g"] = settings["keyint"]
    
        start_time = time.time()
        result = ffmpeg.input(input_file).output(output_file, **output_args).run(
            overwrite_output=True,
            capture_stdout=True,
            capture_stderr=True
        )
        end_time = time.time()
        encoding_time = round(end_time - start_time, 2)
        stderr = result[1].decode("utf-8")
        print(stderr)
        # Optionally extract any encoding log details here
        print(f"Encoded using {settings['codec']} preset {output_args['preset']}: {output_file}")
        return stderr, encoding_time
    except ffmpeg.Error as e:
        print(f"Error encoding with {settings['codec']} preset {settings['preset']}: {e}")
        return None, None

def calculate_vmaf(input_file, encoded_file):
    """
    Calculates the VMAF and other quality metrics using ffmpeg-quality-metrics.
    """
    try:
        ffqm = FfmpegQualityMetrics(input_file, encoded_file)
        metrics = ffqm.calculate(["vmaf"])
        # Average the VMAF values over all frames
        avgvmaf = sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"])
        print(f"VMAF: {round(avgvmaf, 2)}")
        return round(avgvmaf, 2)
    except Exception as e:
        print(f"Error calculating quality metrics for {encoded_file}: {e}")
        return None

def main():
    """
    Main function to perform video encoding using different presets,
    calculate file sizes and VMAF scores, and display the results.
    """
    input_folder = "videos"   # Folder containing input videos
    output_folder = "videos"  # Folder to save output videos
    os.makedirs(output_folder, exist_ok=True)

    results = []

    # Define a list of presets to test.
    presets = [12, 13, 14, 15, 16, 17, 18]

    for file_name in os.listdir(input_folder):
        if file_name.startswith("input_") and file_name.endswith(".y4m"):
            input_file = os.path.join(input_folder, file_name)
            file_size_input = os.path.getsize(input_file) / (1024 * 1024)
            print("Input file size:", round(file_size_input, 2), "MB")
            # Extract file number from file name (assumes format input_XX.y4m)
            file_number = file_name.split("_")[1].split(".")[0]

            # Loop over each preset value for the ever encoder encoder
            for preset in presets:
                for codec_name, settings in ENCODER_SETTINGS.items():
                    settings = ENCODER_SETTINGS[codec_name].copy()
                    settings["preset"] = preset  # Update the preset for this run

                    output_file = os.path.join(output_folder, f"output_{file_number}_{codec_name}_P{preset}.mp4")
                    encoding_results, encoding_time = encode_video(input_file, output_file, settings)
                    if encoding_results is None:
                        continue

                    # Get the output file size in MB
                    output_file_size = os.path.getsize(output_file) / (1024 * 1024)
                    print("Output file size:", round(output_file_size, 2), "MB")

                    # Calculate VMAF score
                    vmaf_score = calculate_vmaf(input_file, output_file)

                    # Append the results for this preset
                    results.append({
                    "Input File": file_name,
                    "Preset": preset,
                    "Input File Size (MB)": round(file_size_input, 2),
                    "Output File Size (MB)": round(output_file_size, 2),
                    "VMAF": vmaf_score,
                    "Encoding Time (s)": encoding_time
                    })

    if not results:
        print("No results to process. Please check your input files and encoding process.")
        return

    # Create a DataFrame from the results
    df = pd.DataFrame(results)
    print(df)
    df.to_csv("preset_comparison_results.csv", index=False)

    # Plotting the comparison: Output File Size and VMAF vs. Preset

    df["VMAF"] = pd.to_numeric(df["VMAF"], errors="coerce")
    df_plot = df.dropna(subset=["VMAF"])
  

    # Optionally, sort the DataFrame by codec and then by bitrate
    df_plot = df_plot.sort_values(by=["Codec"])
    #rate distortion plot

    # Convert your results to a DataFrame (if not done already)
 
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    #ax1.figure(figsize=(10, 6))
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    # Iterate over each unique codec and plot its PSNR vs. target bitrate

    for codec in df_plot["Codec"].unique():
        codec_data = df_plot[df_plot["Codec"] == codec]
        ax1.plot(codec_data["Preset"],
             codec_data["VMAF"],
             marker="o",
             linestyle="-",
             label=codec)
        ax2.plot(codec_data["Preset"],
                codec_data["Encoding Time (s)"],
                marker="x",
                linestyle="-",
                label=codec)
        ax3.plot(codec_data["Preset"],
                codec_data["Input File Size (MB)"],
                marker="x",
                linestyle="-",
                label=codec)
    #plt.xscale("log")  # Set the x-axis to logarithmic scale
    ax1.set_xlabel("Preset")
    ax1.set_ylabel("VMAF Score")
    ax1.set_title("Preset vs. VMAF for Different Codecs")
    ax1.legend(title="Codec")
    ax2.set_xlabel("Preset")
    ax2.set_ylabel("Encoding Time (s)")
    ax2.set_title("Preset vs. Encoding Time for Different Codecs")
    ax2.legend(title="Codec")
    ax3.set_xlabel("Preset")
    ax3.set_ylabel("Encoding Time (s)")
    ax3.set_title("Preset vs. File Size for Different Codecs")
    ax3.legend(title="Codec")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax3.grid(True, linestyle="--", alpha=0.6)
    fig1.savefig("preset_vs_vmaf.png", dpi=300)
    fig2.savefig("preset_vs_encodingtime.png", dpi=300)
    fig3.savefig("preset_vs_filesize.png", dpi=300)
    plt.show()
if __name__ == "__main__":
    main()
