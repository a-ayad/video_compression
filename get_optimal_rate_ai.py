from vmafcalculation import calculate_vmaf
from encode_video import encode_video
import os
import pandas as pd
import matplotlib.pyplot as plt

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
                                  # NVENC presets range from p1 (fastest) to p7 (highest quality); p4 is a balanced starting point
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

def main(input_file, output_folder, codec, rate):

    file_size_input = os.path.getsize(input_file) / (1024 * 1024)
    print("Input file size: ", round(file_size_input, 2), "MB")
    settings = ENCODER_SETTINGS[codec]
    settings["cq"] = rate
    print("settings: ", settings)
    output_file = os.path.join(output_folder, f"output_{file_number}_{codec.lower()}_cq{rate}.mp4")
    results, encoding_time = encode_video(input_file, output_file,codec,settings)
    vmafscore = calculate_vmaf(input_file, output_file)
    print("VMAF score: ", vmafscore)
    return vmafscore, os.path.getsize(output_file) / (1024 * 1024)

if __name__ == "__main__":
    input_folder = "./videos/input_videos"
    output_folder = "./videos/output_videos"
    file_number = 1
    input_file = os.path.join(input_folder, f"input_{file_number}.y4m")
    codec = "AV1_NVENC"
    
    rates = range(15, 61, 5)
    vmaf_scores = []
    compression_rates = []

    for rate in rates:
        vmafscore, output_file_size = main(input_file, output_folder, codec, rate)
        vmaf_scores.append(vmafscore)
        compression_rates.append(output_file_size)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(rates, vmaf_scores, marker='o')
    plt.title('Rate vs VMAF Score')
    plt.xlabel('Rate')
    plt.ylabel('VMAF Score')

    plt.subplot(1, 2, 2)
    plt.plot(rates, compression_rates, marker='o')
    plt.title('Rate vs Compression Rate')
    plt.xlabel('Rate')
    plt.ylabel('Output File Size (MB)')

    plt.tight_layout()
    plt.show()