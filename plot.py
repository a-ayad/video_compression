from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv("results.csv")
print(df)
df["VMAF"] = pd.to_numeric(df["VMAF"], errors="coerce")
df_plot = df.dropna(subset=["VMAF"])
# Ensure the target bitrate is numeric, in case it was saved as a string.
df_plot["Target Bitrate (kbps)"] = pd.to_numeric(df_plot["Target Bitrate (kbps)"], errors='coerce')
# Optionally, sort the DataFrame by codec and then by bitrate
df_plot = df_plot.sort_values(by=["Codec", "Target Bitrate (kbps)"])
#rate distortion plot

# Convert your results to a DataFrame (if not done already)
 
fig1, ax1 = plt.subplots(figsize=(10, 6))
#ax1.figure(figsize=(10, 6))
fig2, ax2 = plt.subplots(figsize=(10, 6))
    # Iterate over each unique codec and plot its PSNR vs. target bitrate

for codec in df_plot["Codec"].unique():
    codec_data = df_plot[df_plot["Codec"] == codec]
    ax1.plot(codec_data["Target Bitrate (kbps)"],
             codec_data["VMAF"],
             marker="o",
             linestyle="-",
             label=codec)
    ax2.plot(codec_data["Target Bitrate (kbps)"],
                codec_data["Encoding Time Calculated"],
                marker="x",
                linestyle="-",
                label=codec)
#plt.xscale("log")  # Set the x-axis to logarithmic scale
ax1.set_xlabel("Target Bitrate (kbps)")
ax1.set_ylabel("VMAF Score")
ax1.set_title("Rate–Distortion Curve: Bitrate vs. VMAF for Different Codecs")
ax1.legend(title="Codec")
ax2.set_xlabel("Target Bitrate (kbps)")
ax2.set_ylabel("Encoding Time (s)")
ax2.set_title("Encoding Time: Bitrate vs. Encoding Time for Different Codecs")
ax2.legend(title="Codec")
ax1.grid(True, linestyle="--", alpha=0.6)
ax2.grid(True, linestyle="--", alpha=0.6)
fig1.savefig("rate_distortion_curve.png", dpi=300)
fig2.savefig("encoding_time_curve.png", dpi=300)
plt.show()