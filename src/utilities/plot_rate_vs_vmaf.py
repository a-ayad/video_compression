from calculate_vmaf import calculate_vmaf
from encode_video import encode_video
import os
import pandas as pd
import matplotlib.pyplot as plt



def main(input_file, output_folder, codec, rate,max_bit_rate,preset):

    file_size_input = os.path.getsize(input_file) / (1024 * 1024)
    print("Input file size: ", round(file_size_input, 2), "MB")

    output_file = os.path.join(output_folder, f"output_{file_number}_2_{codec.lower()}_cq{rate}_P{preset}.mp4")
    results, encoding_time = encode_video(input_file, output_file,codec,rate,max_bit_rate,preset)
    vmafscore = calculate_vmaf(input_file, output_file)
    print("VMAF score: ", vmafscore)
    return vmafscore, os.path.getsize(output_file) / (1024 * 1024)

if __name__ == "__main__":
    input_folder = "./videos/input_videos"
    output_folder = "./videos/temp_scenes"
    file_number = 1
    #input_file = os.path.join(input_folder, f"input_{file_number}.y4m")
    input_file = "./videos/temp_scenes/scene_002.mp4"
    codec = "AV1_NVENC"
    results = []
    rates = range(20, 40, 5)
    rate=30
    max_bit_rates = ['2000k','4000k','8000k','12000k','18000k','24000k']
    presets=['slow','medium','fast']
    for preset in presets:
        vmaf_scores = []
        compression_rates = []
        for max_bit_rate in max_bit_rates:
            vmafscore, output_file_size = main(input_file, output_folder, codec, rate,max_bit_rate, preset)
            if not isinstance(vmafscore, (int, float)):
                continue
            else:
                results.append({
                'rate': rate,
                'max_bit_rate': max_bit_rate,
                'vmaf_score': vmafscore,
                'output_file_size': output_file_size,
                'preset': preset
            })
            
# Create a DataFrame from the results
    df = pd.DataFrame(results)
    print(df)
     # Drop rows with NaN values
    df = df.dropna()
    print("DataFrame after dropping NaN values:")
    print(df)
    # Save the DataFrame to a CSV file
    df.to_csv('maxbitrate_vs_vmaf_and_compression_rate_scene002.csv', index=False)

    # Plot the results
    plt.figure(figsize=(12, 6))
    
    for rate in rates:
        subset = df[df['rate'] == rate]
        plt.subplot(1, 2, 1)
        plt.plot(subset['max_bit_rate'], subset['vmaf_score'], marker='o', label=f'Rate {rate}')
        plt.title('Max bitrate vs VMAF Score')
        plt.xlabel('bitrate')
        plt.ylabel('VMAF Score')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(subset['max_bit_rate'], subset['output_file_size'], marker='o', label=f'Rate {rate}')
        plt.title('Max bitrate vs Compression Rate')
        plt.xlabel('bitrate')
        plt.ylabel('Output File Size (MB)')
        plt.legend()

    plt.tight_layout()
    plt.savefig('maxbitrate_vs_vmaf_and_compression_rate_scene002.png', dpi=300)  # Save the figure
    plt.show()