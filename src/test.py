from calulate_vmaf_adv import calculate_vmaf_advanced
import os
from ffmpeg_quality_metrics import FfmpegQualityMetrics
import time

# Define input and output file paths
input_file = "videos/temp_scenes/scene_002.mp4"
encoded_file = "videos/temp_scenes/output_scene_2_av1_nvenc.mp4"

print("Running VMAF calculation benchmarks...")
print("-" * 60)

# Test standard calculation with clip sampling
print("1. Basic usage with clip sampling:")
start_time = time.time()
vmaf = calculate_vmaf_advanced(input_file, encoded_file)
elapsed = time.time() - start_time
print(f"VMAF: {vmaf}")
print(f"Time taken: {elapsed:.2f} seconds")
print("-" * 60)

# Test with downscaling
print("2. With downscaling enabled:")
start_time = time.time()
vmaf = calculate_vmaf_advanced(input_file, encoded_file, 
                             use_downscaling=True, scale_factor=0.5)
elapsed = time.time() - start_time
print(f"VMAF: {vmaf}")
print(f"Time taken: {elapsed:.2f} seconds")
print("-" * 60)

# Test with parallel processing
print("3. With parallel processing:")
start_time = time.time()
vmaf = calculate_vmaf_advanced(input_file, encoded_file, use_parallel=True)
elapsed = time.time() - start_time
print(f"VMAF: {vmaf}")
print(f"Time taken: {elapsed:.2f} seconds")
print("-" * 60)

# Test with all optimizations 
print("4. Maximum speed - all optimizations enabled:")
start_time = time.time()
vmaf = calculate_vmaf_advanced(input_file, encoded_file, 
                             use_sampling=True, num_clips=3, 
                             use_downscaling=True, scale_factor=0.4,
                             use_parallel=True)
elapsed = time.time() - start_time
print(f"VMAF: {vmaf}")
print(f"Time taken: {elapsed:.2f} seconds")
print("-" * 60)

# Test with no optimizations (highest accuracy)
print("5. Highest accuracy - no optimizations:")
start_time = time.time()
vmaf = calculate_vmaf_advanced(input_file, encoded_file, 
                             use_sampling=False)
elapsed = time.time() - start_time
print(f"VMAF: {vmaf}")
print(f"Time taken: {elapsed:.2f} seconds")
print("-" * 60)

# Print summary table
print("\nSummary:")
print("Method                          | VMAF Score | Time (seconds)")
print("-" * 60)

# You'll need to run the tests again to fill in this table
# or adapt the code to store results in variables

print("Done!")