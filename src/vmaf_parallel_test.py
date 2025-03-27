import os
import time
import concurrent.futures
from calulate_vmaf_adv import calculate_vmaf_advanced

def test_sequential_vmaf(scenes):
    """Process VMAF calculations sequentially for multiple scenes"""
    print("\n== SEQUENTIAL VMAF PROCESSING ==")
    start_time = time.time()
    results = []
    
    for i, (reference, encoded) in enumerate(scenes):
        print(f"Processing scene {i+1}/{len(scenes)}...")
        scene_start = time.time()
        vmaf = calculate_vmaf_advanced(reference, encoded, use_sampling=True)
        scene_time = time.time() - scene_start
        results.append({
            'scene': i+1,
            'reference': os.path.basename(reference),
            'encoded': os.path.basename(encoded),
            'vmaf': vmaf,
            'time': scene_time
        })
        print(f"Scene {i+1} VMAF: {vmaf}, Time: {scene_time:.2f}s")
    
    total_time = time.time() - start_time
    print(f"Total sequential processing time: {total_time:.2f}s")
    return results, total_time

def test_parallel_vmaf(scenes):
    """Process VMAF calculations in parallel for multiple scenes"""
    print("\n== PARALLEL VMAF PROCESSING ==")
    start_time = time.time()
    results = []
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Start all VMAF calculations in parallel
        futures = []
        for i, (reference, encoded) in enumerate(scenes):
            futures.append((i+1, executor.submit(
                calculate_vmaf_advanced, reference, encoded, use_sampling=True
            )))
        
        # Collect results as they complete
        for i, future in futures:
            try:
                vmaf = future.result()
                scene_result = {
                    'scene': i,
                    'reference': os.path.basename(scenes[i-1][0]),
                    'encoded': os.path.basename(scenes[i-1][1]),
                    'vmaf': vmaf,
                }
                results.append(scene_result)
                print(f"Scene {i} VMAF: {vmaf}")
            except Exception as e:
                print(f"Error processing scene {i}: {e}")
    
    total_time = time.time() - start_time
    print(f"Total parallel processing time: {total_time:.2f}s")
    return results, total_time

if __name__ == "__main__":
    # Define your scene files - update paths as needed
    scenes = [
        # (reference_file, encoded_file)
        ("videos/temp_scenes/scene_000.mp4", "videos/temp_scenes/output_scene_0_hevc_nvenc.mp4"),
        ("videos/temp_scenes/scene_001.mp4", "videos/temp_scenes/output_scene_1_hevc_nvenc.mp4"),
        ("videos/temp_scenes/scene_002.mp4", "videos/temp_scenes/output_scene_2_hevc_nvenc.mp4"),
    ]
    
    # Check if files exist
    missing_files = []
    for ref, enc in scenes:
        if not os.path.exists(ref):
            missing_files.append(ref)
        if not os.path.exists(enc):
            missing_files.append(enc)
    
    if missing_files:
        print("Some files don't exist. Please check these paths:")
        for file in missing_files:
            print(f"- {file}")
        exit(1)
    
    # Run sequential test
    seq_results, seq_time = test_sequential_vmaf(scenes)
    
    # Run parallel test
    par_results, par_time = test_parallel_vmaf(scenes)
    
    # Compare and display results
    print("\n== RESULTS COMPARISON ==")
    print(f"Sequential processing: {seq_time:.2f}s")
    print(f"Parallel processing: {par_time:.2f}s")
    print(f"Speed improvement: {(seq_time/par_time):.2f}x faster")
    
    print("\nVMAF Scores:")
    print("Scene | Sequential | Parallel")
    print("-" * 30)
    for i in range(len(scenes)):
        seq_vmaf = next((r['vmaf'] for r in seq_results if r['scene'] == i+1), "N/A")
        par_vmaf = next((r['vmaf'] for r in par_results if r['scene'] == i+1), "N/A")
        print(f"{i+1}     | {seq_vmaf}      | {par_vmaf}")