import time
import concurrent.futures
import tempfile
import subprocess
import json
import os
from ffmpeg_quality_metrics import FfmpegQualityMetrics
    

def calculate_vmaf_advanced(input_file, encoded_file, 
                           use_sampling=True, num_clips=3, clip_duration=2,
                           use_downscaling=False, scale_factor=0.5,
                           use_parallel=False):
    """
    Advanced VMAF calculation with multiple optimization strategies.
    
    Args:
        input_file: Path to reference video
        encoded_file: Path to encoded/distorted video
        use_sampling: Whether to use clip sampling (default: True)
        num_clips: Number of clips to sample if sampling is used (default: 3)
        clip_duration: Duration of each clip in seconds if sampling is used (default: 2)
        use_downscaling: Whether to downscale videos before VMAF calculation (default: False)
        scale_factor: Scale factor for downscaling (default: 0.5)
        use_parallel: Whether to process clips in parallel (default: False)
        
    Returns:
        Average VMAF score
    """
  
    # Check if input files exist
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' does not exist")
        return None
        
    if not os.path.exists(encoded_file):
        print(f"Error: Encoded file '{encoded_file}' does not exist")
        return None
    
    # Create a directory for our temporary files
    temp_dir = tempfile.mkdtemp()
    temp_files = []
    
    # Helper function to extract VMAF score from metrics
    def extract_vmaf_score(metrics):
        """Extract VMAF score from metrics dictionary"""
        try:
            if 'vmaf' in metrics and metrics['vmaf']:
                # Calculate average of frame VMAF scores
                vmaf_score = sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"])
                return round(vmaf_score, 2)
            else:
                print("No VMAF data found in metrics")
                return None
        except Exception as e:
            print(f"Error extracting VMAF score: {e}")
            return None
    
    try:
        # ---------- Handle downscaling if enabled ----------
        if use_downscaling:
            print(f"Downscaling videos to {int(scale_factor * 100)}% for faster VMAF calculation")
            scaled_ref = os.path.join(temp_dir, "ref_scaled.mp4")
            scaled_enc = os.path.join(temp_dir, "enc_scaled.mp4")
            temp_files.extend([scaled_ref, scaled_enc])
            
            # Get video dimensions
            cmd = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height', '-of', 'json', input_file
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            data = json.loads(result.stdout)
            
            try:
                width = int(int(data['streams'][0]['width']) * scale_factor)
                height = int(int(data['streams'][0]['height']) * scale_factor)
                
                # Must be even numbers for YUV formats
                width = width - (width % 2)
                height = height - (height % 2)
                
                # Downscale reference video
                subprocess.run([
                    'ffmpeg', '-y', '-i', input_file, 
                    '-vf', f'scale={width}:{height}',
                    '-c:v', 'libx264', scaled_ref
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Downscale encoded video
                subprocess.run([
                    'ffmpeg', '-y', '-i', encoded_file, 
                    '-vf', f'scale={width}:{height}',
                    '-c:v', 'libx264', scaled_enc
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                # Replace original paths with scaled versions
                input_file_orig = input_file
                encoded_file_orig = encoded_file
                input_file = scaled_ref
                encoded_file = scaled_enc
                
                print(f"Videos successfully downscaled to {width}x{height}")
            except (KeyError, ValueError, subprocess.SubprocessError) as e:
                print(f"Downscaling failed: {e}, using original resolution")
        
        # ---------- Handle clip sampling if enabled ----------
        if use_sampling:
            # Get video duration
            cmd = [
                'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                '-of', 'json', encoded_file
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            data = json.loads(result.stdout)
            
            try:
                duration = float(data['format']['duration'])
                print(f"Video duration: {duration}s")
            except (KeyError, ValueError) as e:
                print(f"Could not determine video duration: {e}, falling back to full calculation")
                # Calculate VMAF on full videos
                ffqm = FfmpegQualityMetrics(input_file, encoded_file)
                metrics = ffqm.calculate(["vmaf"])
                return extract_vmaf_score(metrics)
            
            # Skip sampling for short videos
            if duration <= num_clips * clip_duration * 2:
                print(f"Video too short ({duration}s), calculating full VMAF")
                ffqm = FfmpegQualityMetrics(input_file, encoded_file)
                metrics = ffqm.calculate(["vmaf"])
                return extract_vmaf_score(metrics)
            
            # Calculate strategic sample points (beginning, middle, end)
            sample_points = []
            
            # Beginning segment (first 30%)
            if num_clips >= 1:
                begin_point = duration * 0.1  # 10% into the video
                sample_points.append(begin_point)
            
            # Middle segment (middle 40%)
            if num_clips >= 2:
                middle_point = duration * 0.5  # 50% into the video
                sample_points.append(middle_point)
            
            # End segment (last 30%)
            if num_clips >= 3:
                end_point = duration * 0.9  # 90% into the video
                sample_points.append(end_point)
            
            # Add more evenly distributed points if requested
            if num_clips > 3:
                for i in range(4, num_clips + 1):
                    pos = duration * (i - 0.5) / (num_clips + 1)
                    sample_points.append(pos)
            
            print(f"Using {len(sample_points)} clips for VMAF calculation")
            
            # ---------- Define clip processing function for parallel/sequential use ----------
            def process_clip(start_time):
                """Process a single clip for VMAF calculation"""
                # Adjust start time to ensure we don't go beyond video duration
                start_time = max(0, min(start_time, duration - clip_duration))
                
                # Create temp files for this clip
                ref_clip = os.path.join(temp_dir, f"ref_clip_{start_time:.2f}.mp4")
                enc_clip = os.path.join(temp_dir, f"enc_clip_{start_time:.2f}.mp4")
                
                # Extract clip from reference video
                cmd = [
                    'ffmpeg', '-y', '-ss', str(start_time), '-i', input_file,
                    '-t', str(clip_duration), '-c', 'copy', ref_clip
                ]
                ref_result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if ref_result.returncode != 0:
                    print(f"Failed to extract reference clip at {start_time}s")
                    print(f"Error: {ref_result.stderr.decode('utf-8')}")
                    return None
                
                # Extract clip from encoded video
                cmd = [
                    'ffmpeg', '-y', '-ss', str(start_time), '-i', encoded_file,
                    '-t', str(clip_duration), '-c', 'copy', enc_clip
                ]
                enc_result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                
                if enc_result.returncode != 0:
                    print(f"Failed to extract encoded clip at {start_time}s")
                    print(f"Error: {enc_result.stderr.decode('utf-8')}")
                    return None
                
                temp_files.extend([ref_clip, enc_clip])
                
                # Calculate VMAF for this clip
                try:
                    ffqm = FfmpegQualityMetrics(ref_clip, enc_clip)
                    metrics = ffqm.calculate(["vmaf"])
                    
                    # Extract VMAF score using our helper function
                    clip_vmaf = extract_vmaf_score(metrics)
                    if clip_vmaf is not None:
                        print(f"Clip at {start_time:.2f}s VMAF: {clip_vmaf}")
                    return clip_vmaf
                except Exception as e:
                    print(f"Error calculating VMAF for clip at {start_time:.2f}s: {e}")
                    return None
            
            # ---------- Process clips in parallel or sequentially ----------
            vmaf_scores = []
            
            if use_parallel:
                # Process clips in parallel
                with concurrent.futures.ThreadPoolExecutor(max_workers=min(num_clips, os.cpu_count())) as executor:
                    futures = [executor.submit(process_clip, start_time) for start_time in sample_points]
                    
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            result = future.result()
                            if result is not None:
                                vmaf_scores.append(result)
                        except Exception as e:
                            print(f"Error processing clip: {e}")
            else:
                # Process clips sequentially
                for start_time in sample_points:
                    result = process_clip(start_time)
                    if result is not None:
                        vmaf_scores.append(result)
            
            # Average the scores
            if vmaf_scores:
                avg_vmaf = sum(vmaf_scores) / len(vmaf_scores)
                return round(avg_vmaf, 2)
            else:
                print("No valid VMAF scores calculated from clips, trying direct calculation")
                # Fall back to original calculation
                try:
                    ffqm = FfmpegQualityMetrics(input_file, encoded_file)
                    metrics = ffqm.calculate(["vmaf"])
                    return extract_vmaf_score(metrics)
                except Exception as e:
                    print(f"Direct VMAF calculation failed: {e}")
                    return None
        else:
            # No sampling, calculate VMAF on entire video
            try:
                print("Calculating full VMAF (no sampling)...")
                ffqm = FfmpegQualityMetrics(input_file, encoded_file)
                metrics = ffqm.calculate(["vmaf"])
                return extract_vmaf_score(metrics)
            except Exception as e:
                print(f"Full VMAF calculation failed: {e}")
                return None
            
    except Exception as e:
        print(f"Error in VMAF calculation: {e}")
        # Fall back to direct FFmpeg VMAF calculation
        try:
            print("Attempting direct FFmpeg VMAF calculation...")
            cmd = [
                'ffmpeg', '-i', input_file, '-i', encoded_file,
                '-filter_complex', '[0:v]setpts=PTS-STARTPTS[reference];[1:v]setpts=PTS-STARTPTS[distorted];[reference][distorted]libvmaf=log_fmt=json',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            # Extract VMAF score from output
            for line in result.stderr.splitlines():
                if "VMAF score:" in line:
                    score = float(line.split("VMAF score:")[1].strip())
                    print(f"Direct FFmpeg VMAF score: {score}")
                    return round(score, 2)
            
            print("Could not extract VMAF score from FFmpeg output")
            return None
        except Exception as e2:
            print(f"Fatal error calculating VMAF: {e2}")
            return None
        
    finally:
        # Clean up temp files and directory
        for file in temp_files:
            try:
                if os.path.exists(file):
                    os.unlink(file)
            except Exception as e:
                print(f"Error cleaning up temp file {file}: {e}")
        
        try:
            os.rmdir(temp_dir)
        except Exception as e:
            print(f"Error removing temp directory: {e}")



def calculate_parallel_vmaf(reference_scenes, encoded_files,use_downscaling=False, scale_factor=0.5):
    """Process VMAF calculations in parallel for multiple scenes
    
    Args:
        reference_scenes: List of reference scene file names in the temp folder
        encoded_files: List of encoded file paths
        temp_folder: Path to the temp folder containing reference scenes
        
    Returns:
        Tuple of (results list, total processing time)
    """
    print("\n== PARALLEL VMAF PROCESSING ==")
    start_time = time.time()
    results = []
    
    # Ensure we have valid inputs
    if len(reference_scenes) != len(encoded_files):
        print(f"Warning: Number of reference scenes ({len(reference_scenes)}) doesn't match encoded files ({len(encoded_files)})")
        # Use the minimum length to avoid index errors
        scene_count = min(len(reference_scenes), len(encoded_files))
    else:
        scene_count = len(reference_scenes)
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # Start all VMAF calculations in parallel
        futures = []
        for i in range(scene_count):
            # Construct full path for reference scene from temp folder
            reference_path = reference_scenes[i]
            encoded_path = encoded_files[i]
            if use_downscaling:
                futures.append((i+1, executor.submit(
                calculate_vmaf_advanced, reference_path, encoded_path, use_downscaling=True, scale_factor=scale_factor,
                )))
            else:
                futures.append((i+1, executor.submit(
                    calculate_vmaf_advanced, reference_path, encoded_path
                )))
        # Collect results as they complete
        for i, future in futures:
            try:
                vmaf = future.result()
                scene_result = {
                    'scene': i,
                    'reference': reference_scenes[i-1],
                    'encoded': os.path.basename(encoded_files[i-1]),
                    'vmaf': vmaf,
                }
                results.append(scene_result)
                print(f"Scene {i} VMAF: {vmaf}")
            except Exception as e:
                print(f"Error processing scene {i}: {e}")
    
    total_time = time.time() - start_time
    print(f"Total parallel processing time: {total_time:.2f}s")
    extracted_results= extract_vmaf_scores(results)
    return [score for _, score in extracted_results]


def extract_vmaf_scores(results):
    """
    Extract scene numbers and VMAF values from VMAF calculation results
    
    Args:
        results: List of dictionaries containing scene information and VMAF scores
        
    Returns:
        List of tuples containing (scene_number, vmaf_score)
    """
    vmaf_values = []
    for result in results:
        scene_number = result['scene']
        vmaf_score = result['vmaf']
        vmaf_values.append((scene_number, vmaf_score))
    
    return vmaf_values


# Example usage:
# results = [{'scene': 1, 'reference': './videos/temp_scenes/scene_000.mp4', 
#             'encoded': 'output_scene_0_av1_nvenc.mp4', 'vmaf': 86.37}, 
#            {'scene': 2, 'reference': './videos/temp_scenes/scene_001.mp4', 
#             'encoded': 'output_scene_1_av1_nvenc.mp4', 'vmaf': 90.29}, 
#            {'scene': 3, 'reference': './videos/temp_scenes/scene_002.mp4', 
#             'encoded': 'output_scene_2_av1_nvenc.mp4', 'vmaf': 93.01}]
# 
# vmaf_scores = extract_vmaf_scores(results)
# print(vmaf_scores)  # [(1, 86.37), (2, 90.29), (3, 93.01)]
#
# # To get just the VMAF values:
# vmaf_values = [score for _, score in vmaf_scores]
# print(vmaf_values)  # [86.37, 90.29, 93.01]
#
# # To get average VMAF:
# avg_vmaf = sum(vmaf_values) / len(vmaf_values)
# print(f"Average VMAF: {avg_vmaf:.2f}")  # Average VMAF: 89.89
