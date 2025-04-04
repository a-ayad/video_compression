import cv2
import numpy as np
import time
import os


def analyze_video_fast(video_path, max_frames=100, scale_factor=0.5, sample_interval=3):
    """
    Fast video analysis without any calibration attempts - returns metrics in their
    native scale for model retraining.
    
    Args:
        video_path: Path to video file
        max_frames: Maximum number of frames to analyze
        scale_factor: Scale factor for resizing frames (smaller = faster)
        sample_interval: Only analyze every Nth frame (higher = faster)
        
    Returns:
        Dictionary with raw video metrics
    """
  
    total_start_time = time.time()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return None
    
    # Get video properties
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    resolution = (frame_width, frame_height)
    
    # Calculate number of frames to analyze
    frames_to_analyze = min(max_frames, frame_count)
    
    # Initialize arrays for metrics
    motion_metrics = []
    edge_densities = []
    texture_complexities = []
    temporal_infos = []
    spatial_infos = []
    color_complexities = []
    motion_variances = []
    saliency_metrics = []
    grain_noise_levels = []
    scene_change_count = 0
    
    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video")
        cap.release()
        return None
    
    # Resize frame if needed
    if scale_factor != 1.0:
        prev_frame = cv2.resize(prev_frame, None, fx=scale_factor, fy=scale_factor, 
                               interpolation=cv2.INTER_LINEAR)
    
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    # Process frames
    processed_frames = 1
    frame_idx = 0
    
    while processed_frames < frames_to_analyze:
        # Skip frames according to sample interval
        for _ in range(sample_interval - 1):
            ret = cap.grab()  # Just grab frame, don't decode
            if not ret:
                break
            frame_idx += 1
        
        # Read and decode frame
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_idx += 1
        processed_frames += 1
        
        # Resize frame
        if scale_factor != 1.0:
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, 
                               interpolation=cv2.INTER_LINEAR)
        
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # === MOTION METRICS (SPARSE OPTICAL FLOW) ===
        prev_points = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=100, qualityLevel=0.3, 
            minDistance=7, blockSize=7)
        
        if prev_points is not None and len(prev_points) > 0:
            next_points, status, _ = cv2.calcOpticalFlowPyrLK(
                prev_gray, curr_gray, prev_points, None,
                winSize=(15, 15), maxLevel=2)
            
            # Process points
            magnitudes = []
            
            for i, (new, old, flag) in enumerate(zip(next_points, prev_points, status)):
                if flag == 1:  # Point was found
                    old_pt = old.flatten()
                    new_pt = new.flatten()
                    
                    # Calculate distance
                    dx = new_pt[0] - old_pt[0]
                    dy = new_pt[1] - old_pt[1]
                    magnitude = np.sqrt(dx*dx + dy*dy)
                    magnitudes.append(magnitude)
            
            if magnitudes:
                # Use raw magnitudes - no scaling
                motion = np.mean(magnitudes)
                motion_var = np.std(magnitudes)
                
                motion_metrics.append(motion)
                motion_variances.append(motion_var)
        
        # If sparse optical flow failed, use block matching
        if len(motion_metrics) <= processed_frames - 1:
            # Simple block-based motion estimation
            block_size = 16
            motion_blocks = []
            
            # Divide image into blocks and calculate differences
            for y in range(0, curr_gray.shape[0], block_size):
                for x in range(0, curr_gray.shape[1], block_size):
                    # Get block regions
                    y_end = min(y + block_size, curr_gray.shape[0])
                    x_end = min(x + block_size, curr_gray.shape[1])
                    
                    # Extract blocks
                    prev_block = prev_gray[y:y_end, x:x_end]
                    curr_block = curr_gray[y:y_end, x:x_end]
                    
                    # Calculate mean absolute difference
                    diff = np.abs(prev_block.astype(np.float32) - curr_block.astype(np.float32))
                    block_motion = np.mean(diff)
                    motion_blocks.append(block_motion)
            
            # Calculate motion statistics
            avg_motion = np.mean(motion_blocks)
            motion_variance = np.std(motion_blocks)
            
            motion_metrics.append(avg_motion)
            motion_variances.append(motion_variance)
        
        # === OTHER METRICS ===
        # These functions need to be imported or defined
        try:
            # Edge density
            edge_density = compute_edge_density(curr_gray)
            edge_densities.append(edge_density)
            
            # Texture complexity
            texture = compute_texture_complexity(curr_gray)
            texture_complexities.append(texture)
            
            # Temporal information
            ti = compute_temporal_information(prev_gray, curr_gray)
            temporal_infos.append(ti)
            
            # Spatial information (compute less frequently)
            if processed_frames % 2 == 0:
                si = compute_spatial_information(curr_gray)
                spatial_infos.append(si)
            
            # Color complexity
            cc = compute_color_complexity(frame)
            color_complexities.append(cc)
            
            # Grain noise and saliency (compute less frequently)
            if processed_frames % 5 == 0:
                try:
                    saliency_val = compute_saliency_metric(frame)
                    saliency_metrics.append(saliency_val)
                except:
                    saliency_metrics.append(0.0)
                
                grain_noise = compute_grain_noise_level(curr_gray)
                grain_noise_levels.append(grain_noise)
            
            # Scene change detection
            if is_scene_change(prev_gray, curr_gray, threshold=30):
                scene_change_count += 1
        except NameError as e:
            print(f"Error: {e}. Make sure the required functions are defined or imported.")
            cap.release()
            return None
        
        # Update previous frame
        prev_gray = curr_gray
    
    # Release video
    cap.release()
    
    # Fill in missing values for metrics we calculated less frequently
    while len(spatial_infos) < len(motion_metrics):
        spatial_infos.append(spatial_infos[-1] if spatial_infos else 0)
    
    while len(saliency_metrics) < len(motion_metrics):
        saliency_metrics.append(saliency_metrics[-1] if saliency_metrics else 0)
        
    while len(grain_noise_levels) < len(motion_metrics):
        grain_noise_levels.append(grain_noise_levels[-1] if grain_noise_levels else 0)
    
    # Combine metrics
    results = {
        "metrics_avg_motion": np.mean(motion_metrics) if motion_metrics else 0,
        "metrics_avg_edge_density": np.mean(edge_densities) if edge_densities else 0,
        "metrics_avg_texture": np.mean(texture_complexities) if texture_complexities else 0,
        "metrics_avg_temporal_information": np.mean(temporal_infos) if temporal_infos else 0,
        "metrics_avg_spatial_information": np.mean(spatial_infos) if spatial_infos else 0,
        "metrics_avg_color_complexity": np.mean(color_complexities) if color_complexities else 0,
        "metrics_scene_change_count": scene_change_count,
        "metrics_avg_motion_variance": np.mean(motion_variances) if motion_variances else 0,
        "metrics_avg_saliency": np.mean(saliency_metrics) if saliency_metrics else 0,
        "metrics_avg_grain_noise": np.mean(grain_noise_levels) if grain_noise_levels else 0,
        "metrics_frame_rate": frame_rate,
        "metrics_resolution": resolution
    }
    
    total_time = time.time() - total_start_time
    print(f"Fast analysis of {os.path.basename(video_path)} completed in {total_time:.2f} seconds")
    
    return results


# Helper functions for video analysis
def compute_edge_density(gray_frame, threshold1=100, threshold2=200):

    edges = cv2.Canny(gray_frame, threshold1, threshold2)
    edge_pixels = np.count_nonzero(edges)
    total_pixels = edges.size
    return edge_pixels / total_pixels

def compute_texture_complexity(gray_frame):

    hist = cv2.calcHist([gray_frame], [0], None, [256], [0, 256]).ravel()
    hist_norm = hist / hist.sum()  # normalize histogram
    entropy = -np.sum([p * np.log2(p) for p in hist_norm if p > 0])
    return entropy

def compute_temporal_information(prev_gray, curr_gray):
 
    diff = cv2.absdiff(prev_gray, curr_gray)
    return np.std(diff)

def compute_spatial_information(gray_frame):
   
    sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.std(gradient_magnitude)

def compute_color_complexity(frame):
    
    channels = cv2.split(frame)
    entropies = []
    for channel in channels:
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256]).ravel()
        hist_sum = np.sum(hist)
        if hist_sum > 0:
            hist_norm = hist / hist_sum
            entropy = -np.sum([p * np.log2(p) for p in hist_norm if p > 0])
        else:
            entropy = 0
        entropies.append(entropy)
    return np.mean(entropies)

def is_scene_change(prev_gray, curr_gray, threshold=30):

    diff = cv2.absdiff(prev_gray, curr_gray)
    mean_diff = np.mean(diff)
    return mean_diff > threshold

def compute_saliency_metric(frame):

    try:
        saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        success, saliencyMap = saliency_detector.computeSaliency(frame)
        if success:
            saliencyMap = (saliencyMap * 255).astype("uint8")
            return np.mean(saliencyMap)
    except:
        pass
    return 0

def compute_grain_noise_level(gray_frame, kernel_size=3):

    blurred = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)
    noise = gray_frame.astype(np.float32) - blurred.astype(np.float32)
    return np.std(noise)



if __name__ == "__main__":
    video_path = './videos/temp_scenes/scene_000.mp4'
    
    metrics2= analyze_video_fast(video_path,max_frames=50)
    print("optimized metrics")
    if metrics2:        
        for key, value in metrics2.items():
            if isinstance(value, tuple):
                print(f"{key}: {value[0]}x{value[1]}")
            else:
                print(f"{key}: {value:.3f}")
