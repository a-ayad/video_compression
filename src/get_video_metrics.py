import cv2
import numpy as np

def compute_optical_flow_metric(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

def compute_motion_variance(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.std(magnitude)

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
    # Use Sobel to compute gradients in x and y directions
    sobelx = cv2.Sobel(gray_frame, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray_frame, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    return np.std(gradient_magnitude)

def compute_color_complexity(frame):
    # Compute the average entropy across each color channel
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
    # A simple scene change detector based on the mean absolute difference.
    diff = cv2.absdiff(prev_gray, curr_gray)
    mean_diff = np.mean(diff)
    return mean_diff > threshold

def compute_saliency_metric(frame):
    """
    Compute a saliency metric using OpenCV's saliency module.
    Tries the spectral residual method first; if it returns a very low value,
    falls back to the fine-grained method.
    Returns the mean saliency value scaled to 0-255.
    """
    mean_saliency = 0

    # First attempt: Spectral Residual method.
    try:
        saliency_detector = cv2.saliency.StaticSaliencySpectralResidual_create()
        (success, saliencyMap) = saliency_detector.computeSaliency(frame)
        if success:
            saliencyMap = (saliencyMap * 255).astype("uint8")
            mean_saliency = np.mean(saliencyMap)
    except Exception as e:
        # If any error occurs, we will try the alternative method.
        pass

    # If the spectral residual method returns a very low value, try fine-grained.
    if mean_saliency < 1:
        try:
            saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
            (success, saliencyMap) = saliency_detector.computeSaliency(frame)
            if success:
                saliencyMap = (saliencyMap * 255).astype("uint8")
                mean_saliency = np.mean(saliencyMap)
        except Exception as e:
            pass

    return mean_saliency

def compute_grain_noise_level(gray_frame, kernel_size=3):
    """
    Estimate the grain noise level in the frame.
    This is done by applying a Gaussian blur to remove high-frequency content and
    computing the standard deviation of the difference between the original frame and the blurred version.
    """
    blurred = cv2.GaussianBlur(gray_frame, (kernel_size, kernel_size), 0)
    noise = gray_frame.astype(np.float32) - blurred.astype(np.float32)
    noise_std = np.std(noise)
    return noise_std

def analyze_video(video_path, max_frames=100, scale_factor=0.5):
    """
    Process the video and compute a suite of metrics:
      - Motion (mean optical flow magnitude)
      - Edge Density
      - Texture Complexity (entropy)
      - Temporal Information (std. of frame differences)
      - Spatial Information (std. of Sobel gradient)
      - Color Complexity (entropy over color channels)
      - Motion Variance (std. of optical flow magnitude)
      - Saliency Metric (average saliency from spectral residual)
      - Scene Change Count (number of significant frame changes)
      - Grain Noise Level (std. of residual noise after blurring)
      - Frame Rate
      - Resolution
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video")
        return None

    # Get frame rate and resolution
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    resolution = (frame_width, frame_height)

    # Resize frame if needed.
    if scale_factor != 1.0:
        prev_frame = cv2.resize(prev_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Lists to hold metrics computed per frame or frame pair.
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

    frame_count = 1
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if scale_factor != 1.0:
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Existing metrics.
        motion = compute_optical_flow_metric(prev_gray, curr_gray)
        motion_metrics.append(motion)
        density = compute_edge_density(curr_gray)
        edge_densities.append(density)
        texture = compute_texture_complexity(curr_gray)
        texture_complexities.append(texture)

        # Additional metrics.
        ti = compute_temporal_information(prev_gray, curr_gray)
        temporal_infos.append(ti)
        si = compute_spatial_information(curr_gray)
        spatial_infos.append(si)
        cc = compute_color_complexity(frame)
        color_complexities.append(cc)
        mv = compute_motion_variance(prev_gray, curr_gray)
        motion_variances.append(mv)
        saliency_val = compute_saliency_metric(frame)
        saliency_metrics.append(saliency_val)
        grain_noise = compute_grain_noise_level(curr_gray)
        grain_noise_levels.append(grain_noise)

        # Scene change detection.
        if is_scene_change(prev_gray, curr_gray, threshold=30):
            scene_change_count += 1

        prev_gray = curr_gray
        frame_count += 1

    cap.release()

    # Compute averages for each metric.
    results = {
        "avg_motion": np.mean(motion_metrics) if motion_metrics else 0,
        "avg_edge_density": np.mean(edge_densities) if edge_densities else 0,
        "avg_texture": np.mean(texture_complexities) if texture_complexities else 0,
        "avg_temporal_information": np.mean(temporal_infos) if temporal_infos else 0,
        "avg_spatial_information": np.mean(spatial_infos) if spatial_infos else 0,
        "avg_color_complexity": np.mean(color_complexities) if color_complexities else 0,
        "scene_change_count": scene_change_count,
        "avg_motion_variance": np.mean(motion_variances) if motion_variances else 0,
        "avg_saliency": np.mean(saliency_metrics) if saliency_metrics else 0,
        "avg_grain_noise": np.mean(grain_noise_levels) if grain_noise_levels else 0,
        "frame_rate": frame_rate,
        "resolution": resolution
    }

    return results

if __name__ == "__main__":
    video_path = './videos/input_videos/720p50_mobcal_ter.y4m'
    metrics = analyze_video(video_path, max_frames=1000, scale_factor=0.5)
    if metrics:
        print("Video Metrics:")
        for key, value in metrics.items():
            if isinstance(value, tuple):
                print(f"{key}: {value[0]}x{value[1]}")
            else:
                print(f"{key}: {value:.3f}")