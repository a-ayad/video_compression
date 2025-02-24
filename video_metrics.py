import cv2
import numpy as np

# Example usage:
avg_motion = 12.0      # Example measured motion metric
avg_edge_density = 0.25  # Example edge density (fraction of edge pixels)
avg_texture = 3.0      # Example texture complexity (entropy)




def compute_optical_flow_metric(prev_gray, curr_gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15, iterations=3,
        poly_n=5, poly_sigma=1.2, flags=0)
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(magnitude)

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






def analyze_video(video_path, max_frames=100, scale_factor=0.5):
    """
    Process the video, scaling each frame by scale_factor before computing metrics.
    
    Parameters:
      video_path: Path to the video file.
      max_frames: Maximum number of frames to process.
      scale_factor: Factor to scale down the frame resolution.
    
    Returns:
      Tuple with average motion, edge density, and texture complexity.
    """
    cap = cv2.VideoCapture(video_path)
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read video")
        return None

    # Optionally, scale down the frame
    if scale_factor != 1.0:
        prev_frame = cv2.resize(prev_frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        #print("Resized frame to: {}".format(prev_frame.shape))
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    
    motion_metrics = []
    edge_densities = []
    texture_complexities = []
    
    frame_count = 1
    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if scale_factor != 1.0:
            frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_LINEAR)
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Compute metrics for this frame pair
        motion = compute_optical_flow_metric(prev_gray, curr_gray)
        motion_metrics.append(motion)
        
        density = compute_edge_density(curr_gray)
        texture = compute_texture_complexity(curr_gray)
        edge_densities.append(density)
        texture_complexities.append(texture)
        
        prev_gray = curr_gray
        frame_count += 1

    cap.release()
    
    avg_motion = np.mean(motion_metrics) if motion_metrics else 0
    avg_edge_density = np.mean(edge_densities) if edge_densities else 0
    avg_texture = np.mean(texture_complexities) if texture_complexities else 0
    
    #print("Average Motion Metric: {:.3f}".format(avg_motion))
    #print("Average Edge Density: {:.3f}".format(avg_edge_density))
    #print("Average Texture Complexity (Entropy): {:.3f}".format(avg_texture))


    
    return avg_motion, avg_edge_density, avg_texture


if __name__ == "__main__":
   
# Example usage:
    video_path = './videos/videos_2/output-Scene-002.mp4'
    avg_motion, avg_edge_density, avg_texture = analyze_video(video_path, max_frames=150, scale_factor=0.5)
    suggested_crf = lookup_crf(avg_motion, avg_edge_density, avg_texture)
    print("Suggested CRF:", suggested_crf)

