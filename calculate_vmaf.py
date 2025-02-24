from ffmpeg_quality_metrics import FfmpegQualityMetrics




def calculate_vmaf(input_file, encoded_file):
    """
    Calculates the VMAF and other quality metrics using ffmpeg-quality-metrics.
    """
    try:
        # Example: downscale frames and sample every 10th frame.
        ffqm = FfmpegQualityMetrics(input_file, encoded_file)
        metrics = ffqm.calculate(["vmaf"])
        # Average the VMAF values over all frames
        avgvmaf = sum([frame["vmaf"] for frame in metrics["vmaf"]]) / len(metrics["vmaf"])
        #print(f"VMAF: {round(avgvmaf, 2)}")
        return round(avgvmaf, 2)
    except Exception as e:
        print(f"Error calculating quality metrics: {e}")
        return None,
