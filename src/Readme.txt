Pipeline architecture:

* main:
	1-split_video_into_scenes.py
	"this function splits the video into scenes using pyscene detect library"

	2-get_video_metrics.py
	"this segments calculate multiple metrics to describe the complexity of the video"


	2-find_optimal_cq.py
		2.1-predict_vmaf.py
		
		
	3-encode_video.py
	
	4- merge_videos.py

	5-calculate_vmaf.py