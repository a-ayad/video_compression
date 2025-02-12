from scenedetect import open_video, SceneManager, split_video_ffmpeg,save_images,VideoManager,backends
from scenedetect.detectors import ContentDetector,HistogramDetector,hash_detector,AdaptiveDetector
from scenedetect.video_splitter import split_video_ffmpeg


def split_video_into_scenes(video_path, threshold=3):
    # Open our video, create a scene manager, and add a detector.
    # You can use VideoManager.FFMPEG or VideoManager.OPENCV.
    video_manager = VideoManager([video_path])
    
    scene_manager = SceneManager()
    scene_manager.add_detector(AdaptiveDetector(adaptive_threshold=3,min_scene_len=120))
    # Detect scenes using the video manager as the frame source.
    print(backends.AVAILABLE_BACKENDS)
    scene_manager.detect_scenes(frame_source=video_manager, show_progress=True)
    scene_list = scene_manager.get_scene_list()
    split_video_ffmpeg(video_path, scene_list, output_dir='./videos/videos_2/', output_file_template='$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4', show_progress=True)
    for i, scene in enumerate(scene_list):
        print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].get_frames(),
        scene[1].get_timecode(), scene[1].get_frames(),))
    
    save_images(scene_list,video_manager, num_images=2,output_dir="./videos/videos_2", image_name_template='$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER')

def main():
    # Example usage:
    video_path = "./videos/videos_2/output.y4m"
    
    split_video_into_scenes(video_path)



if __name__ == "__main__":
    main()