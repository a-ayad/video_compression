
from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg, save_images
import datetime
import time


def split_video_into_scenes(video_path, threshold=3):
    # Open our video, create a scene manager, and add a detector.
    # You can use VideoManager.FFMPEG or VideoManager.OPENCV.
    
    # Detect scenes using the video manager as the frame source.
    scene_list = detect(video_path, AdaptiveDetector())
     
    scenes_duration=[] 
    for i, scene in enumerate(scene_list):
        print('    Scene %2d: Start %s / Frame %d, End %s / Frame %d' % (
        i+1,
        scene[0].get_timecode(), scene[0].get_frames(),
        scene[1].get_timecode(), scene[1].get_frames(),))
        scene_duration_in_frames=int(scene[1].get_frames()-scene[0].get_frames())
        scene_start_time=  time.strptime(scene[0].get_timecode().split(',')[0],'%H:%M:%S.%f')
        scene_start_time = datetime.timedelta(hours=scene_start_time.tm_hour,minutes=scene_start_time.tm_min,seconds=scene_start_time.tm_sec).total_seconds()
        scene_end_time=  time.strptime(scene[1].get_timecode().split(',')[0],'%H:%M:%S.%f')
        scene_end_time = datetime.timedelta(hours=scene_end_time.tm_hour,minutes=scene_end_time.tm_min,seconds=scene_end_time.tm_sec).total_seconds()
        scene_duration_in_seconds= scene_end_time-scene_start_time
        frame_rate= scene_duration_in_frames/scene_duration_in_seconds
        scenes_duration.append(scene_duration_in_frames)
        print("Frame rate: ", frame_rate)
        print("Scenes duration in frames: ", scene_duration_in_frames)
        print("Scenes duration in seconds: ", scene_duration_in_seconds)

    split_video_ffmpeg(video_path, scene_list,output_dir='./videos/input_videos', output_file_template='$VIDEO_NAME-Scene-$SCENE_NUMBER.y4m',
                    show_progress=True, arg_override='-c copy')

    print(f"Video split into {len(scene_list)} scenes without re-encoding")
    print("Scenes duration: ", scenes_duration)
    return scenes_duration
    
    #save_images(scene_list,video_manager, num_images=2,output_dir="./videos/videos_2", image_name_template='$VIDEO_NAME-Scene-$SCENE_NUMBER-$IMAGE_NUMBER')

    
 

if __name__ == "__main__":
    
   # Example usage:
    video_path = "./videos/input_videos/input_1.y4m"
    
    split_video_into_scenes(video_path)




