import os
import io
from typing import Callable
import cv2
import uuid
import pandas as pd
#import tempfile
import ultralytics
import torch
from yolo_helper import make_callback_adapter_with_counter, convert_tracking_results_to_pandas

#import av
#import subprocess
import moviepy.editor as moviepy

class VideoHandler():

    def __init__(self, video: bytes):
        """ Constructor """
        self.byte_video = io.BytesIO(video.read())
        self.temp_id = str(uuid.uuid4())
        self.video_path = self.temp_id + ".mp4"

        #write file to given video_path
        with open(self.video_path, 'wb') as out:
            out.write(self.byte_video.read())
        
    def get_video_path(self):
        return self.video_path
    
    # def get_tracked_video_file(self)

    def get_video_stats(self) -> tuple:
        """
        Retrieve statistics about a video file.

        Returns:
        - tuple: A tuple containing video statistics in the following format:
            (fps (int), width (int), height (int), total_frames (int))
        """
        vf = cv2.VideoCapture(self.video_path)

        fps = int(vf.get(cv2.CAP_PROP_FPS))
        width = int(vf.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vf.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(vf.get(cv2.CAP_PROP_FRAME_COUNT))

        return (fps, width, height, total_frames)
    
    def track(self,progressbar_callback: Callable) -> tuple[pd.DataFrame, str]:
        """
        Perform object tracking on the video with YOLOv8.
        Args:
            progressbar_callback (Callable[int]): a callback accepting 1 argument (frame number)
        Return:
            DataFrame with tracking results
            Processed video path
        """
        pretrained_model = "yolov8n.pt"
        model = ultralytics.YOLO(pretrained_model, verbose=True)
        yolo_progress_reporting_event = "on_predict_batch_start"
        progress_callback_wrapped = make_callback_adapter_with_counter(yolo_progress_reporting_event, 
                                                                       lambda _,counter: progressbar_callback(counter))
        model.add_callback(yolo_progress_reporting_event, progress_callback_wrapped)

        device = 0 if torch.cuda.is_available() else 'cpu' 
        outputdir=os.getcwd()
        print(f"Using outputdir: {outputdir}")
        tracking_results = model.track(source=self.video_path, conf=0.2, iou=0.6, show=False, device=device, stream=True, save=True, save_dir=outputdir, exist_ok=True, project=outputdir)
        results_df = convert_tracking_results_to_pandas(tracking_results)
        # NB - workaround for the bug in Ultralytics that ignores the path passed to save_dir
        #processed_video_path = os.path.join(tmpdir, "runs/detect/track", self.temp_id + ".avi")
        processed_video_path = os.path.join(outputdir, "track", self.temp_id + ".avi")
        converted_video_path = os.path.join(outputdir, "track", self.temp_id + ".mp4")
        # "TODO" - huge security hole! replace with encoding via PyAV library
        print(f"Converting the output ({processed_video_path}) to the format readable by streamlit")
        # os.system(f"ffmpeg -y -i {processed_video_path} -vcodec libx264 {converted_video_path}")
        # os.system("ffmpeg -y -i {processed_video_path} -vcodec libx264 {converted_video_path}")
        clip = moviepy.VideoFileClip(processed_video_path)
        clip.write_videofile(converted_video_path)
        """ ffmpeg_command = [
            'ffmpeg', '-y', '-i', processed_video_path, '-vcodec', 'libx264', converted_video_path
            ]

        # Run the command and capture output and errors
        result = subprocess.run(ffmpeg_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Check if the command was successful
        if result.returncode == 0:
           print("Conversion successful!")
        else:
            print(f"Error during conversion: {result.stderr.decode('utf-8')}")
         """
        
        """ # Using PyAV instead of os.system for video conversion
        print(f"Converting the output ({processed_video_path}) to the format readable by streamlit using PyAV")

        with av.open(processed_video_path) as input_container:
            with av.open(converted_video_path, mode='w') as output_container:
                # Create a video stream in the output container (H.264 codec)
                output_stream = output_container.add_stream('libx264')

                for frame in input_container.decode(video=0):  # Decode the input video
                    # Encode the video frames into the output stream
                    for packet in output_stream.encode(frame):
                        output_container.mux(packet)

                # Flush the stream at the end of the video to finalize encoding
                for packet in output_stream.encode(None):
                    output_container.mux(packet) """

        print(f"Conversion complete")
        return results_df, converted_video_path
    
    def __del__(self):
        """
        Remove a video file.
        """
        os.remove(self.video_path)
