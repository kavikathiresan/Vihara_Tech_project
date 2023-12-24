# Yolov3 Object detection
YOLOv3 (You Only Look Once, Version 3) is a real-time object detection algorithm that identifies specific objects in videos, live feeds, or images. The YOLO machine learning algorithm uses features learned by a deep convolutional neural network to detect an object.
Detecting objects in a video using YOLO V3 algorithm. The approach is quite similar to detecting images with YOLO. We get every frame of a video like an image and detect objects at that frame using yolo. Then draw the boxes, labels and iterate through all the frame in a given video. Adjust the confidence and nms threshold to see how the algorithm's detections change. The annotated video will be stored in the output folder in .mp4 file format. Make sure to add yolov3.weights file to the model folder to build and run with docker.

## The directories structure should as follow:

yolo-coco-data/ : The YOLOv3 object detector pre-trained (on the COCO dataset) model files. These were trained by the Darknet team should be kept here.

images/ : This folder should contain static images which we will be used to perform object detection on for testing and evaluation purposes.

videos/ : This directory should contains sample test videos for testing. After performing object detection with YOLO on video, we’ll process videos in real time camera input. Also Output videos that have been processed by YOLO and annotated with bounding boxes and class names will appear at this location.
# Running This Command
python filename.py --video path/file/location
# Redis Database 
Redis offers a fast, in-memory data store to power live streaming use cases. Redis can be used to store metadata about users' profiles and viewing histories, authentication information/tokens for millions of users, and manifest files to enable CDNs to stream videos to millions of mobile and desktop users at a time.
- its a catching system
Here I am storing my video object detected person frame in redis database 
