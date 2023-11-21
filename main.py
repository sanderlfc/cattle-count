import numpy as np
from deep_sort.deep_sort import DeepSort

import time
import cv2
from ultralytics import YOLO

# Path to the DeepSort model weights:
ds_model_weights = '/Users/dd/ikt213g23h/assignments/solutions/mainprosjekt/deep_sort/deep/checkpoint/ckpt.t7'

# DeepSort tracker initialization:
ds_tracker = DeepSort(model_path=ds_model_weights, max_age=70)

# Loading input video:
video = '/Users/dd/ikt213g23h/assignments/solutions/mainprosjekt/video/cow.mp4'
capture = cv2.VideoCapture(video)

video_output = '{}_out.mp4'.format(video)

# Capture properties:
width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) #gets width
height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) #gets height
fps = capture.get(cv2.CAP_PROP_FPS) #gets frames per second

# Define output video properties:
output_f = cv2.VideoWriter_fourcc(*'mp4v') #creating four character code/ formate for video
output_path = 'output.mp4'
out = cv2.VideoWriter(output_path, output_f, fps, (width, height))



# creating list to store frames and set to store unique track id's:
frames = []
unique_track_ids = set()

i = 0

# setting variables to 0:
count, fps, elaps = 0, 0, 0
start_time = time.perf_counter()

# Main loop for video processing
while capture.isOpened():
    # Read a frame from the video
    ret, frame = capture.read()

    if ret:
        # Convert the frame to RGB format
        og_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = og_frame.copy()

        model2 = YOLO('/Users/dd/ikt213g23h/assignments/solutions/mainprosjekt/runs/detect/train39/weights/last.pt')
        # Perform object detection using YOLO
        detection_results = model2(frame, device='cpu', classes=0, conf=0.5)
        classes = ['cattle']

        # loops through YOLO detection results and gets attributes:
        for detection_result in detection_results:
            boxes = detection_result.boxes
            probs = detection_result.probs #gets probabilities
            cls = boxes.cls.tolist() #gets classes and input to list
            xyxy = boxes.xyxy #gets box coordinates
            conf = boxes.conf #get confidence
            xywh = boxes.xywh #gets coordinates

            #loops to classes and get class name from list:
            for classIndex in cls:
                className = classes[int(classIndex)]

        # Convert relevant info to numpy arrays:
        pred_cls = np.array(cls) #convert classes
        conf = conf.detach().cpu().numpy()
        xyxy = xyxy.detach().cpu().numpy()
        bboxes_xywh = xywh  #setting new variable
        bboxes_xywh = xywh.cpu().numpy()
        bboxes_xywh = np.array(bboxes_xywh, dtype=float) #setting datatype float

        # Update DeepSort tracker with object detections:
        tracks = ds_tracker.update(bboxes_xywh, conf, og_frame)

        # Process each track and draw bounding boxes on the frame:
        for track in ds_tracker.tracker.tracks:
            track_id = track.track_id
            hits = track.hits
            x1, y1, x2, y2 = track.to_tlbr()  # Get bounding box coordinates in (x1, y1, x2, y2) format
            w = x2 - x1  # Calculate width
            h = y2 - y1  # Calculate height

            # setting colors.
            blue = (255, 0, 0)
            green = (0, 255, 0)
            red = (0, 0, 255)


            # Solution for assigning colors to boxes based on track_id for giving variation:
            color_ID = track_id % 3
            if color_ID == 0:
                color = red
            elif color_ID == 1:
                color = blue
            else:
                color = green

            # Drawing bounding box on frame:
            cv2.rectangle(og_frame, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color, 2)

            # Adding text with name of class and tracking id:
            text_color = (0, 0, 0)  # Black color for text
            cv2.putText(og_frame, f"{className}-{track_id}", (int(x1) + 10, int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            # Adding track_ids to the set with unique IDs:
            unique_track_ids.add(track_id)

        #counting the amount of cattle based on the length:
        cattle_count = len(unique_track_ids)

        # Update FPS and place on frame:
        current_time = time.perf_counter()
        elaps = (current_time - start_time)
        count += 1
        if elaps > 1:
            fps = count / elaps
            count = 0
            start_time = current_time

        # cattle count:
        cv2.putText(og_frame, f"Cattle Count: {cattle_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),2, cv2.LINE_AA)

        frames.append(og_frame)         # Append frame to list

        # Write the frame to the output video file
        out.write(cv2.cvtColor(og_frame, cv2.COLOR_RGB2BGR))

capture.release()
out.release()
cv2.destroyAllWindows()