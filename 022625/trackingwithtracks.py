from collections import defaultdict

import cv2
from ultralytics import YOLO
import numpy as np


model = YOLO('/Users/julianna/Desktop/spermTrack/022625/run13/weights/best.pt')  # Load a custom trained model
results = model.track(source='/Users/julianna/Downloads/VISEM_tracking/VISEM_Tracking_Train_v4/Train/14/14.mp4', show=True, tracker="bytetrack.yaml", stream=True)

video_path = '/Users/julianna/Downloads/VISEM_tracking/VISEM_Tracking_Train_v4/Train/14/14.mp4'
cap = cv2.VideoCapture(video_path)
track_history = defaultdict(lambda: [])

#set up the VideoWriter objectq
fourcc = cv2.VideoWriter_fourcc(*'MP4V')  # You can also use other codecs like 'MP4V' or 'MJPG'
out = cv2.VideoWriter('/Users/julianna/Desktop/spermTrack/022625/run13/videos/14tracks.mp4', fourcc, 49.0, (int(cap.get(3)), int(cap.get(4))))

# Loop through the video frames
while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=True)

        # Get the boxes and track IDs
        boxes = results[0].boxes.xywh.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()

        # Visualize the results on the frame
        annotated_frame = results[0].plot()

        # Plot the tracks
        for box, track_id in zip(boxes, track_ids):
            x, y, w, h = box
            track = track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 90 tracks for 90 frames
                track.pop(0)

            # Draw the tracking lines
            points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(annotated_frame, [points], isClosed=False, color=(230, 230, 230), thickness=1)

        # write the frame to the video
        out.write(annotated_frame)
        # Display the annotated frame
        cv2.imshow("YOLOv11 Tracking", annotated_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # Break the loop if the end of the video is reached
        break

# Release the video capture object and close the display window
cap.release()
out.release()
cv2.destroyAllWindows()