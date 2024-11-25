import cv2
import os
import json
import numpy as np
from some_inference_model import run_inference  # hypothetical inference model
from utils import save_annotations, save_video  # hypothetical utility functions

def main():
    # 1) Extract frames from a video path you specify
    video_path = "path_to_your_video.mp4"  # Replace with your video path
    output_dir = "data/frames"
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video file
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0
    
    while True:
        ret, frame = video_capture.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f"frame_{frame_count}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    video_capture.release()
    print(f"Extracted {frame_count} frames from {video_path}")
    
    # 2) Specify video relative location to the rink (e.g., Coordinates, transformation matrix)
    # Assuming we have some method to define or calculate the rink location on each frame.
    rink_area = (100, 100, 800, 600)  # Placeholder, you may need to compute this based on video or manually
    
    # 3) Run inference on those frames - save result in COCO format to data/annotations/inference.json
    inference_results = []
    for frame_index in range(frame_count):
        frame_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
        frame = cv2.imread(frame_path)

        # Run inference to detect objects/people in the frame
        detections = run_inference(frame)  # Assuming run_inference returns a list of detected bounding boxes
        
        # Format results in COCO format
        for detection in detections:
            result = {
                "image_id": frame_index,
                "bbox": detection['bbox'],  # [x, y, w, h] format
                "category_id": detection['category_id'],
                "score": detection['score']
            }
            inference_results.append(result)
    
    # Save the inference results in COCO format
    annotation_path = "data/annotations/inference.json"
    save_annotations(inference_results, annotation_path)
    print(f"Saved inference results to {annotation_path}")
    
    # 4) Use data/annotations/inference.json to predict locations for each of the bboxes for each of the frames
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    # For each annotation, predict the location (e.g., tracking or position in the rink)
    predicted_locations = {}
    
    for annotation in annotations:
        frame_id = annotation['image_id']
        bbox = annotation['bbox']
        
        # Predict the location of the bounding box on the rink
        predicted_location = predict_location_on_rink(bbox, rink_area)  # Placeholder function
        predicted_locations[frame_id] = predicted_locations.get(frame_id, []) + [predicted_location]

    print("Predicted locations for bounding boxes.")

    # 5) Use those predicted locations to identify where each person is on the rink - make a picture for each frame
    output_frame_dir = "data/annotated_frames"
    os.makedirs(output_frame_dir, exist_ok=True)
    
    for frame_index in range(frame_count):
        frame_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
        frame = cv2.imread(frame_path)
        
        # Overlay predicted locations on the frame
        for location in predicted_locations.get(frame_index, []):
            # Draw bounding boxes or markers on the frame
            x, y, w, h = location
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Save the annotated frame
        annotated_frame_path = os.path.join(output_frame_dir, f"annotated_frame_{frame_index}.jpg")
        cv2.imwrite(annotated_frame_path, frame)

    print(f"Annotated frames saved to {output_frame_dir}")

    # 6) Turn all those frames into a movie
    video_output_path = "data/output_movie.mp4"
    frame_files = sorted([f for f in os.listdir(output_frame_dir) if f.endswith(".jpg")])

    # Set up video writer
    first_frame = cv2.imread(os.path.join(output_frame_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, 30.0, (width, height))

    # Write frames to video
    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(output_frame_dir, frame_file))
        video_writer.write(frame)

    video_writer.release()
    print(f"Saved the movie to {video_output_path}")

def predict_location_on_rink(bbox, rink_area):
    """
    Example function that predicts where a bounding box might be located on the rink.
    You can define your own logic to map detected objects to positions on the rink.
    """
    x, y, w, h = bbox
    rink_x, rink_y, rink_w, rink_h = rink_area
    
    # Transform the bbox coordinates to be relative to the rink (this is a placeholder)
    predicted_x = max(0, min(rink_x + (x % rink_w), rink_x + rink_w))
    predicted_y = max(0, min(rink_y + (y % rink_h), rink_y + rink_h))
    
    return (predicted_x, predicted_y, w, h)

if __name__ == '__main__':
    main()
