import cv2
import os
import json
import numpy as np
from some_inference_model import run_inference  # hypothetical inference model
from utils import save_annotations, save_video  # hypothetical utility functions

# Function to manually select points in a frame
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((x, y))  # Add the clicked point to the points list
        print(f"Point selected: ({x}, {y})")

# Function to run object detection (assumes YOLOv5 or similar model)
def detect_points_using_object_detection(frame):
    # For example, assume you use a pre-trained YOLOv5 model
    model = YOLOv5("yolov5s.pt")  # Replace with your model
    results = model(frame)

    # Example: Look for known objects (e.g., face-off dots or net corners)
    points = []
    for result in results.xywh[0]:
        class_id = int(result[5])  # Class ID of the object
        if class_id == 0:  # Assume 0 is the class ID for face-off dots
            x, y, w, h = result[:4]
            points.append((x, y))
        elif class_id == 1:  # Assume 1 is the class ID for net corners
            x, y, w, h = result[:4]
            points.append((x, y))
    return points

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
    
    # 2) Specify video relative location to the rink (using homography)
    print("Manually selecting points or using object detection for reference points...")

    # Choose one of the following methods to extract points:
    # Method 1: Manually select points (click on the face-off dots and net corners in a frame)
    selected_points = []
    frame_for_selecting_points = cv2.imread(os.path.join(output_dir, "frame_0.jpg"))
    cv2.imshow("Select Points", frame_for_selecting_points)
    cv2.setMouseCallback("Select Points", select_point, param=selected_points)
    print("Click on the reference points (e.g., face-off dots, net corners).")
    cv2.waitKey(0)  # Wait until you click points
    cv2.destroyAllWindows()

    # Method 2: Use object detection to detect points in the frame
    # selected_points = detect_points_using_object_detection(frame_for_selecting_points)

    print("Selected points (pixel coordinates):", selected_points)

    # Now, set the rink coordinates for these points. Replace these with actual rink coordinates
    rink_points = np.array([
        [170 22],  # Defensive zone face-off dot 1 (known rink coordinate)
        [170, 63],  # Defensive zone face-off dot 2 (known rink coordinate)
        [190, 39.5],  # Top left corner of the net (known rink coordinate)
        [190, 45.5]   # Top right corner of the net (known rink coordinate)
    ], dtype='float32')

    # Define the corresponding points in the video frame (in pixel coordinates)
    video_points = np.array(selected_points, dtype='float32')

    # Calculate the homography matrix
    H, _ = cv2.findHomography(video_points, rink_points)

    print("Homography Matrix:")
    print(H)

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
        predicted_location = predict_location_on_rink(bbox, H)  # Using homography to get rink coordinates
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

def predict_location_on_rink(bbox, H):
    """
    Example function that predicts where a bounding box might be located on the rink.
    It uses the homography transformation to map the bbox from the video frame to the rink coordinates.
    """
    x, y, w, h = bbox
    frame_point = (x + w // 2, y + h // 2)  # Taking the center of the bbox
    
    # Apply the homography transformation
    rink_point = apply_homography_to_point(frame_point, H)
    
    # Return the predicted position on the rink
    return (rink_point[0] - w // 2, rink_point[1] - h // 2, w, h)  # Adjust back to bbox location

def apply_homography_to_point(frame_point, H):
    """
    Applies the homography matrix to a point in the video frame and returns the corresponding point on the rink.
    """
    frame_point_homogeneous = np.array([frame_point[0], frame_point[1], 1], dtype='float32')
    rink_point_homogeneous = np.dot(H, frame_point_homogeneous)
    
    # Normalize the result to get the actual rink coordinates
    rink_point = rink_point_homogeneous[:2] / rink_point_homogeneous[2]
    return rink_point

if __name__ == '__main__':
    main()
