import cv2
import os
import json
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import numpy as np
from utils import save_annotations  # hypothetical utility function to save annotations

# Function to manually select points in a frame
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((x, y))  # Add the clicked point to the points list
        print(f"Point selected: ({x}, {y})")

# Function to load your trained DETR model
def load_trained_model(model_path):
    # Load your fine-tuned DETR model and processor
    model = DetrForObjectDetection.from_pretrained(model_path)  # Replace with path to your model
    processor = DetrImageProcessor.from_pretrained(model_path)  # Same path for processor
    model.eval()  # Set the model to evaluation mode
    return model, processor

# Function to run inference on a single frame using DETR
def run_inference(model, processor, frame):
    # Preprocess the frame to be in the format that DETR expects
    inputs = processor(images=frame, return_tensors="pt")
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Post-process the outputs to get bounding boxes, labels, and scores
    target_sizes = torch.tensor([frame.shape[0], frame.shape[1]])  # Use the frame's height and width
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    
    return boxes, labels, scores

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

    print("Selected points (pixel coordinates):", selected_points)

    # Now, set the rink coordinates for these points. Replace these with actual rink coordinates
    rink_points = np.array([
        [0, 0],  # Defensive zone face-off dot 1 (known rink coordinate)
        [0, 1],  # Defensive zone face-off dot 2 (known rink coordinate)
        [0, 2],  # Top left corner of the net (known rink coordinate)
        [0, 3]   # Top right corner of the net (known rink coordinate)
    ], dtype='float32')

    # Define the corresponding points in the video frame (in pixel coordinates)
    video_points = np.array(selected_points, dtype='float32')

    # Calculate the homography matrix
    H, _ = cv2.findHomography(video_points, rink_points)

    print("Homography Matrix:")
    print(H)

    # 3) Run inference on those frames - save result in COCO format to data/annotations/inference.json
    model_path = "models/trained_model.pth"
    model, processor = load_trained_model(model_path)

    inference_results = []
    for frame_index in range(frame_count):
        frame_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
        frame = cv2.imread(frame_path)

        # Run inference to detect objects/people in the frame
        boxes, labels, scores = run_inference(model, processor, frame)  # Use your fine-tuned DETR for inference
        
        # Format results in COCO format
        for i in range(len(boxes)):
            result = {
                "image_id": frame_index,
                "bbox": boxes[i].tolist(),  # Convert to list if needed
                "category_id": labels[i].item(),  # Assuming labels are integers
                "score": scores[i].item()  # Confidence score
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
    return (rink)
