import cv2
import os
import json
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
import numpy as np

# Function to manually select points in a frame
def select_point(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param.append((x, y))  # Add the clicked point to the points list
        print(f"Point selected: ({x}, {y})")

# Function to load your trained DETR model
def load_trained_model(model_path):
    model = DetrForObjectDetection.from_pretrained(model_path)
    processor = DetrImageProcessor.from_pretrained(model_path)
    model.eval()  # Set the model to evaluation mode
    return model, processor

# Function to run inference on a single frame using DETR
def run_inference(model, processor, frame):
    inputs = processor(images=frame, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([frame.shape[0], frame.shape[1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    
    boxes = results["boxes"].cpu().numpy()
    labels = results["labels"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    
    return boxes, labels, scores

# Function to save annotations in COCO format
def save_annotations(annotations, output_path):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "player", "supercategory": "sports"}]
    }
    
    for idx, annotation in enumerate(annotations):
        image_id = annotation['image_id']
        bbox = annotation['bbox']
        category_id = annotation['category_id']
        score = annotation['score']
        
        # Check if this image has been added to coco_format['images']
        if not any(image['id'] == image_id for image in coco_format['images']):
            coco_format['images'].append({
                "id": image_id,
                "file_name": f"frame_{image_id}.jpg",  # Assuming frame names are sequential like "frame_0.jpg"
                "height": 720,  # Replace with actual height
                "width": 1280  # Replace with actual width
            })
        
        coco_format['annotations'].append({
            "id": idx + 1,  # Incremental ID for each annotation
            "image_id": image_id,
            "category_id": category_id,
            "bbox": bbox,
            "score": score,
            "area": bbox[2] * bbox[3],  # Width * Height
            "iscrowd": 0
        })

    # Save the annotations to the JSON file
    with open(output_path, 'w') as f:
        json.dump(coco_format, f)
    print(f"Annotations saved to {output_path}")

# Function to apply homography transformation to a point
def apply_homography_to_point(point, H):
    point_homogeneous = np.array([point[0], point[1], 1])
    transformed_point = np.dot(H, point_homogeneous)
    x_rink, y_rink = transformed_point[0] / transformed_point[2], transformed_point[1] / transformed_point[2]
    return (x_rink, y_rink)

def main():
    # 1) Extract frames from a video path you specify
    video_path = "path_to_your_video.mp4"
    output_dir = "data/frames"
    os.makedirs(output_dir, exist_ok=True)
    
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
    selected_points = []
    frame_for_selecting_points = cv2.imread(os.path.join(output_dir, "frame_0.jpg"))
    cv2.imshow("Select Points", frame_for_selecting_points)
    cv2.setMouseCallback("Select Points", select_point, param=selected_points)
    print("Click on the reference points (e.g., face-off dots, net corners).")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rink_points = np.array([
        [170, 22],  # Replace with actual rink coordinates
        [170, 63],
        [190, 39.5],
        [190, 45.5]
    ], dtype='float32')

    video_points = np.array(selected_points, dtype='float32')
    H, _ = cv2.findHomography(video_points, rink_points)
    print("Homography Matrix:", H)

    # 3) Run inference on frames and save annotations
    model_path = "models/trained_model.pth"
    model, processor = load_trained_model(model_path)

    inference_results = []
    for frame_index in range(frame_count):
        frame_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
        frame = cv2.imread(frame_path)
        boxes, labels, scores = run_inference(model, processor, frame)
        
        for i in range(len(boxes)):
            result = {
                "image_id": frame_index,
                "bbox": boxes[i].tolist(),
                "category_id": labels[i].item(),
                "score": scores[i].item()
            }
            inference_results.append(result)
    
    annotation_path = "data/annotations/inference.json"
    save_annotations(inference_results, annotation_path)

    # 4) Use data/annotations/inference.json to predict locations for each of the bboxes for each of the frames
    with open(annotation_path, 'r') as f:
        annotations = json.load(f)
    
    predicted_locations = {}
    for annotation in annotations['annotations']:
        frame_id = annotation['image_id']
        bbox = annotation['bbox']
        predicted_location = apply_homography_to_point((bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2), H)
        predicted_locations[frame_id] = predicted_locations.get(frame_id, []) + [predicted_location]

    # 5) Annotate frames with predicted locations and save
    output_frame_dir = "data/annotated_frames"
    os.makedirs(output_frame_dir, exist_ok=True)
    
    for frame_index in range(frame_count):
        frame_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
        frame = cv2.imread(frame_path)
        
        for location in predicted_locations.get(frame_index, []):
            x, y = location
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
        
        annotated_frame_path = os.path.join(output_frame_dir, f"annotated_frame_{frame_index}.jpg")
        cv2.imwrite(annotated_frame_path, frame)

    # 6) Turn all those frames into a movie
    video_output_path = "data/output_movie.mp4"
    frame_files = sorted([f for f in os.listdir(output_frame_dir) if f.endswith(".jpg")])

    first_frame = cv2.imread(os.path.join(output_frame_dir, frame_files[0]))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(video_output_path, fourcc, 30.0, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(os.path.join(output_frame_dir, frame_file))
        video_writer.write(frame)

    video_writer.release()
    print(f"Saved the movie to {video_output_path}")

if __name__ == '__main__':
    main()
