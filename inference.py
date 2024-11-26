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

# Class to handle model loading and inference
class ModelHandler:
    def __init__(self, model_path):
        self.model = self.load_trained_model(model_path)
        self.processor = DetrImageProcessor.from_pretrained(model_path)
        self.model.eval()  # Set the model to evaluation mode

    def load_trained_model(self, model_path):
        model = DetrForObjectDetection.from_pretrained(model_path)
        return model

    def run_inference(self, frame):
        inputs = self.processor(images=frame, return_tensors="pt")

        with torch.no_grad():
            outputs = self.model(**inputs)

        target_sizes = torch.tensor([frame.shape[0], frame.shape[1]])
        results = self.processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=0.9
        )[0]

        boxes = results["boxes"].cpu().numpy()
        labels = results["labels"].cpu().numpy()
        scores = results["scores"].cpu().numpy()

        return boxes, labels, scores

# Function to save annotations in COCO format
def save_annotations(annotations, output_path):
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "player", "supercategory": "sports"}],
    }

    for idx, annotation in enumerate(annotations):
        image_id = annotation["image_id"]
        bbox = annotation["bbox"]
        category_id = annotation["category_id"]
        score = annotation["score"]

        # Check if this image has been added to coco_format['images']
        if not any(image["id"] == image_id for image in coco_format["images"]):
            coco_format["images"].append(
                {
                    "id": image_id,
                    "file_name": f"frame_{image_id}.jpg",  # Assuming frame names are sequential like "frame_0.jpg"
                    "height": 720,
                    "width": 1280,
                }
            )

        coco_format["annotations"].append(
            {
                "id": idx + 1,  # Incremental ID for each annotation
                "image_id": image_id,
                "category_id": category_id,
                "bbox": bbox,
                "score": score,
                "area": bbox[2] * bbox[3],  # Width * Height
                "iscrowd": 0,
            }
        )

    # Save the annotations to the JSON file
    with open(output_path, "w") as f:
        json.dump(coco_format, f)
    print(f"Annotations saved to {output_path}")

# Function to apply homography to a point
def apply_homography_to_point(point, H):
    point_homogeneous = np.array([point[0], point[1], 1])
    transformed_point = np.dot(H, point_homogeneous)
    x_rink = transformed_point[0] / transformed_point[2]
    y_rink = transformed_point[1] / transformed_point[2]
    return (x_rink, y_rink)

# Function to draw the predicted locations on the rink image
def draw_on_rink(predicted_location, rink_image, rink_width, rink_height):
    # Scale the location based on the desired rink image dimensions
    scaled_x = int(predicted_location[0] * rink_width)
    scaled_y = int(predicted_location[1] * rink_height)
    
    # Draw a circle (black) on the rink image
    cv2.circle(rink_image, (scaled_x, scaled_y), 5, (0, 0, 0), -1)
    return rink_image

def main():
    # 1) Extract frames from a video path you specify
    video_path = "assets/IMG-2113.mov"
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

    rink_points = np.array(
        [
            [170, 22],  # Replace with actual rink coordinates
            [170, 63],
            [190, 39.5],
            [190, 45.5],
        ],
        dtype="float32",
    )

    video_points = np.array(selected_points, dtype="float32")
    H, _ = cv2.findHomography(video_points, rink_points)
    print("Homography Matrix:", H)

    # 3) Run inference on frames and save annotations
    model_path = "models/trained_model.pth"
    model_handler = ModelHandler(model_path)

    inference_results = []
    for frame_index in range(frame_count):
        frame_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
        frame = cv2.imread(frame_path)
        boxes, labels, scores = model_handler.run_inference(frame)

        for i in range(len(boxes)):
            result = {
                "image_id": frame_index,
                "bbox": boxes[i].tolist(),
                "category_id": labels[i].item(),
                "score": scores[i].item(),
            }
            inference_results.append(result)

    annotation_path = "data/annotations/inference.json"
    os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
    save_annotations(inference_results, annotation_path)

    # 4) Use data/annotations/inference.json to predict locations for each of the bboxes for each of the frames
    with open(annotation_path, "r") as f:
        annotations = json.load(f)

    predicted_locations = {}
    for annotation in annotations["annotations"]:
        frame_id = annotation["image_id"]
        bbox = annotation["bbox"]
        predicted_location = apply_homography_to_point(
            (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3]), H  # Bottom center of the bbox
        )
        predicted_locations[frame_id] = predicted_locations.get(frame_id, []) + [
            predicted_location
        ]

    # 5) Annotate frames with predicted locations and save
    rink_image_path = "assets/rink_image.jpg"  # Path to rink image
    rink_image = cv2.imread(rink_image_path)
    rink_width, rink_height = 397, 200
    
    output_frame_dir = "data/annotated_frames"
    os.makedirs(output_frame_dir, exist_ok=True)

    # 6) Video for regular frames with predicted bounding boxes
    video_output_path_with_bboxes = "data/output_with_bboxes.mp4"
    first_frame = cv2.imread(os.path.join(output_dir, f"frame_0.jpg"))
    height, width, _ = first_frame.shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer_with_bboxes = cv2.VideoWriter(video_output_path_with_bboxes, fourcc, 30.0, (width, height))

    # 7) Video for blank rink images with black circles
    video_output_path_with_rink = "data/output_with_rink.mp4"
    video_writer_with_rink = cv2.VideoWriter(video_output_path_with_rink, fourcc, 30.0, (rink_width, rink_height))

    for frame_index in range(frame_count):
        # 7.1) Frame with predicted bounding boxes
        frame_path = os.path.join(output_dir, f"frame_{frame_index}.jpg")
        frame = cv2.imread(frame_path)
        boxes, _, _ = model_handler.run_inference(frame)
        for bbox in boxes:
            # Draw the bounding boxes on the frame
            x_min, y_min, x_max, y_max = map(int, bbox)
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        video_writer_with_bboxes.write(frame)

        # 7.2) Blank rink image with predicted locations
        rink_frame = rink_image.copy()
        for location in predicted_locations.get(frame_index, []):
            rink_frame = draw_on_rink(location, rink_frame, rink_width, rink_height)
        video_writer_with_rink.write(rink_frame)

    video_writer_with_bboxes.release()
    video_writer_with_rink.release()
    print(f"Saved the video with bounding boxes to {video_output_path_with_bboxes}")
    print(f"Saved the rink video with circles to {video_output_path_with_rink}")

if __name__ == "__main__":
    main()
