# src/process.py

import os
import cv2
import json
import random
from tqdm import tqdm
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt

def extract_frames(video_path, output_folder):
    """Extract frames from a video and save them as images."""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_number = 1  # Start from 1 to match frame numbers in annotations

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Add a progress bar
    with tqdm(total=frame_count, desc=f"Extracting frames from {os.path.basename(video_path)}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Save frame as JPG file
            frame_filename = os.path.join(output_folder, f"{frame_number}.jpg")
            cv2.imwrite(frame_filename, frame)
            frame_number += 1

            # Update the progress bar
            pbar.update(1)

    cap.release()

    print(f"Completed extracting frames from {os.path.basename(video_path)} into {output_folder}")

def find_videos_and_extract_frames(root_folder, output_folder):
    """Find all .mp4 files in the root_folder and extract their frames into output_folder."""
    video_files = []

    # Collect all video file paths
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(dirpath, file))

    # Process each video file
    for video_path in video_files:
        # Generate output path based on the video path
        relative_path = os.path.relpath(os.path.dirname(video_path), root_folder)
        output_video_folder = os.path.join(output_folder, relative_path)

        # Extract frames
        extract_frames(video_path, output_video_folder)

def extract_annotations(root_folder):
    """Find all .txt files in the root_folder and extract annotations."""
    gt_files = []
    # Collect all text file paths
    for dirpath, _, filenames in os.walk(root_folder):
        for file in filenames:
            if file.endswith(".txt"):
                gt_files.append(os.path.join(dirpath, file))
    annotations_array = []
    # Process each text file with tqdm progress bar
    for gt_path in tqdm(gt_files, desc="Processing text files", unit="file"):
        with open(gt_path, 'r') as file:
            lines = file.readlines()
            for line in lines:
                frame_number, player_id, x1, y1, w, h, confidence, tracklet_id, visibility = line.strip().split(',')
                annotation = {
                    "frame_number": int(frame_number),
                    "player_id": int(player_id),
                    "bbox": [float(x1), float(y1), float(w), float(h)],
                    "confidence": float(confidence),
                    "tracklet_id": int(tracklet_id),
                    "visibility": float(visibility),
                    "file_path": gt_path  # Store the file path for later processing
                }
                annotations_array.append(annotation)
    return annotations_array

def process_filename(file_path, frame_number):
    """Convert the file path to the required format."""
    parts = file_path.split(os.sep)

    # Extract relevant parts from the path
    video_sequence = parts[-2]  # e.g., '001'
    video_folder = parts[-3]    # e.g., 'CAR_VS_BOS_2019'

    # Return the desired filename structure
    return os.path.join(video_folder, video_sequence, f"{frame_number}.jpg")

def create_coco_annotations(annotations, images_folder):
    """Generate COCO-style annotations, including only images that exist."""
    images = []
    coco_annotations = []
    categories = []
    category_map = {}
    image_map = {}
    annotation_id = 1
    image_id = 1

    # Process each annotation
    for annotation in tqdm(annotations, desc="Generating COCO annotations", unit="annotation"):
        file_path = process_filename(annotation['file_path'], annotation['frame_number'])

        # Check if the corresponding image exists in the images folder
        image_full_path = os.path.join(images_folder, file_path)
        if not os.path.exists(image_full_path):
            # Skip if the image does not exist
            continue

        # Create the image entry if not already added
        if file_path not in image_map:
            image_map[file_path] = image_id
            image_entry = {
                "id": image_id,
                "file_name": file_path,
                "width": 1920,  # Adjust as needed
                "height": 1080  # Adjust as needed
            }
            images.append(image_entry)
            image_id += 1

        # Create the category if not already added
        player_id = annotation["player_id"]
        if player_id not in category_map:
            category_map[player_id] = len(category_map) + 1
            category_entry = {
                "id": category_map[player_id],
                "name": f"player_{player_id}"
            }
            categories.append(category_entry)

        # Create the annotation
        coco_annotation = {
            "id": annotation_id,
            "image_id": image_map[file_path],
            "category_id": category_map[player_id],
            "bbox": annotation["bbox"],
            "score": annotation["confidence"],
            "tracklet_id": annotation["tracklet_id"],
            "visibility": annotation["visibility"],
            "iscrowd": 0
        }
        coco_annotations.append(coco_annotation)
        annotation_id += 1

    # Construct the final COCO JSON
    coco_data = {
        "images": images,
        "annotations": coco_annotations,
        "categories": categories
    }

    return coco_data

def draw_bounding_boxes(image_path, annotations, output_path):
    """Draw bounding boxes on an image and save the result."""
    # Load the image
    image = cv2.imread(image_path)

    if image is None:
        print(f"Error loading image: {image_path}")
        return

    # Draw each bounding box
    for annotation in annotations:
        # Extract bounding box coordinates
        x, y, w, h = annotation['bbox']

        # Calculate top-left and bottom-right coordinates
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))

        # Draw the bounding box
        cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)

        # Optionally, add the player ID as text near the box
        player_id = annotation["category_id"]
        cv2.putText(image, f"Player {player_id}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)

    # Save the image with bounding boxes
    cv2.imwrite(output_path, image)

def split_dataset(coco_data, train_ratio=0.8):
    """Split the dataset into training and testing sets."""
    # Get all image IDs
    image_ids = [image['id'] for image in coco_data['images']]

    # Shuffle image IDs
    random.shuffle(image_ids)

    # Split image IDs into train and test
    num_train = int(len(image_ids) * train_ratio)
    train_image_ids = set(image_ids[:num_train])
    test_image_ids = set(image_ids[num_train:])

    # Split images
    train_images = [image for image in coco_data['images'] if image['id'] in train_image_ids]
    test_images = [image for image in coco_data['images'] if image['id'] in test_image_ids]

    # Split annotations
    train_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in train_image_ids]
    test_annotations = [ann for ann in coco_data['annotations'] if ann['image_id'] in test_image_ids]

    # Prepare train and test data
    train_data = {
        "images": train_images,
        "annotations": train_annotations,
        "categories": coco_data['categories']
    }

    test_data = {
        "images": test_images,
        "annotations": test_annotations,
        "categories": coco_data['categories']
    }

    return train_data, test_data


def overlay_boxes(coco_json_path, images_folder, output_folder):
    """Overlay bounding boxes on images based on COCO annotations and save the results."""
    # Create output directory if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the COCO JSON file
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # Group annotations by image_id
    annotations_by_image = {}
    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(annotation)

    # Process each image and overlay bounding boxes
    for image_info in tqdm(coco_data['images'], desc="Overlaying boxes on images"):
        image_file_name = image_info['file_name']
        image_path = os.path.join(images_folder, image_file_name)

        # Prepare output path in the overlays/ folder
        output_image_path = os.path.join(output_folder, image_file_name)
        output_image_dir = os.path.dirname(output_image_path)

        # Create subdirectories if they do not exist
        if not os.path.exists(output_image_dir):
            os.makedirs(output_image_dir)

        # Get the annotations for this image
        image_id = image_info['id']
        if image_id in annotations_by_image:
            annotations = annotations_by_image[image_id]
            draw_bounding_boxes(image_path, annotations, output_image_path)

def create_video_from_images(images_folder, output_video_path, frame_rate):
    """Create a video file from images in a specified folder."""
    # Get list of images from the folder
    image_files = [f for f in os.listdir(images_folder) if f.endswith('.jpg')]

    if not image_files:
        print("No images found in the specified folder.")
        return

    # Sort images to ensure correct order
    image_files.sort(key=lambda x: int(os.path.splitext(x)[0]))  # Assuming images are named as numbers

    # Read the first image to get the size
    first_image_path = os.path.join(images_folder, image_files[0])
    first_image = cv2.imread(first_image_path)

    if first_image is None:
        print(f"Error loading image: {first_image_path}")
        return

    height, width, _ = first_image.shape

    # Create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 format
    video_writer = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))

    # Write each image to the video
    for image_file in tqdm(image_files, desc="Creating video"):
        image_path = os.path.join(images_folder, image_file)
        frame = cv2.imread(image_path)

        if frame is None:
            print(f"Error loading image: {image_path}")
            continue

        video_writer.write(frame)

    # Release the VideoWriter object
    video_writer.release()

def generate_heatmap(coco_json_path):
    """Generate and display a heatmap based on the bounding boxes in the COCO annotations."""
    # Load the COCO annotations JSON file
    with open(coco_json_path, 'r') as file:
        data = json.load(file)

    # Extract annotations and frame paths
    annotations = data.get('annotations', [])
    images = data.get('images', [])

    # Initialize variables to determine the global heatmap dimensions
    max_height, max_width = 0, 0

    # Create a mapping from image id to file path and dimensions
    image_id_to_info = {}
    for image in images:
        image_id_to_info[image['id']] = (image['file_name'], image['height'], image['width'])
        max_height = max(max_height, image['height'])
        max_width = max(max_width, image['width'])

    # Initialize a global heatmap to accumulate bounding boxes
    heatmap = np.zeros((max_height, max_width), dtype=np.float32)

    # Iterate through annotations and update the heatmap
    for annotation in annotations:
        image_id = annotation['image_id']
        bbox = annotation['bbox']

        # Get the file path and dimensions of the image
        image_info = image_id_to_info.get(image_id)
        if not image_info:
            continue
        _, height, width = image_info

        # Ensure the bounding box coordinates are within the image dimensions
        x, y, w, h = map(int, bbox)
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        w = min(w, width - x)
        h = min(h, height - y)

        # Accumulate the bounding box into the heatmap
        heatmap[y:y+h, x:x+w] += 1

    # Normalize the heatmap for better visualization
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)

    # Convert heatmap to an appropriate type for visualization
    heatmap = heatmap.astype(np.float32)

    # Display the heatmap using Matplotlib
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title('Bounding Box Heatmap')
    plt.show()