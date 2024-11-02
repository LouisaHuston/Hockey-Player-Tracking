# src/process.py
import os
import cv2
import json
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def extract_frames_and_annotations(dataset_path, images_dir, annotations_dir):
    """
    Extracts frames from videos and annotations, and organizes them into train and test sets.
    """
    videos_path = os.path.join(dataset_path, 'clips')
    annotations_path = os.path.join(dataset_path, 'MOT_Challenge_Sytle_Label')

    # Define train and test directories
    train_images_dir = os.path.join(images_dir, 'train')
    test_images_dir = os.path.join(images_dir, 'test')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(test_images_dir, exist_ok=True)

    # Extract frames from videos
    extract_frames(videos_path, train_images_dir, test_images_dir)

    # Extract annotations
    extract_annotations(annotations_path, annotations_dir)

def extract_frames(videos_path, train_images_dir, test_images_dir):
    """
    Extracts frames from video files and saves them into train and test directories.
    """
    for split in ['train', 'test']:
        split_videos_path = os.path.join(videos_path, split)
        split_images_dir = os.path.join(train_images_dir if split == 'train' else test_images_dir)

        for root, _, files in os.walk(split_videos_path):
            for file in files:
                if file.endswith('.mp4'):
                    video_path = os.path.join(root, file)
                    output_folder = os.path.join(split_images_dir, os.path.splitext(file)[0])
                    os.makedirs(output_folder, exist_ok=True)
                    extract_frames_from_video(video_path, output_folder)

def extract_frames_from_video(video_path, output_folder):
    """
    Extracts frames from a single video file.
    """
    cap = cv2.VideoCapture(video_path)
    frame_number = 0

    with tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), desc=f"Extracting frames from {os.path.basename(video_path)}", unit="frame") as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_filename = f"{frame_number}.jpg"
            cv2.imwrite(os.path.join(output_folder, frame_filename), frame)
            frame_number += 1
            pbar.update(1)
    cap.release()

def extract_annotations(annotations_path, annotations_dir):
    """
    Extracts annotations and saves them into train.json and test.json in COCO format.
    """
    for split in ['train', 'test']:
        split_annotations_path = os.path.join(annotations_path, split)
        annotations = parse_annotations(split_annotations_path)
        coco_data = convert_to_coco_format(annotations, split)
        json_path = os.path.join(annotations_dir, f"{split}.json")
        with open(json_path, 'w') as f:
            json.dump(coco_data, f)

def parse_annotations(annotations_path):
    """
    Parses annotation text files into a list of annotation dictionaries.
    """
    annotations = []
    for root, _, files in os.walk(annotations_path):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        elements = line.strip().split(',')
                        if len(elements) == 9:
                            frame_number, player_id, x1, y1, w, h, confidence, tracklet_id, visibility = elements
                            annotation = {
                                "frame_number": int(frame_number),
                                "player_id": int(player_id),
                                "bbox": [float(x1), float(y1), float(w), float(h)],
                                "confidence": float(confidence),
                                "tracklet_id": int(tracklet_id),
                                "visibility": float(visibility),
                                "file_path": file_path
                            }
                            annotations.append(annotation)
    return annotations

def convert_to_coco_format(annotations, split):
    """
    Converts annotations to COCO format.
    """
    images = []
    annotations_coco = []
    categories = []
    category_map = {}
    image_map = {}
    annotation_id = 1
    image_id = 1

    for annotation in tqdm(annotations, desc=f"Converting {split} annotations to COCO format", unit="annotation"):
        file_name = construct_image_filename(annotation['file_path'], annotation['frame_number'], split)
        image_full_path = os.path.join('data', 'images', split, file_name)
        if not os.path.exists(image_full_path):
            continue

        if file_name not in image_map:
            image_map[file_name] = image_id
            image_entry = {
                "id": image_id,
                "file_name": file_name,
                "width": 1920,
                "height": 1080
            }
            images.append(image_entry)
            image_id += 1

        player_id = annotation["player_id"]
        if player_id not in category_map:
            category_map[player_id] = len(category_map) + 1
            category_entry = {
                "id": category_map[player_id],
                "name": f"player_{player_id}"
            }
            categories.append(category_entry)

        coco_annotation = {
            "id": annotation_id,
            "image_id": image_map[file_name],
            "category_id": category_map[player_id],
            "bbox": annotation["bbox"],
            "score": annotation["confidence"],
            "tracklet_id": annotation["tracklet_id"],
            "visibility": annotation["visibility"],
            "iscrowd": 0,
            "area": annotation["bbox"][2] * annotation["bbox"][3]
        }
        annotations_coco.append(coco_annotation)
        annotation_id += 1

    coco_data = {
        "images": images,
        "annotations": annotations_coco,
        "categories": categories
    }
    return coco_data

def construct_image_filename(file_path, frame_number, split):
    """
    Constructs the image filename based on the annotation file path and frame number.
    """
    base_name = os.path.basename(file_path).replace('.txt', '')
    return os.path.join(base_name, f"{frame_number}.jpg")

def create_coco_annotations(images_dir, annotations_dir):
    """
    Placeholder function if additional processing is needed.
    """
    pass  # Processing is done during extraction

def overlay_boxes_on_images(images_dir, annotations_dir):
    """
    Overlays bounding boxes on images and saves them.
    """
    for split in ['train', 'test']:
        images_split_dir = os.path.join(images_dir, split)
        annotations_path = os.path.join(annotations_dir, f"{split}.json")
        output_dir = os.path.join('overlays', split)
        os.makedirs(output_dir, exist_ok=True)

        with open(annotations_path, 'r') as f:
            coco_data = json.load(f)

        annotations_by_image = {}
        for annotation in coco_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_by_image:
                annotations_by_image[image_id] = []
            annotations_by_image[image_id].append(annotation)

        image_id_to_filename = {image['id']: image['file_name'] for image in coco_data['images']}

        for image_id, annotations in tqdm(annotations_by_image.items(), desc=f"Overlaying boxes on {split} images"):
            file_name = image_id_to_filename[image_id]
            image_path = os.path.join(images_split_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            draw_bounding_boxes(image_path, annotations, output_path)

def draw_bounding_boxes(image_path, annotations, output_path):
    """
    Draws bounding boxes on a single image.
    """
    image = cv2.imread(image_path)
    if image is None:
        return
    for annotation in annotations:
        x, y, w, h = annotation['bbox']
        top_left = (int(x), int(y))
        bottom_right = (int(x + w), int(y + h))
        cv2.rectangle(image, top_left, bottom_right, color=(0, 255, 0), thickness=2)
        category_id = annotation["category_id"]
        cv2.putText(image, f"ID {category_id}", (int(x), int(y - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.imwrite(output_path, image)

def create_heatmap(annotations_dir):
    """
    Creates a heatmap from the bounding boxes in the annotations.
    """
    # Implement heatmap creation as needed
    pass

def create_video_from_images(images_dir):
    """
    Creates videos from images.
    """
    # Implement video creation as needed
    pass
