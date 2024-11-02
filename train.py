# main.py
import os
from src.download_data import download_dataset
from src.process import (
    extract_frames_and_annotations,
    create_coco_annotations,
    overlay_boxes_on_images,
    create_heatmap,
    create_video_from_images
)

def main():
    # Step 1: Download the dataset
    dataset_url = 'https://github.com/grant81/hockeyTrackingDataset.git'
    dataset_path = 'hockeyTrackingDataset'
    download_dataset(dataset_url, dataset_path)

    # Step 2: Process the data
    data_dir = 'data'
    images_dir = os.path.join(data_dir, 'images')
    annotations_dir = os.path.join(data_dir, 'annotations')

    # Create directories if they don't exist
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)

    # Extract frames and annotations
    extract_frames_and_annotations(dataset_path, images_dir, annotations_dir)

    # Create COCO annotations
    create_coco_annotations(images_dir, annotations_dir)

    # Optionally, overlay bounding boxes on images
    overlay_boxes_on_images(images_dir, annotations_dir)

    # Optionally, create a heatmap
    create_heatmap(annotations_dir)

    # Optionally, create a video from images
    create_video_from_images(images_dir)

    # Step 9: Start the Training Process

if __name__ == "__main__":
    main()
