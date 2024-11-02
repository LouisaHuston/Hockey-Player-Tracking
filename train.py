# main.py

import os
import json
import shutil

from src.download_data import download_dataset
from src.process import (
    find_videos_and_extract_frames,
    extract_annotations,
    create_coco_annotations,
    overlay_boxes,
    create_video_from_images,
    generate_heatmap,
    split_dataset
)

def main():
    # Step 1: Download the dataset
    dataset_dir = download_dataset()

    # Change current working directory to dataset_dir
    os.chdir(dataset_dir)

    # Step 2: Extract frames from all videos
    root_videos_folder = 'clips'
    images_output_folder = 'data/images/all'
    find_videos_and_extract_frames(root_videos_folder, images_output_folder)

    # Step 3: Extract all annotations
    root_annotations_folder = 'MOT_Challenge_Sytle_Label'
    all_annotations = extract_annotations(root_annotations_folder)

    # Step 4: Create COCO annotations for all data
    images_folder = images_output_folder
    coco_data = create_coco_annotations(all_annotations, images_folder)

    # Step 5: Perform train/test split
    train_data, test_data = split_dataset(coco_data, train_ratio=0.8)

    # Create directories for train and test images
    train_images_folder = 'data/images/train'
    test_images_folder = 'data/images/test'
    os.makedirs(train_images_folder, exist_ok=True)
    os.makedirs(test_images_folder, exist_ok=True)

    # Step 6: Move images to train and test folders
    for image_info in train_data['images']:
        src_image_path = os.path.join(images_folder, image_info['file_name'])
        dst_image_path = os.path.join(train_images_folder, image_info['file_name'])
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        shutil.move(src_image_path, dst_image_path)

    for image_info in test_data['images']:
        src_image_path = os.path.join(images_folder, image_info['file_name'])
        dst_image_path = os.path.join(test_images_folder, image_info['file_name'])
        os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
        shutil.move(src_image_path, dst_image_path)

    # Step 7: Save the COCO annotations to JSON files
    os.makedirs('data/annotations', exist_ok=True)
    with open('data/annotations/train.json', 'w') as f:
        json.dump(train_data, f)
    print("COCO JSON for train set generated and saved as 'data/annotations/train.json'.")

    with open('data/annotations/test.json', 'w') as f:
        json.dump(test_data, f)
    print("COCO JSON for test set generated and saved as 'data/annotations/test.json'.")

    # Step 6: Overlay bounding boxes on images
    output_folder = 'overlays/'
    overlay_boxes('data/annotations/train.json', train_images_folder, output_folder)
    overlay_boxes('data/annotations/test.json', test_images_folder, output_folder)
    print(f"All images with overlays saved in {output_folder}")

    # Step 7: Create video from images
    output_video_path = 'output_video.mp4'
    frame_rate = 30  # Adjust as needed
    create_video_from_images(output_folder, output_video_path, frame_rate)
    print(f"Video saved to {output_video_path}")

    # Step 9: Start the Training Process

if __name__ == "__main__":
    main()
