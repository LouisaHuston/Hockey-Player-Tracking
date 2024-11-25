# main.py

from src.download_data import download_dataset
from src.process import (
    find_videos_and_extract_frames,
    list_frames,
    save_frames_to_file,
    extract_annotations,
    create_coco_annotations,
    overlay_boxes,
    create_video_from_images,
    generate_heatmap,
)

import logging
import json
import os

# Configure logging to use green color
logging.basicConfig(
    level=logging.INFO,
    format="\033[92m%(message)s\033[0m"  # Green color
)

def main():
    # Step 1: Download the dataset
    dataset_dir = 'hockeyTrackingDataset'
    download_dataset(dataset_dir)
    logging.info("Dataset downloaded successfully.")
    
    # Step 2: Extract frames from videos
    root_folder = f"{dataset_dir}/clips"
    find_videos_and_extract_frames(root_folder)
    logging.info("Frames extracted from videos.")
    
    # Step 3: List frames and save to file
    frames = list_frames('data/images')
    output_file = 'frame_list.txt'
    save_frames_to_file(frames, output_file)
    logging.info(f"Successfully saved {len(frames)} frame paths to {output_file}")
    
    # Step 4: Extract annotations
    test_root_folder = f"{dataset_dir}/MOT_Challenge_Sytle_Label/test/"
    train_root_folder = f"{dataset_dir}/MOT_Challenge_Sytle_Label/train/"
    test_annotations = extract_annotations(test_root_folder)
    train_annotations = extract_annotations(train_root_folder)
    annotations_array = test_annotations + train_annotations
    logging.info("Annotations extracted.")
    
    # Step 5: Create COCO annotations
    coco_data = create_coco_annotations(annotations_array, "data/images/")
    
    # Save the COCO annotations to a JSON file
    os.makedirs('data/annotations/', exist_ok=True)
    with open('data/annotations/coco_annotations.json', 'w') as f:
        json.dump(coco_data, f, indent=4, sort_keys=True)
    logging.info("COCO JSON generated and saved as 'coco_annotations.json'.")
    
    # Step 6: Overlay bounding boxes on images
    overlay_boxes('data/annotations/coco_annotations.json', 'data/images', 'data/overlays/')
    logging.info(f"All images with overlays saved in 'data/overlays/'")
    
    # Step 7: Create video from images
    images_subfolder = os.path.join('data/overlays/', 'allstar_2019', '001')  # Adjust the path as needed
    output_video_path = 'data/videos/output_video.mp4'
    os.makedirs('data/videos/', exist_ok=True)
    frame_rate = 30  # Adjust as needed
    create_video_from_images(images_subfolder, output_video_path, frame_rate)
    logging.info(f"Video saved to {output_video_path}")
    
    # Step 8: Generate heatmap
    generate_heatmap('data/annotations/coco_annotations.json')
    logging.info("Heatmap generated.")

if __name__ == "__main__":
    main()
