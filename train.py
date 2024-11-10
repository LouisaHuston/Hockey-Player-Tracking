# main.py

import os
import json

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

def main():
    # Step 1: Download the dataset
    dataset_dir = download_dataset()

    # Change current working directory to dataset_dir
    os.chdir(dataset_dir)

    # Step 2: Extract frames from videos
    root_folder = "clips"
    find_videos_and_extract_frames(root_folder)

    # Step 3: List frames and save to file
    root_dir = 'images'
    frames = list_frames(root_dir)
    output_file = 'frame_list.txt'
    save_frames_to_file(frames, output_file)
    print(f"Successfully saved {len(frames)} frame paths to {output_file}")

    # Step 4: Extract annotations
    test_root_folder = 'MOT_Challenge_Sytle_Label/test/'
    train_root_folder = 'MOT_Challenge_Sytle_Label/train/'
    test_annotations = extract_annotations(test_root_folder)
    train_annotations = extract_annotations(train_root_folder)
    annotations_array = test_annotations + train_annotations

    # Step 5: Create COCO annotations
    images_folder = 'images/'
    coco_data = create_coco_annotations(annotations_array, images_folder)

    # Save the COCO annotations to a JSON file
    with open('coco_annotations.json', 'w') as f:
        json.dump(coco_data, f)
    print("COCO JSON generated and saved as 'coco_annotations.json'.")

    # Step 6: Overlay bounding boxes on images
    output_folder = 'overlays/'
    overlay_boxes('coco_annotations.json', images_folder, output_folder)
    print(f"All images with overlays saved in {output_folder}")

    # # Step 7: Create video from images
    # images_subfolder = os.path.join(output_folder, 'allstar_2019', '001')  # Adjust the path as needed
    # output_video_path = 'output_video.mp4'
    # frame_rate = 30  # Adjust as needed
    # create_video_from_images(images_subfolder, output_video_path, frame_rate)
    # print(f"Video saved to {output_video_path}")

    # # Step 8: Generate heatmap
    # generate_heatmap('coco_annotations.json')

    # Step 9: Start the Training Process
    import torch
    from model import setup_model
    
    def train_model(data_dir, num_epochs=10):
        model, train_loader, device = setup_model(data_dir)
    
        for epoch in range(num_epochs):
            model.train()  # Set model to training mode
            print(f'Starting epoch {epoch + 1}/{num_epochs}')
    
            for images, _ in train_loader:
                images = images.to(device)
                outputs = model(images)  # Forward pass
    
        # Save the trained model
        torch.save(model.state_dict(), "co_detr_model.pth")

if __name__ == "__main__":
    main()
