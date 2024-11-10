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
from src.model import setup_model

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
    import torch.optim as optim
    from tqdm import tqdm  # For progress bars (optional)
    
    def train_model(data_dir, num_epochs=10, learning_rate=1e-5):
        # Set up the model, data loaders, and device
        model, train_loader, test_loader, device = setup_model(data_dir)
        
        # Define the optimizer
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Set model to training mode
        model.train()
        
        for epoch in range(num_epochs):
            print(f"Starting epoch {epoch + 1}/{num_epochs}")
            
            # Initialize loss variables (optional)
            epoch_loss = 0.0
            
            # Iterate over the training data
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                # Get the images and targets (labels and bounding boxes)
                images = batch['pixel_values'].to(device)
                targets = batch['labels'].to(device)
                boxes = batch['boxes'].to(device)
    
                # Create a dictionary for the targets (DETR expects targets in a specific format)
                target = {
                    "labels": targets,
                    "boxes": boxes
                }
    
                # Zero the gradients
                optimizer.zero_grad()
    
                # Forward pass: Compute predicted outputs by passing inputs to the model
                outputs = model(images, labels=target['labels'], pixel_values=images)
    
                # Losses are stored in outputs.logits
                loss = outputs.loss
                epoch_loss += loss.item()
    
                # Backpropagation
                loss.backward()
    
                # Update model parameters
                optimizer.step()
            
            # Print average loss for the epoch
            print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(train_loader)}")
        
        # Save the trained model's state_dict
        torch.save(model.state_dict(), "detr_model.pth")
        print("Model saved to 'detr_model.pth'")



if __name__ == "__main__":
    main()
