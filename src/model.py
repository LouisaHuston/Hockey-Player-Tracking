# src/model.py

import os
import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor

def get_data_loader(images_dir, annotations_path, batch_size=4):
    """
    Create a data loader for the dataset with the specified batch size.
    Args:
        images_dir (str): Path to the directory containing images.
        annotations_path (str): Path to the JSON file containing COCO annotations.
        batch_size (int): Batch size for data loading.
    """
    # Initialize the image processor for DETR
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    
    # Create the custom dataset (make sure COCODataset is defined correctly for DETR)
    dataset = COCODataset(images_dir, annotations_path, processor=processor)
    
    # Create and return the DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=dataset.collate_fn)

def setup_model(data_dir):
    """
    Setup the model, data loaders, and device for training.
    Args:
        data_dir (str): Path to the data directory containing images and annotations.
    """
    # Define the paths for the train and test images and annotations
    train_images_dir = os.path.join(data_dir, "images/train")
    test_images_dir = os.path.join(data_dir, "images/test")
    train_annotations_path = os.path.join(data_dir, "annotations/train.json")
    test_annotations_path = os.path.join(data_dir, "annotations/test.json")

    # Setup the data loaders
    train_loader = get_data_loader(train_images_dir, train_annotations_path)
    test_loader = get_data_loader(test_images_dir, test_annotations_path)

    # Load the pre-trained DETR model
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=91)  # Adjust `num_labels` if needed

    # Setup device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, train_loader, test_loader, device
