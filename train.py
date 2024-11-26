from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.coco_dataset import COCODataset, collate_fn
from src.train_detr import train_model
from src.split import split_coco_json

import torch
import os

def main():
    # Paths to dataset and annotations
    annotations_path = 'data/annotations/coco_annotations.json'
    annotations_output_dir = 'data/annotations'
    train_annotations = os.path.join(annotations_output_dir, 'train_annotations.json')
    test_annotations = os.path.join(annotations_output_dir, 'test_annotations.json')
    img_dir = 'data/images'

    # Ensure required directories exist
    os.makedirs(annotations_output_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Split the dataset into train and test sets
    split_coco_json(
        annotations_path=annotations_path,
        output_dir=annotations_output_dir,
        train_file_name='train_annotations.json',
        test_file_name='test_annotations.json'
    )

    # Load the pretrained DETR model and processor
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Create dataset instances
    train_dataset = COCODataset(train_annotations, img_dir, processor)
    test_dataset = COCODataset(test_annotations, img_dir, processor)

    # Use this custom collate function with DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=7, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=7, shuffle=False, collate_fn=collate_fn)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Start training
    train_model(train_dataloader, test_dataloader, model, optimizer, device)

if __name__ == "__main__":
    main()
