
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.coco_dataset import COCODataset, collate_fn
from src.detr import train_model
from src.split import split_coco_json

import torch
import json
import os

def main():
    # Paths to dataset and annotations
    annotations_path = 'data/annotations/coco_annotations.json'
    train_annotations = 'train_annotations.json'
    test_annotations = 'test_annotations.json'
    annotations_output_dir = 'data/annotations'
    img_dir = 'data/images'

    # Ensure required directories exist
    os.makedirs(annotations_output_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    # Split the dataset into train and test sets
    split_coco_json(
        annotations_path=annotations_path,
        output_dir=annotations_output_dir,
        train_file_name=train_annotations,
        test_file_name=test_annotations
    )

    # Load the pretrained DETR model and processor
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Create dataset instances
    train_dataset = COCODataset(f'data/annotations/{train_annotations}', img_dir, processor, max_images=800)
    test_dataset = COCODataset(f'data/annotations/{test_annotations}', img_dir, processor, max_images=200)

    # Save the test annotations
    output_path = 'test.json'
    with open(output_path, "w") as json_file:
        json.dump(test_dataset.coco, json_file, indent=4)

    
    # Use this custom collate function with DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=7, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=7, shuffle=False, collate_fn=collate_fn)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start training
    train_model(train_dataloader, test_dataloader, model, optimizer, device)

    # save model
    torch.save(model.state_dict(), "models/trained_model.pth")

if __name__ == "__main__":
    main()
