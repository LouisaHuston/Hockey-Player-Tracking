import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.coco_dataset import COCODataset, collate_fn
from src.detr import train_model
from src.split import split_coco_json
import os
import json

def main():
    # Paths to dataset and annotations
    annotations_path = 'data/annotations/coco_annotations.json'
    train_annotations = 'data/annotations/train_annotations.json'
    test_annotations = 'data/annotations/test_annotations.json'
    train_img_dir = 'data/images/train'
    test_img_dir = 'data/images/test'

    # Split the dataset
    split_coco_json(annotations_path, 'data/annotations', 'train_annotations.json', 'test_annotations.json')

    # Load the pretrained DETR model and processor
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Create dataset instances
    train_dataset = COCODataset(train_annotations, train_img_dir, processor)
    test_dataset = COCODataset(test_annotations, test_img_dir, processor)

    # Use this custom collate function with DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=7, shuffle=True, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=7, shuffle=False, collate_fn=collate_fn)

    # Define optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Start training
    train_model(train_dataloader, test_dataloader, model, optimizer, device)