import os
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from co_detr import CO_Detr  # Adjust import based on co-DETR implementation
from co_detr.dataset import COCODataset  # Replace with your dataset class if different

def get_data_loader(images_dir, annotations_path, batch_size=4):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # Add more transformations as needed
    ])
    
    dataset = COCODataset(images_dir, annotations_path, transform=transform)  # Adjust dataset class if necessary
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def setup_model(data_dir):
    train_images_dir = os.path.join(data_dir, "images/train")
    test_images_dir = os.path.join(data_dir, "images/test")
    train_annotations_path = os.path.join(data_dir, "annotations/train.json")
    test_annotations_path = os.path.join(data_dir, "annotations/test.json")

    train_loader = get_data_loader(train_images_dir, train_annotations_path)
    test_loader = get_data_loader(test_images_dir, test_annotations_path)

    model = CO_Detr(num_classes=91)  # Adjust based on your number of classes

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, train_loader, test_loader, device
