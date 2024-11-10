def get_data_loader(images_dir, annotations_path, batch_size=4):
    """
    Create a data loader for the dataset with the specified batch size.
    Args:
        images_dir (str): Path to the directory containing images.
        annotations_path (str): Path to the JSON file containing COCO annotations.
        batch_size (int): Batch size for data loading.
    """
    # Define a transform to preprocess the images (DETR requires specific preprocessing)
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Assuming your COCODataset class returns a dictionary with 'image' and 'annotations'
    dataset = COCODataset(images_dir, annotations_path, processor=processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def setup_model(data_dir):
    """
    Setup the model and dataloaders for training and testing.
    Args:
        data_dir (str): Path to the data directory containing images and annotations.
    """
    import os
    import torch
    from torch.utils.data import DataLoader
    from transformers import DetrForObjectDetection, DetrImageProcessor
    from torchvision import transforms
    from co_detr.dataset import COCODataset  # Replace with your own dataset class if different
    
    # Define the directories for images and annotations
    train_images_dir = os.path.join(data_dir, "images/train")
    test_images_dir = os.path.join(data_dir, "images/test")
    train_annotations_path = os.path.join(data_dir, "annotations/train.json")
    test_annotations_path = os.path.join(data_dir, "annotations/test.json")

    # Create data loaders for training and testing
    train_loader = get_data_loader(train_images_dir, train_annotations_path)
    test_loader = get_data_loader(test_images_dir, test_annotations_path)

    # Load the pre-trained DETR model
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=91)  # Adjust num_labels based on the number of classes

    # Setup device (CUDA or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    return model, train_loader, test_loader, device
