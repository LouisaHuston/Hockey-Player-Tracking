import torch
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection, DetrImageProcessor
from src.coco_dataset import COCODataset, collate_fn

def evaluate_model(test_dataloader, model, device):
    model.eval()
    test_running_loss = 0.0
    total_batches = len(test_dataloader)

    with torch.no_grad():
        for batch_idx, (pixel_values, target) in enumerate(test_dataloader):
            pixel_values = pixel_values.to(device)
            target = [{k: v.to(device) for k, v in t.items()} for t in target]

            outputs = model(pixel_values=pixel_values, labels=target)
            loss = outputs.loss
            test_running_loss += loss.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"Batch {batch_idx + 1}/{total_batches}, Test Loss: {loss.item():.4f}")

    avg_test_loss = test_running_loss / total_batches
    print(f"Average Test Loss: {avg_test_loss:.4f}")

def main():
    # Paths to dataset and annotations
    test_annotations = 'data/annotations/test_annotations.json'
    img_dir = 'data/images'
    model_weights_path = 'weights/model_epoch_10.pth'  # Change this to the desired epoch checkpoint

    # Load the pretrained DETR model and processor
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50")
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

    # Load the test dataset
    test_dataset = COCODataset(test_annotations, img_dir, processor)
    test_dataloader = DataLoader(test_dataset, batch_size=7, shuffle=False, collate_fn=collate_fn)

    # Load saved model weights
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)

    # Run evaluation
    evaluate_model(test_dataloader, model, device)

if __name__ == "__main__":
    main()
