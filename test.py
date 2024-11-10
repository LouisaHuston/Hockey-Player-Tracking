# test.py
import torch
import json
import numpy as np
from src.model import setup_model
from src.dataset import COCODataset  # Make sure the path is correct
from src.evaluation import (calculate_accuracy, calculate_precision, calculate_recall, calculate_f1_score)

# Function to load the trained model
def load_model(model, model_path='detr_model.pth'):
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model

# Function to evaluate the model
def evaluate_model(model, data_loader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            images = batch['pixel_values'].to(device)
            targets = batch['labels'].to(device)
            boxes = batch['boxes'].to(device)

            # Run the model on the input batch
            outputs = model(images)
            
            # Post-process the outputs (we're assuming output format for DETR)
            logits = outputs.logits
            prob = logits.softmax(-1)  # Calculate softmax over the classes
            
            # Convert to Numpy for easier handling in metrics
            predicted_classes = prob.argmax(-1).cpu().numpy()
            true_classes = targets.cpu().numpy()
            
            all_preds.extend(predicted_classes.flatten())  # Flatten the arrays
            all_labels.extend(true_classes.flatten())       # Flatten the arrays
    
    return all_preds, all_labels

# Main function to test the model
def main():
    # Load model and setup
    dataset_dir = 'path_to_your_dataset'  # Replace with actual dataset directory path
    model, train_loader, test_loader, device = setup_model(dataset_dir)
    
    # Load the trained model
    model = load_model(model)
    
    # Evaluate the model on test data
    all_preds, all_labels = evaluate_model(model, test_loader, device)
    
    # Calculate and print evaluation metrics
    accuracy = calculate_accuracy(all_labels, all_preds)
    precision = calculate_precision(all_labels, all_preds)
    recall = calculate_recall(all_labels, all_preds)
    f1 = calculate_f1_score(all_labels, all_preds)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

if __name__ == "__main__":
    main()
