from src.model import setup_model

from torch import optim
from tqdm import tqdm

import logging
import torch

# Configure logging to use green color
logging.basicConfig(
    level=logging.INFO,
    format="\033[92m%(message)s\033[0m"  # Green color
)

# Step 9: Start the Training Process
def train_model(data_dir, num_epochs=10, learning_rate=1e-5):
    model, train_loader, test_loader, device = setup_model(data_dir)
    
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    model.train()

    for epoch in range(num_epochs):
        logging.info(f"Starting epoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0.0
        
        # Iterate over the training data
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images = batch['pixel_values'].to(device)
            targets = batch['labels'].to(device)
            boxes = batch['boxes'].to(device)

            # Create target dictionary (DETR expects this format)
            target = {
                "labels": targets,
                "boxes": boxes
            }

            # Zero the gradients
            optimizer.zero_grad()

            # Forward pass: Compute predicted outputs
            outputs = model(images, labels=target['labels'], pixel_values=images)

            # Loss is stored in outputs.loss
            loss = outputs.loss
            epoch_loss += loss.item()

            # Backpropagation
            loss.backward()

            # Update model parameters
            optimizer.step()

        # Print average loss for the epoch
        logging.info(f"Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss / len(train_loader)}")

    torch.save(model.state_dict(), "detr_model.pth")
    logging.info("Model saved to 'detr_model.pth'")

if __name__ == "__main__":
    train_model(data_dir=dataset_dir, num_epochs=10, learning_rate=1e-5)

