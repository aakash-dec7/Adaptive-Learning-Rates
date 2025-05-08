import os
import torch
from Logger import logger
from datetime import datetime
from Utils.utils import LoadModel
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import StepLR


class TrainModel:
    def __init__(self, data_path="Data/train_dataset.pt", batch_size=32):
        """
        Initializes training class with model, tokenizer, dataset, optimizer, and scheduler.
        """
        logger.info("Initializing training pipeline...")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load pretrained model
        self.model = LoadModel(model_dir="Model/TrainedModel/No_ALR").to(self.device)
        logger.info("Model loaded and moved to device.")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("Tokenizer/Custom")
        logger.info("Tokenizer loaded.")

        # Load training dataset
        self.dataset = torch.load(data_path, weights_only=False)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        logger.info(f"Dataset loaded from {data_path} with batch size {batch_size}.")

        # Initialize optimizer and scheduler
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-5, weight_decay=0.01
        )
        self.lr_list = []

        self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)
        logger.info("Optimizer and scheduler initialized.")

    def forward(self, num_epochs=30):
        """
        Runs the training loop for the given number of epochs.
        """
        self.model.train()
        epoch_loss_list = []

        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            epoch_loss = 0.0

            # Append current learning rate
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.lr_list.append(current_lr)

            for input_ids, attention_mask, labels in self.dataloader:
                self.optimizer.zero_grad()

                # Move data to the correct device
                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                output = self.model(
                    input_ids=input_ids, attention_mask=attention_mask, labels=labels
                )
                loss = output.loss

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

                epoch_loss += loss.item()

                logger.debug(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

            # Calculate and log average epoch loss
            avg_loss = epoch_loss / len(self.dataloader)
            epoch_loss_list.append(avg_loss)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

            # Step scheduler and clean cache
            self.scheduler.step()
            torch.cuda.empty_cache()

        # Save trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "Model/TrainedModel/No_ALR"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}.pth")

        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved at: {save_path}")

        return epoch_loss_list, self.lr_list
