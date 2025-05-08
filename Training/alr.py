import os
import torch
from Logger import logger
from datetime import datetime
from ALR.ALR import ALRScheduler
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
        self.model = LoadModel(model_dir="Model/TrainedModel/ALR").to(self.device)
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
        # self.scheduler = StepLR(self.optimizer, step_size=5, gamma=0.1)

        logger.info("Optimizer and scheduler initialized.")

        # Initialize ALRScheduler
        self.alr = ALRScheduler(
            optimizer=self.optimizer,
            base_lr=1e-5,
            min_lr=1e-6,
            max_lr=1e-4,
        )

        self.lr_list = []

        logger.info("Training model initialized successfully.")

    def get_gradient_norm(self):
        """
        Computes the average gradient norm for decoder layers.
        """
        grad_info = {}

        for name, param in self.model.named_parameters():
            if param.grad is not None and "decoder.layers" in name:
                layer_id = name.split(".")[2]
                grad_info.setdefault(layer_id, []).append(param.grad.norm().item())

        avg_norms = {layer: sum(vals) / len(vals) for layer, vals in grad_info.items()}
        overall_avg = sum(avg_norms.values()) / len(avg_norms) if avg_norms else 0.0
        return overall_avg

    def get_activation_norm(self, activations, num_layers=6, batch_size=32):
        """
        Computes the average L2 norm of activation outputs.
        """
        act_norms = [torch.norm(act).item() for act in activations]
        avg_norm = sum(act_norms) / len(act_norms) if act_norms else 0.0
        avg_norm = avg_norm / (num_layers * batch_size)
        return avg_norm

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
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_activations=True,
                )
                loss = output.loss
                activations = output.activations

                # Backward pass and optimization
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                self.optimizer.step()

                epoch_loss += loss.item()

                grad_norm = self.get_gradient_norm()
                act_norm = self.get_activation_norm(activations)

                logger.debug(f"Epoch {epoch+1} - Loss: {loss.item():.4f}")

            # Calculate and log average epoch loss
            avg_loss = epoch_loss / len(self.dataloader)
            epoch_loss_list.append(avg_loss)
            logger.info(f"Epoch {epoch+1}/{num_epochs} - Avg Loss: {avg_loss:.4f}")

            # Step scheduler and clean cache
            # self.scheduler.step()
            self.alr.update_learning_rate(grad_norm, act_norm)
            torch.cuda.empty_cache()

        # Save trained model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = "Model/TrainedModel/ALR"
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{timestamp}.pth")

        torch.save(self.model.state_dict(), save_path)
        logger.info(f"Model saved at: {save_path}")

        return epoch_loss_list, self.lr_list
