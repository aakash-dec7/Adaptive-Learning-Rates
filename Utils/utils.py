import os
import torch
from Logger import logger
from Model.model import DecTransformer


class LoadModel:
    def __new__(cls, model_dir="Model/TrainedModel"):
        """
        Loads the most recent trained model from the specified directory.
        If no weights are found, returns a base model.
        """
        # Determine device availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Initialize the model and move it to the appropriate device
        model = DecTransformer().to(device)
        logger.debug("Initialized base DecTransformer model.")

        # Check if model directory exists
        if not os.path.exists(model_dir):
            logger.warning(
                f"Directory '{model_dir}' does not exist. Returning base model."
            )
            return model

        # List all .pth weight files in the directory
        weight_files = [f for f in os.listdir(model_dir) if f.endswith(".pth")]

        # If no weights are found, return base model
        if not weight_files:
            logger.warning(
                f"No model weights found in '{model_dir}'. Returning base model."
            )
            return model

        # Sort files to get the latest one
        weight_files.sort(reverse=True)
        latest_weight_path = os.path.join(model_dir, weight_files[0])

        # Load the latest weights into the model
        try:
            model.load_state_dict(torch.load(latest_weight_path, map_location=device))
            logger.info(f"Successfully loaded model weights from: {latest_weight_path}")
        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            logger.info("Returning base model due to load failure.")

        return model
