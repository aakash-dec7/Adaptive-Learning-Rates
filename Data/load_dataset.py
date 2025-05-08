import json
from Logger import logger
from datasets import load_dataset


class LoadDataset:
    def __init__(self, file_path: str = "Data/dialogue.jsonl"):
        """
        Initializes the dataset loader with a target file path.
        """
        self.file_path = file_path
        logger.info(f"Initialized LoadDataset with file path: {self.file_path}")

    def load_and_preprocess(self):
        """
        Loads the DailyDialog dataset and writes user-assistant dialogue pairs
        to a .jsonl file in the specified format.
        """
        logger.info("Loading dataset 'frankdarkluo/DailyDialog'...")
        try:
            dataset = load_dataset(
                "frankdarkluo/DailyDialog", split="train", trust_remote_code=True
            )
            logger.info("Dataset loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return

        context = dataset["context"]
        response = dataset["response"]

        logger.info(f"Saving dialogue pairs to {self.file_path}...")
        try:
            with open(self.file_path, "w", encoding="utf-8") as f:
                count = 0
                for user, assistant in zip(context, response):
                    if user and assistant:
                        # Clean and format each pair before writing
                        example = {"user": user.strip(), "assistant": assistant.strip()}
                        f.write(json.dumps(example, ensure_ascii=False) + "\n")
                        count += 1
                logger.info(f"Saved {count} valid dialogue pairs.")
        except Exception as e:
            logger.error(f"Error writing to file: {e}")
