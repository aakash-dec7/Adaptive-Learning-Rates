import json
import torch
import random
from Logger import logger
from transformers import AutoTokenizer
from torch.utils.data import TensorDataset


class PreprocessTrainData:
    def __init__(self, file_path="Data/dialogue.jsonl"):
        """
        Initializes the preprocessing class with the dialogue file path,
        sets up the tokenizer, and prepares the training data.
        """
        self.file_path = file_path
        self.tokenizer = AutoTokenizer.from_pretrained("Tokenizer/Custom")

        logger.info("Initialized PreprocessTrainData.")

        self.data = self.load_data()

    def load_data(self):
        """
        Loads dialogue data from the JSONL file and randomly selects 2500 dialogue pairs.
        """
        with open(self.file_path, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        if len(data) > 5000:
            data = random.sample(data, 5000)

        logger.debug(f"Loaded {len(data)} dialogue pairs from {self.file_path}")
        return data

    def preprocess_train_data(self):
        """
        Tokenizes inputs and expected outputs, creates labels for training,
        and stores the processed dataset.
        """
        input_ids_data = []
        attention_mask_data = []
        labels_data = []
        full_text_list = []

        logger.info("Starting data preprocessing...")

        for i, item in enumerate(self.data):
            inp = item["user"].strip()
            out = item["assistant"].strip()
            if not inp or not out:  # Skip empty inputs or outputs
                logger.warning(f"Skipping empty input or output at index {i}")
                continue

            # Construct the full sequence: user prompt + assistant response
            full_text = f"<|user|> {inp} <|assistant|> {out}"
            full_text_list.append(full_text)

            # Tokenize full text sequence
            tokenized = self.tokenizer(
                full_text,
                max_length=50,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )

            input_ids = tokenized["input_ids"][0]  # Shape: [max_length]
            attention_mask = tokenized["attention_mask"][0]  # Shape: [max_length]

            # Create labels for causal language modeling (shift for next token prediction)
            labels = input_ids.clone()  # Copy the input IDs to create labels
            labels[:-1] = input_ids[
                1:
            ]  # Shift all tokens one position left (next token prediction)
            labels[-1] = (
                self.tokenizer.pad_token_id
            )  # Last token has no target, set to pad_token_id

            # Mask user prompt (tokens before <|assistant|>) with pad_token_id
            try:
                assistant_index = full_text.index(
                    "<|assistant|>"
                )  # Find assistant's start
                tokens_before_assistant = self.tokenizer(
                    full_text[:assistant_index],
                    truncation=True,
                    max_length=50,
                    return_tensors="pt",
                )["input_ids"].shape[1]
                labels[:tokens_before_assistant] = (
                    self.tokenizer.pad_token_id
                )  # Ignore user prompt in loss
            except ValueError:
                logger.warning(f"Skipping sample {i}: <|assistant|> not found in text")
                continue

            input_ids_data.append(input_ids)
            attention_mask_data.append(attention_mask)
            labels_data.append(labels)

        if not input_ids_data:
            raise ValueError("No valid samples processed. Check input data.")

        # Combine all into a single TensorDataset
        dataset = TensorDataset(
            torch.stack(input_ids_data),
            torch.stack(attention_mask_data),
            torch.stack(labels_data),
        )

        # Save preprocessed dataset
        dataset_path = "Data/train_dataset.pt"
        torch.save(dataset, dataset_path)
        logger.info(f"Saved training dataset to {dataset_path}")

        return dataset
