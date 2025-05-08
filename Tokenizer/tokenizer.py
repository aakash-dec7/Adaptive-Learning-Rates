from Logger import logger
from transformers import AutoTokenizer


def customize_and_save_tokenizer(
    model_name: str = "bert-base-uncased", save_path: str = "Tokenizer/Custom"
):
    """
    Loads a pre-trained tokenizer, adds custom special tokens, and saves the updated tokenizer.
    """

    logger.info(f"Loading tokenizer from pre-trained model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Define custom and standard special tokens
    special_tokens = {
        "unk_token": "[UNK]",  # Unknown token
        "pad_token": "[PAD]",  # Padding token
        "bos_token": "[BOS]",  # Beginning of sentence token
        "eos_token": "[EOS]",  # End of sentence token
        "additional_special_tokens": [  # Additional custom tokens
            "<|user|>",
            "<|assistant|>",
            "<eos>",
        ],
    }

    logger.info("Adding custom special tokens to the tokenizer.")
    tokenizer.add_special_tokens(special_tokens)

    logger.info(f"Saving the updated tokenizer to: {save_path}")
    tokenizer.save_pretrained(save_path)

    logger.info("Tokenizer customization and saving completed successfully.")
