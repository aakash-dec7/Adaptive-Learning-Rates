import torch
import pandas as pd
from Data.load_dataset import LoadDataset
from Preprocess.preprocess import PreprocessTrainData
from Tokenizer.tokenizer import customize_and_save_tokenizer
from Training.alr import TrainModel as Train_ALR
from Training.step_lr import TrainModel as Train_Step_LR
from Training.alr_pairwise import TrainModel as Train_ALR_Pairwise
from Plot.lr import plot_lr_comparison
from Plot.loss import plot_loss_comparison


def main():
    # Step 1: Load and save dataset
    dataset_loader = LoadDataset()
    dataset_loader.load_and_preprocess()

    # Step 2: Configure Tokenizer
    customize_and_save_tokenizer()

    # Step 3: Preprocess dataset
    preprocessor = PreprocessTrainData()
    preprocessor.preprocess_train_data()

    # Step 4: Train model
    train_no_alr = Train_Step_LR()
    no_alr_loss, no_alr_LR = train_no_alr.forward()

    # Step 5: Train model
    train_alr = Train_ALR()
    alr_loss, alr_LR = train_alr.forward()

    # Step 6: Train model
    train_alr_pair = Train_ALR_Pairwise()
    alr_pair_loss, alr_pair_LR = train_alr_pair.forward()

    # Step 7: Save Loss and LR as CSV
    max_len_lr = max(len(no_alr_LR), len(alr_LR), len(alr_pair_LR))
    lr_df = pd.DataFrame(
        {
            "Step-LR": no_alr_LR + [None] * (max_len_lr - len(no_alr_LR)),
            "ALR": alr_LR + [None] * (max_len_lr - len(alr_LR)),
            "ALR-Pairwise": alr_pair_LR + [None] * (max_len_lr - len(alr_pair_LR)),
        }
    )
    lr_df.to_csv("LR.csv", index=False)

    max_len = max(len(no_alr_loss), len(alr_loss), len(alr_pair_loss))
    loss_df = pd.DataFrame(
        {
            "Step-LR": no_alr_loss + [None] * (max_len - len(no_alr_loss)),
            "ALR": alr_loss + [None] * (max_len - len(alr_loss)),
            "ALR-Pairwise": alr_pair_loss + [None] * (max_len - len(alr_pair_loss)),
        }
    )
    loss_df.to_csv("Loss.csv", index=False)

    # Step 8: Plot Loss Comparison
    plot_loss_comparison(
        loss_lists=[no_alr_loss, alr_loss, alr_pair_loss],
        labels=["Step-LR", "ALR", "ALR-Pairwise"],
    )

    # Step 9: Plot LR Comparison
    plot_lr_comparison(
        lr_lists=[no_alr_LR, alr_LR, alr_pair_LR],
        labels=["Step-LR", "ALR", "ALR-Pairwise"],
    )


if __name__ == "__main__":
    main()
