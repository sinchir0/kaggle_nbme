import os
from pathlib import Path

import pandas as pd
import tokenizers
import torch
import transformers
import wandb
from calc_text_len import calc_length
from config import CFG
from cv import add_fold
from logger import get_logger
from preprocess import fix_incorrect_annoation, preprocess_features, preprocess_train
from score import get_result
from set_seed import seed_everything
from set_wandb import run_wandb
from train import train_loop
from transformers import AutoTokenizer

if __name__ == "__main__":

    # ====================================================
    # Directory Setting
    # ====================================================

    # OUTPUT_DIR = Path(__file__).parents[0]
    # MODEL_DIR = OUTPUT_DIR / "output_model"
    # if not os.path.exists(MODEL_DIR):
    #     os.makedirs(MODEL_DIR)

    CFG._output_dir = Path(__file__).parents[0]
    CFG._model_dir = CFG._output_dir / "output_model"
    if not os.path.exists(CFG._model_dir):
        os.makedirs(CFG._output_dir)

    # ====================================================
    # CFG
    # ====================================================

    if CFG.debug:
        CFG.epochs = 2
        CFG.trn_fold = [0]

    # ====================================================
    # wandb
    # ====================================================

    if CFG.wandb:
        run_wandb(CFG)

    # ====================================================
    # Library
    # ====================================================

    print(f"tokenizers.__version__: {tokenizers.__version__}")
    print(f"transformers.__version__: {transformers.__version__}")

    CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # LOGGER = get_logger()
    CFG._logger = get_logger()

    seed_everything(seed=42)

    # ====================================================
    #   Data Loading
    # ====================================================
    data_path = os.path.join(Path(__file__).parents[2], "input")
    train = pd.read_csv(os.path.join(data_path, "train.csv"))
    features = pd.read_csv(os.path.join(data_path, "features.csv"))
    patient_notes = pd.read_csv(os.path.join(data_path, "patient_notes.csv"))

    # ====================================================
    #   Preprocess
    # ====================================================

    train = preprocess_train(train)
    train = fix_incorrect_annoation(train)
    train["annotation_length"] = train["annotation"].apply(len)
    features = preprocess_features(features)

    # ====================================================
    #   Merge
    # ====================================================

    train = train.merge(features, on=["feature_num", "case_num"], how="left")
    train = train.merge(patient_notes, on=["pn_num", "case_num"], how="left")

    # ====================================================
    # CV split
    # ====================================================
    train = add_fold(train=train, n_fold=CFG.n_fold)

    if CFG.debug:
        train = train.sample(n=10, random_state=0).reset_index(drop=True)

    # ====================================================
    # tokenizer
    # ====================================================
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(CFG._output_dir / "tokenizer")
    CFG.tokenizer = tokenizer

    # Dataset
    # ====================================================
    # Define max_len
    # ====================================================
    if CFG.debug:
        CFG.max_len = 1000
    else:
        pn_history_lengths = calc_length(patient_notes, "pn_history", tokenizer)
        feature_text_lengths = calc_length(features, "feature_text", tokenizer)

        CFG.max_len = max(pn_history_lengths) + max(feature_text_lengths) + 3  # cls & sep & sep
        CFG._logger.info(f"max_len: {CFG.max_len}")

    # ====================================================
    # Dataset
    # ====================================================

    # Model
    # ====================================================
    # Model
    # ====================================================

    # Helpler functions
    # ====================================================
    # Helper functions
    # ====================================================

    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(
                    folds=train,
                    fold=fold,
                    CFG=CFG,
                )
                oof_df = pd.concat([oof_df, _oof_df])
                # Save fold result
                CFG._logger.info(f"========== fold: {fold} result ==========")
                get_result(
                    oof_df=_oof_df,
                    max_len=CFG.max_len,
                    tokenizer=CFG.tokenizer,
                    logger=CFG._logger,
                )
        # Save CV result
        oof_df = oof_df.reset_index(drop=True)
        CFG._logger.info("========== CV ==========")
        get_result(
            oof_df=oof_df,
            max_len=CFG.max_len,
            tokenizer=CFG.tokenizer,
            logger=CFG._logger,
        )
        oof_df.to_pickle(CFG._output_dir / "oof_df.pkl")

    if CFG.wandb:
        wandb.finish()
