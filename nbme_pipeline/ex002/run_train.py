import os
from pathlib import Path

import pandas as pd
import tokenizers
import torch
import transformers
import wandb
from calc_text_len import calc_length
from classify_env import classify_env
from config import CFG
from cv import add_fold
from get_tokenizer import get_tokenizer
from logger import get_logger
from preprocess import fix_incorrect_annoation, preprocess_features, preprocess_train
from score import get_result
from set_seed import seed_everything
from set_wandb import run_wandb
from train import train_loop

if __name__ == "__main__":

    # Directory Setting
    CFG.__data_dir, CFG.__output_dir, CFG.__model_dir = classify_env(
        competiton_name=CFG.competition, exp_name=CFG.exp_name
    )

    if not os.path.exists(CFG.__model_dir):
        os.makedirs(CFG.__output_dir, exist_ok=True)

    if not os.path.exists(CFG.__model_dir / "output_model"):
        os.makedirs(CFG.__output_dir / "output_model", exist_ok=True)

    # dubug
    if CFG.debug:
        CFG.epochs = 2
        CFG.trn_fold = [0]

    # wandb
    if CFG.wandb:
        run_wandb(CFG=CFG)

    # Library
    print(f"tokenizers.__version__: {tokenizers.__version__}")
    print(f"transformers.__version__: {transformers.__version__}")

    CFG.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CFG.__logger = get_logger()

    # fix seed
    seed_everything(seed=42)

    # Data Loading
    train = pd.read_csv(CFG.__data_dir / "train.csv")
    features = pd.read_csv(CFG.__data_dir / "features.csv")
    patient_notes = pd.read_csv(CFG.__data_dir / "patient_notes.csv")

    # Preprocess
    train = preprocess_train(train)
    train = fix_incorrect_annoation(train)
    train["annotation_length"] = train["annotation"].apply(len)
    features = preprocess_features(features)

    # Merge
    train = train.merge(features, on=["feature_num", "case_num"], how="left")
    train = train.merge(patient_notes, on=["pn_num", "case_num"], how="left")

    # CV split
    train = add_fold(train=train, n_fold=CFG.n_fold)

    if CFG.debug:
        train = train.sample(n=10, random_state=0).reset_index(drop=True)

    # tokenizer
    CFG.tokenizer = get_tokenizer(model_name=CFG.model, output_dir=CFG.__output_dir)

    # Define max_len
    if CFG.debug:
        CFG.max_len = 466
    else:
        pn_history_lengths = calc_length(patient_notes, "pn_history", CFG.tokenizer)
        feature_text_lengths = calc_length(features, "feature_text", CFG.tokenizer)
        CFG.max_len = max(pn_history_lengths) + max(feature_text_lengths) + 3  # cls & sep & sep
    CFG.__logger.info(f"max_len: {CFG.max_len}")

    # train
    if CFG.train:
        oof_df = pd.DataFrame()
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(folds=train, fold=fold, CFG=CFG,)
                oof_df = pd.concat([oof_df, _oof_df])
                # Save fold result
                CFG.__logger.info(f"========== fold: {fold} result ==========")
                get_result(
                    oof_df=_oof_df, max_len=CFG.max_len, tokenizer=CFG.tokenizer, logger=CFG.__logger,
                )
        # Save CV result
        oof_df = oof_df.reset_index(drop=True)
        CFG.__logger.info("========== CV ==========")
        get_result(
            oof_df=oof_df, max_len=CFG.max_len, tokenizer=CFG.tokenizer, logger=CFG.__logger,
        )
        oof_df.to_pickle(CFG.__output_dir / "oof_df.pkl")

    # finish wandb
    if CFG.wandb:
        wandb.finish()
