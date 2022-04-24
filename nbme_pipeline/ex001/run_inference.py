import gc

import numpy as np
import pandas as pd
import torch
from classify_env import classify_env
from config import CFG
from dataset import TestDataset
from inference import inference_fn
from logger import get_logger
from model import CustomModel
from pred import get_predictions
from score import create_labels_for_scoring, get_char_probs, get_results, get_score
from set_seed import seed_everything
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    CFG = CFG()

    # dubug
    if CFG.debug:
        CFG.epochs = 2
        CFG.trn_fold = [0]

    # set seed
    seed_everything(seed=42)

    # Directory Setting
    CFG._data_dir, CFG._exp_dir, CFG._output_model_dir = classify_env(
        competiton_name=CFG.competition, exp_name=CFG.exp_name
    )

    # logger
    CFG._logger = get_logger()

    # tokenizer
    CFG.tokenizer = AutoTokenizer.from_pretrained(CFG._exp_dir / "tokenizer")

    # oof
    oof = pd.read_pickle(CFG._exp_dir / "oof_df.pkl")

    # search best threshold
    truths = create_labels_for_scoring(oof)
    char_probs = get_char_probs(oof["pn_history"].values, oof[[i for i in range(CFG.max_len)]].values, CFG.tokenizer)
    best_th = 0.5
    best_score = 0.0
    for th in np.arange(0.40, 0.60, 0.01):  # TODO: ここの探索範囲を広げても良いかも
        th = np.round(th, 2)
        results = get_results(char_probs, th=th)
        preds = get_predictions(results)
        score = get_score(preds, truths)
        if best_score < score:
            best_th = th
            best_score = score
        CFG._logger.info(f"th: {th}  score: {score:.5f}")
    CFG._logger.info(f"best_th: {best_th}  score: {best_score:.5f}")

    # data loading
    test = pd.read_csv(CFG._data_dir / "test.csv")
    submission = pd.read_csv(CFG._data_dir / "sample_submission.csv")
    features = pd.read_csv(CFG._data_dir / "features.csv")

    def preprocess_features(features: pd.DataFrame) -> pd.DataFrame:
        features.loc[27, "feature_text"] = "Last-Pap-smear-1-year-ago"
        return features

    features = preprocess_features(features)
    patient_notes = pd.read_csv(CFG._data_dir / "patient_notes.csv")

    test = test.merge(features, on=["feature_num", "case_num"], how="left")
    test = test.merge(patient_notes, on=["pn_num", "case_num"], how="left")

    test_dataset = TestDataset(CFG, test)
    test_loader = DataLoader(
        test_dataset,
        batch_size=CFG.batch_size,
        shuffle=False,
        num_workers=CFG.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    predictions = []

    for fold in CFG.trn_fold:
        model = CustomModel(cfg=CFG, config_path=CFG._output_model_dir / "config.pth", pretrained=False)
        state = torch.load(
            CFG._output_model_dir / f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth",
            map_location=torch.device("cpu"),
        )
        model.load_state_dict(state["model"])
        prediction = inference_fn(test_loader, model, device)
        # (testの行数、max_len(=466)、1) => (testの行数、max_len(=466))
        prediction = prediction.reshape((len(test), CFG.max_len))
        char_probs = get_char_probs(test["pn_history"].values, prediction, CFG.tokenizer)
        predictions.append(char_probs)
        del model, state, prediction, char_probs
        gc.collect()
        torch.cuda.empty_cache()
    predictions = np.mean(predictions, axis=0)

    # submission
    results = get_results(predictions, th=best_th)
    submission["location"] = results
    submission[["id", "location"]].to_csv(CFG._exp_dir / f"{CFG.exp_name}_submission.csv", index=False)
