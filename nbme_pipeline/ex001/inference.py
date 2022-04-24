from typing import Tuple

import numpy as np
import pandas as pd
import torch
from config import CFG
from pred import get_predictions
from score import create_labels_for_scoring, get_char_probs, get_results, get_score
from tqdm import tqdm


def inference_fn(test_loader, model, device):
    preds = []
    model.eval()
    model.to(device)
    tk0 = tqdm(test_loader, total=len(test_loader))
    for inputs in tk0:
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        with torch.no_grad():
            y_preds = model(inputs)
        preds.append(y_preds.sigmoid().to("cpu").numpy())
    predictions = np.concatenate(preds)
    return predictions


def search_best_threshold(cfg: CFG, oof: pd.DataFrame) -> Tuple[int, int]:
    # search best threshold
    truths = create_labels_for_scoring(oof)
    char_probs = get_char_probs(oof["pn_history"].values, oof[[i for i in range(cfg.max_len)]].values, cfg.tokenizer)
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
        cfg.__logger.info(f"th: {th}  score: {score:.5f}")
    return best_th, best_score
