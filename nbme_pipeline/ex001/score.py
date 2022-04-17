import ast
import itertools

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from pred import get_predictions


def get_char_probs(texts, predictions, tokenizer):
    results = [np.zeros(len(t)) for t in texts]
    for i, (text, prediction) in enumerate(zip(texts, predictions)):
        encoded = tokenizer(text, add_special_tokens=True, return_offsets_mapping=True)
        for idx, (offset_mapping, pred) in enumerate(
            zip(encoded["offset_mapping"], prediction)
        ):
            start = offset_mapping[0]
            end = offset_mapping[1]
            results[i][start:end] = pred
    return results


def get_result(oof_df: pd.DataFrame, max_len: int, tokenizer, logger) -> None:
    """Save prediction result by logger"""
    labels = create_labels_for_scoring(oof_df)
    predictions = oof_df[[i for i in range(max_len)]].values
    char_probs = get_char_probs(oof_df["pn_history"].values, predictions, tokenizer)
    results = get_results(char_probs, th=0.5)
    preds = get_predictions(results)
    score = get_score(labels, preds)
    logger.info(f"Score: {score:<.4f}")


def get_results(char_probs, th=0.5):
    results = []
    for char_prob in char_probs:
        result = np.where(char_prob >= th)[0] + 1
        result = [
            list(g)
            for _, g in itertools.groupby(
                result, key=lambda n, c=itertools.count(): n - next(c)
            )
        ]
        result = [f"{min(r)} {max(r)}" for r in result]
        result = ";".join(result)
        results.append(result)
    return results


def micro_f1(preds: list[int], truths: list[int]) -> float:
    # From https://www.kaggle.com/theoviel/evaluation-metric-folds-baseline
    """
    Micro f1 on binary arrays.

    Args:
        preds (list of lists of ints): Predictions.
        truths (list of lists of ints): Ground truths.

    Returns:
        float: f1 score.
    """
    # Micro : aggregating over all instances
    preds = np.concatenate(preds)
    truths = np.concatenate(truths)
    return f1_score(truths, preds)


def spans_to_binary(spans: list[int], length=None) -> np.ndarray:
    """
    Converts spans to a binary array indicating whether each character is in the span.

    Args:
        spans (list of lists of two ints): Spans.

    Returns:
        np array [length]: Binarized spans.
    """
    length = np.max(spans) if length is None else length
    binary = np.zeros(length)
    for start, end in spans:
        binary[start:end] = 1
    return binary


def span_micro_f1(preds, truths):
    """
    Micro f1 on spans.

    Args:
        preds (list of lists of two ints): Prediction spans.
        truths (list of lists of two ints): Ground truth spans.

    Returns:
        float: f1 score.
    """
    bin_preds = []
    bin_truths = []
    for pred, truth in zip(preds, truths):
        if not len(pred) and not len(truth):
            continue
        length = max(
            np.max(pred) if len(pred) else 0, np.max(truth) if len(truth) else 0
        )
        bin_preds.append(spans_to_binary(pred, length))
        bin_truths.append(spans_to_binary(truth, length))
    return micro_f1(bin_preds, bin_truths)


def create_labels_for_scoring(df):
    # example: ['0 1', '3 4'] -> ['0 1; 3 4']
    df["location_for_create_labels"] = [ast.literal_eval(f"[]")] * len(df)
    for i in range(len(df)):
        lst = df.loc[i, "location"]
        if lst:
            new_lst = ";".join(lst)
            df.loc[i, "location_for_create_labels"] = ast.literal_eval(
                f'[["{new_lst}"]]'
            )
    # create labels
    truths = []
    for location_list in df["location_for_create_labels"].values:
        truth = []
        if len(location_list) > 0:
            location = location_list[0]
            for loc in [s.split() for s in location.split(";")]:
                start, end = int(loc[0]), int(loc[1])
                truth.append([start, end])
        truths.append(truth)
    return truths


def get_score(y_true, y_pred):
    score = span_micro_f1(y_true, y_pred)
    return score
