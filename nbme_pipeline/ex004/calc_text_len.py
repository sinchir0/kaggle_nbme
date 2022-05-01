import pandas as pd
import transformers
from tqdm import tqdm


def calc_length(
    df: pd.DataFrame,
    text_col: str,
    tokenizer: transformers.models.deberta.tokenization_deberta_fast.DebertaTokenizerFast,
) -> list:
    lengths = []
    tk0 = tqdm(df[text_col].fillna("").values, total=len(df))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        lengths.append(length)
    return lengths
