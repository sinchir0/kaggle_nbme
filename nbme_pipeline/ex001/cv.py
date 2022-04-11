import pandas as pd
from sklearn.model_selection import GroupKFold


def add_fold(train: pd.DataFrame, n_fold: int) -> pd.DataFrame:
    Fold = GroupKFold(n_splits=n_fold)
    groups = train["pn_num"].values
    for n, (_, val_index) in enumerate(Fold.split(train, train["location"], groups)):
        train.loc[val_index, "fold"] = int(n)
    train["fold"] = train["fold"].astype(int)
    return train
