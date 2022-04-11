import os
from pathlib import Path
import pandas as pd


def data_load(data_path: Path) -> :
    df = pd.read_csv(os.path.join(data_path, "train.csv"))
    features = pd.read_csv(os.path.join(data_path, "features.csv"))
    patient_notes = pd.read_csv(os.path.join(data_path, "patient_notes.csv"))
    return train, features, patient_notes
