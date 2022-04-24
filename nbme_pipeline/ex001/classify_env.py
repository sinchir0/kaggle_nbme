import os
import platform
from pathlib import Path
from typing import Tuple


def classify_env(competiton_name: str, exp_name: str) -> Tuple[Path, Path, Path]:
    # Colab
    if "COLAB_GPU" in set(os.environ.keys()):
        DATA_DIR = Path("/", "content", "drive", "MyDrive", "Kaggle", competiton_name, "input")
        EXP_DIR = DATA_DIR.parents[0] / exp_name
        OUTPUT_MODEL_DIR = EXP_DIR / "output_model"

        print("Set environ for COLAB")

    # kaggle
    elif "KAGGLE_URL_BASE" in set(os.environ.keys()):
        DATA_DIR = Path("input/")
        EXP_DIR = Path("./")
        OUTPUT_MODEL_DIR = EXP_DIR / "output_model"

        print("Set environ for Kaglle")

    # macOS
    elif platform.system() == "Darwin":
        DATA_DIR = Path(__file__).parents[2] / "input"
        EXP_DIR = Path(__file__).parents[0]
        OUTPUT_MODEL_DIR = EXP_DIR / "output_model"

        print("Set environ for macOS")

    else:
        raise ValueError("Can't classify your environment")

    return DATA_DIR, EXP_DIR, OUTPUT_MODEL_DIR
