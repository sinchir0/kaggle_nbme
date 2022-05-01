import os
import platform
from pathlib import Path, PosixPath
from typing import Tuple


def classify_env(competiton_name: str, exp_name: str) -> Tuple[PosixPath, PosixPath, PosixPath]:
    # Colab
    if "COLAB_GPU" in set(os.environ.keys()):
        DATA_DIR = Path("/", "content", "drive", "MyDrive", "Kaggle", competiton_name, "input")
        OUTPUT_DIR = DATA_DIR.parents[0] / exp_name
        FILE_DIR = Path(__file__).parents[0]
        ENV_TYPE = "colab"

    # kaggle
    elif "KAGGLE_URL_BASE" in set(os.environ.keys()):
        DATA_DIR = Path("input/")
        OUTPUT_DIR = Path("./")
        FILE_DIR = Path(__file__).parents[0]
        ENV_TYPE = "kaggle"

    # macOS
    elif platform.system() == "Darwin":
        DATA_DIR = Path(__file__).parents[2] / "input"
        OUTPUT_DIR = FILE_DIR = Path(__file__).parents[0]
        ENV_TYPE = "macOS"

    else:
        raise ValueError("Can't classify your environment")

    print(f"Set environ for {ENV_TYPE}")
    return DATA_DIR, OUTPUT_DIR, FILE_DIR
