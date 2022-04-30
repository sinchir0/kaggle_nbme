import importlib
import shutil
from pathlib import Path, PosixPath

import transformers


def copy_tokenizer(output_dir: PosixPath) -> None:

    transformers_path = Path(transformers.__file__).parent

    input_dir = Path(output_dir / "deberta-v2-3-fast-tokenizer")

    convert_file = input_dir / "convert_slow_tokenizer.py"
    conversion_path = transformers_path / convert_file.name

    if conversion_path.exists():
        conversion_path.unlink()

    shutil.copy(convert_file, transformers_path)

    deberta_v2_path = transformers_path / "models" / "deberta_v2"

    for filename in ["tokenization_deberta_v2.py", "tokenization_deberta_v2_fast.py", "deberta__init__.py"]:
        if str(filename).startswith("deberta"):
            filepath = deberta_v2_path / str(filename).replace("deberta", "")
        else:
            filepath = deberta_v2_path / filename
        if filepath.exists():
            filepath.unlink()

        shutil.copy(input_dir / filename, filepath)

    # TODO: 2回trainしないと動かないのをなんとかする
    # importlib.reload(transformers)


def get_tokenizer(model_name: str, output_dir: PosixPath):
    # DebertaV2TokenizerFastを使うためのコードを利用しているpythonのsite-packages/transformers配下にコピーする
    copy_tokenizer(output_dir)
    from transformers.models.deberta_v2 import DebertaV2TokenizerFast

    tokenizer = DebertaV2TokenizerFast.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    return tokenizer
