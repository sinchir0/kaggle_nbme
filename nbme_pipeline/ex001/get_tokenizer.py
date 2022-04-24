from pathlib import PosixPath

from transformers import AutoTokenizer


def get_tokenizer(model_name: str, output_dir: PosixPath) -> AutoTokenizer:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(output_dir / "tokenizer")
    return tokenizer
