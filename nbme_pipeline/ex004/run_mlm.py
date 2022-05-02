import argparse
import logging
import math
import os
import random
from pathlib import Path

import datasets
import pandas as pd
import torch
import transformers
from accelerate import Accelerator
from classify_env import classify_env
from config import CFG
from datasets import load_dataset
from get_tokenizer import copy_tokenizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    CONFIG_MAPPING,
    MODEL_MAPPING,
    AdamW,
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    SchedulerType,
    get_scheduler,
    set_seed,
)

if __name__ == "__main__":
    # Directory Setting
    CFG.__data_dir, CFG.__output_dir, CFG.__file_dir = classify_env(
        competiton_name=CFG.competition, exp_name=CFG.exp_name
    )

    if not Path.exists(CFG.__output_dir / "output_model"):
        Path.mkdir(CFG.__output_dir / "output_model", parents=True, exist_ok=True)

    copy_tokenizer(file_dir=CFG.__file_dir)

    # load data

    # patient_notes = pd.read_csv(CFG.__data_dir / "patient_notes.csv")
    mimic_data = pd.read_csv(CFG.__data_dir / "MIMIC-III-Final.csv")

    breakpoint()

    if CFG.debug:
        # patient_notes = patient_notes.sample(n=10, random_state=0).reset_index(drop=True)
        mimic_data = mimic_data.sample(n=10, random_state=0).reset_index(drop=True)
    else:
        mimic_data = mimic_data.sample(n=10000, random_state=0).reset_index(drop=True)

    # mlm_data = patient_notes[["pn_history"]]
    # mlm_data = mlm_data.rename(columns={"pn_history": "text"})
    # mlm_data.to_csv(CFG.__output_dir / "mlm_data.csv", index=False)

    # mlm_data_val = patient_notes[["pn_history"]]
    # mlm_data_val = mlm_data_val.rename(columns={"pn_history": "text"})
    # mlm_data_val.to_csv(CFG.__output_dir / "mlm_data_val.csv", index=False)

    mlm_data = mimic_data[["TEXT"]]
    mlm_data = mlm_data.rename(columns={"TEXT": "text"})
    mlm_data.to_csv(CFG.__output_dir / "mlm_data.csv", index=False)

    mlm_data_val = mimic_data[["TEXT"]]
    mlm_data_val = mlm_data_val.rename(columns={"TEXT": "text"})
    mlm_data_val.to_csv(CFG.__output_dir / "mlm_data_val.csv", index=False)

    logger = logging.getLogger(__name__)
    MODEL_CONFIG_CLASSES = list(MODEL_MAPPING.keys())
    MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

    # config
    class TrainConfig:
        train_file = "mlm_data.csv"
        validation_file = "mlm_data.csv"
        validation_split_percentage = 5
        pad_to_max_length = True
        model_name_or_path = "microsoft/deberta-v3-large"
        config_name = "microsoft/deberta-v3-large"
        tokenizer_name = "microsoft/deberta-v3-large"
        per_device_train_batch_size = 1
        per_device_eval_batch_size = 1
        learning_rate = 5e-5
        weight_decay = 0.0
        num_train_epochs = 2  # change to 5
        max_train_steps = None
        gradient_accumulation_steps = 1
        lr_scheduler_type = "constant_with_warmup"
        num_warmup_steps = 0
        output_dir = "output"
        use_slow_tokenizer = False
        max_seq_length = 512
        seed = 33
        line_by_line = False
        preprocessing_num_workers = 4
        overwrite_cache = True
        mlm_probability = 0.15

    config = TrainConfig()

    if config.train_file is not None:
        extension = config.train_file.split(".")[-1]
        assert extension in ["csv", "json", "txt"], "`train_file` should be a csv, json or txt file."
    if config.validation_file is not None:
        extension = config.validation_file.split(".")[-1]
        assert extension in ["csv", "json", "txt"], "`validation_file` should be a csv, json or txt file."
    if config.output_dir is not None:
        Path.mkdir(CFG.__output_dir / config.output_dir, parents=True, exist_ok=True)

    def main():
        train_args = TrainConfig()
        accelerator = Accelerator()
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )
        logger.info(accelerator.state)
        logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)

        # loggingのレベルの調整
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()

        if train_args.seed is not None:
            set_seed(train_args.seed)

        # dataをload_datasetが受け取れる形へ変換
        data_files = {}
        if train_args.train_file is not None:
            data_files["train"] = str(CFG.__output_dir / train_args.train_file)
        if train_args.validation_file is not None:
            data_files["validation"] = str(CFG.__output_dir / train_args.validation_file)

        # textファイルへの対応
        extension = train_args.train_file.split(".")[-1]
        if extension == "txt":
            extension = "text"

        raw_datasets = load_dataset(extension, data_files=data_files)

        if train_args.config_name:
            config = AutoConfig.from_pretrained(train_args.config_name)
        elif config.model_name_or_path:
            config = AutoConfig.from_pretrained(train_args.model_name_or_path)
        else:
            config = CONFIG_MAPPING[train_args.model_type]()
            logger.warning("You are instantiating a new config instance from scratch.")

        if train_args.tokenizer_name:
            tokenizer = AutoTokenizer.from_pretrained(
                train_args.tokenizer_name, use_fast=not train_args.use_slow_tokenizer
            )
        elif train_args.model_name_or_path:
            tokenizer = AutoTokenizer.from_pretrained(
                train_args.model_name_or_path, use_fast=not train_args.use_slow_tokenizer
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported by this script."
                "You can do it from another script, save it, and load it from here, using --tokenizer_name."
            )

        if train_args.model_name_or_path:
            model = AutoModelForMaskedLM.from_pretrained(
                train_args.model_name_or_path, from_tf=bool(".ckpt" in train_args.model_name_or_path), config=config,
            )
        else:
            logger.info("Training new model from scratch")
            model = AutoModelForMaskedLM.from_config(config)

        # model.resize_token_embeddings(len(tokenizer))

        column_names = raw_datasets["train"].column_names
        text_column_name = "text" if "text" in column_names else column_names[0]

        if train_args.max_seq_length is None:
            max_seq_length = tokenizer.model_max_length
            if max_seq_length > 1024:
                logger.warning(
                    f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                    "Picking 1024 instead. You can change that default value by passing --max_seq_length xxx."
                )
                max_seq_length = 1024
        else:
            if train_args.max_seq_length > tokenizer.model_max_length:
                logger.warning(
                    f"The max_seq_length passed ({train_args.max_seq_length}) is larger than the maximum length for the"
                    f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
                )
            max_seq_length = min(train_args.max_seq_length, tokenizer.model_max_length)

        def tokenize_function(examples):
            return tokenizer(examples[text_column_name], return_special_tokens_mask=True)

        tokenized_datasets = raw_datasets.map(
            tokenize_function,
            batched=True,
            num_proc=train_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not train_args.overwrite_cache,
        )

        def group_texts(examples):
            concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
            total_length = len(concatenated_examples[list(examples.keys())[0]])
            total_length = (total_length // max_seq_length) * max_seq_length
            result = {
                k: [t[i : i + max_seq_length] for i in range(0, total_length, max_seq_length)]
                for k, t in concatenated_examples.items()
            }
            return result

        tokenized_datasets = tokenized_datasets.map(
            group_texts,
            batched=True,
            num_proc=train_args.preprocessing_num_workers,
            load_from_cache_file=not train_args.overwrite_cache,
        )
        train_dataset = tokenized_datasets["train"]
        eval_dataset = tokenized_datasets["validation"]

        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=train_args.mlm_probability)
        train_dataloader = DataLoader(
            train_dataset, shuffle=True, collate_fn=data_collator, batch_size=train_args.per_device_train_batch_size
        )
        eval_dataloader = DataLoader(
            eval_dataset, collate_fn=data_collator, batch_size=train_args.per_device_eval_batch_size
        )

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": train_args.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=train_args.learning_rate)

        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader
        )

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / train_args.gradient_accumulation_steps)
        if train_args.max_train_steps is None:
            train_args.max_train_steps = train_args.num_train_epochs * num_update_steps_per_epoch
        else:
            train_args.num_train_epochs = math.ceil(train_args.max_train_steps / num_update_steps_per_epoch)

        lr_scheduler = get_scheduler(
            name=train_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=train_args.num_warmup_steps,
            num_training_steps=train_args.max_train_steps,
        )

        total_batch_size = (
            train_args.per_device_train_batch_size * accelerator.num_processes * train_args.gradient_accumulation_steps
        )

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(train_dataset)}")
        logger.info(f"  Num Epochs = {train_args.num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {train_args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {train_args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {train_args.max_train_steps}")
        # Only show the progress bar once on each machine.
        progress_bar = tqdm(range(train_args.max_train_steps), disable=not accelerator.is_local_main_process)
        completed_steps = 0

        for epoch in range(train_args.num_train_epochs):
            model.train()
            for step, batch in enumerate(train_dataloader):
                outputs = model(**batch)
                loss = outputs.loss
                loss = loss / train_args.gradient_accumulation_steps
                accelerator.backward(loss)
                if step % train_args.gradient_accumulation_steps == 0 or step == len(train_dataloader) - 1:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    progress_bar.update(1)
                    completed_steps += 1

                if completed_steps >= train_args.max_train_steps:
                    break

            model.eval()
            losses = []
            for step, batch in enumerate(eval_dataloader):
                with torch.no_grad():
                    outputs = model(**batch)

                loss = outputs.loss
                losses.append(accelerator.gather(loss.repeat(train_args.per_device_eval_batch_size)))

            losses = torch.cat(losses)
            losses = losses[: len(eval_dataset)]
            perplexity = math.exp(torch.mean(losses))

            logger.info(f"epoch {epoch}: perplexity: {perplexity}")

        if train_args.output_dir is not None:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(CFG.__output_dir / train_args.output_dir, save_function=accelerator.save)


if __name__ == "__main__":
    main()
