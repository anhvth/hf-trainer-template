# base_trainer.py
from argparse import ArgumentParser
import logging
import os
from loguru import logger
import tabulate
import torch
import transformers
from transformers import (
    Trainer,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import yaml

from hftrainer.trainer.base_dataclasses import (
    DataArguments,
    LoraArguments,
    ModelArguments,
    TrainingArguments,
)


def init_model(model_args, training_args, lora_args, device_map=None):
    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    model_load_kwargs = {
        "low_cpu_mem_usage": not transformers.deepspeed.is_deepspeed_zero3_enabled(),
    }

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    config.use_cache = False

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        quantization_config=(
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=compute_dtype,
            )
            if training_args.use_lora and lora_args.q_lora
            else None
        ),
        **model_load_kwargs,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token_id = tokenizer.eos_token_id

    if training_args.use_lora:

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=lora_args.lora_target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
    return model, tokenizer


if os.getenv("JUPYTER") == "True":
    from speedy import imemoize

    logger.info("Using imemoize in Jupyter notebook.")
    init_model = imemoize(init_model)


def parse_args(config_path=None, verbose=True):
    parser = ArgumentParser()
    if config_path is None:
        parser.add_argument("--config_path", type=str, default=config_path)
        return parser.parse_args()["config_path"]

    def load_yaml(file_path):
        with open(file_path, "r") as f:
            return yaml.safe_load(f)

    all_args = load_yaml(config_path)

    if verbose:
        for k in all_args.keys():
            print("\n\nArguments table:", k)
            v = all_args[k]
            v = tabulate.tabulate(all_args[k].items(), tablefmt="fancy_grid")
            print(v)

    model_args = ModelArguments(**all_args["model_args"])
    data_args = DataArguments(**all_args["data_args"])
    training_args = TrainingArguments(**all_args["training_args"])
    lora_args = LoraArguments(**all_args["lora_args"])

    return model_args, data_args, training_args, lora_args


class BaseTrainer(Trainer):
    def __init__(self, config_path, verbose=True):
        self.model_args, self.data_args, self.training_args, self.lora_args = (
            parse_args(config_path, verbose=verbose)
        )
        self.load_model()
        dataset = self.load_datasets()

        # Initialize the CustomTrainer parent class
        super().__init__(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=self.tokenizer,
        )

    def load_model(self):
        logger.debug("Loading model and tokenizer with provided arguments.")
        self.model, self.tokenizer = init_model(
            self.model_args, self.training_args, self.lora_args
        )
        logger.debug("Model and tokenizer loaded successfully.")

    def load_datasets(self):
        raise NotImplementedError(
            "Subclasses should implement this method to load datasets."
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
