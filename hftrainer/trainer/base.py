# base_trainer.py
import torch
import transformers
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from dataclasses import dataclass, field
from typing import Dict, Optional, List
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = "Qwen1.5/Qwen-1_8B"


@dataclass
class DataArguments:
    data_path: str = None
    eval_data_path: str = None
    lazy_preprocess: bool = False


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = None
    optim: str = "adamw_torch"
    model_max_length: int = 8192
    use_lora: bool = False


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "gate_proj",
            "down_proj",
        ]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


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


class BaseTrainer:
    def __init__(
        self,
        model_args: ModelArguments,
        data_args: DataArguments,
        training_args: TrainingArguments,
        lora_args: LoraArguments,
    ):
        self.model_args, self.data_args, self.training_args, self.lora_args = (
            model_args,
            data_args,
            training_args,
            lora_args,
        )
        self.load_datasets()
        self.load_model()
        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            tokenizer=self.tokenizer,
        )

    def load_model(self):
        self.model, self.tokenizer = init_model(
            self.model_args, self.training_args, self.lora_args
        )

    def load_datasets(self):
        # Placeholder for loading datasets, to be implemented in subclasses
        raise NotImplementedError(
            "Subclasses should implement this method to load datasets."
        )

    def train(self):
        self.trainer.train()
