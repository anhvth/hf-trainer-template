# src/base_trainer.py
import transformers
from transformers import Trainer, TrainingArguments, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from dataclasses import dataclass, field
from typing import Dict, Optional, List
import torch

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = "Qwen/Qwen-7B"

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
    lora_target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "gate_proj", "down_proj"
    ])
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
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
        ) if training_args.use_lora and lora_args.q_lora else None,
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

    return model, tokenizer

class BaseTrainer:
    def __init__(self, args):
        self.model_args = ModelArguments(**args["model_args"])
        self.data_args = DataArguments(**args["data_args"])
        self.training_args = TrainingArguments(**args["training_args"])
        self.lora_args = LoraArguments(**args["lora_args"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.tokenizer = init_model(
            self.model_args, self.training_args, self.lora_args
        )

        if self.training_args.use_lora:
            from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
            lora_config = LoraConfig(
                r=self.lora_args.lora_r,
                lora_alpha=self.lora_args.lora_alpha,
                target_modules=self.lora_args.lora_target_modules,
                lora_dropout=self.lora_args.lora_dropout,
                bias=self.lora_args.lora_bias,
                task_type="CAUSAL_LM",
            )
            if self.lora_args.q_lora:
                self.model = prepare_model_for_kbit_training(
                    self.model, use_gradient_checkpointing=self.training_args.gradient_checkpointing
                )
            self.model = get_peft_model(self.model, lora_config)

        self.trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=None,
            eval_dataset=None,
            tokenizer=self.tokenizer
        )

    def train(self):
        self.trainer.train()
