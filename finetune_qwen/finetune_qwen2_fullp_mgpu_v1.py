from dataclasses import dataclass, field
import json
import math
import logging
import os
from typing import Dict, Optional, List
import torch
from torch.utils.data import Dataset
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
import transformers
from transformers import Trainer, GPTQConfig, deepspeed
from transformers.trainer_pt_utils import LabelSmoother
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from accelerate.utils import DistributedType

import sys
sys.path.append('..')


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )


def train():

    from utils.fix_seed import seed_everything
    seed_everything(12)

    # base_model_path = "/data/Qwen/Qwen2-0.5B-Instruct"
    base_model_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5b_fullp_prv_qa2_exp8/checkpoint-561'

    training_args = TrainingArguments(
        bf16=True,
        num_train_epochs=3,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        eval_strategy='no',
        save_strategy='steps',
        save_steps=100,
        save_total_limit=10,
        learning_rate=1e-5,
        weight_decay=0.1,
        adam_beta1=0.1,
        adam_beta2=0.95,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        report_to='none',
        gradient_checkpointing=True,
        output_dir="output_qwen2_0.5b_fullp_prv_qa2_exp8_plus",
        log_level='debug',
        deepspeed="./ds_config_zero2.json",
    )

    global local_rank
    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        # pretrained_model_name_or_path='/mnt2/expGPT/finetune_qwen/output_qwen2_0.5b_fullp_prv_qa_exp7/checkpoint-378',
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        **model_load_kwargs,
    )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # from finetune_qwen2.datasets.load_dataset_med import load_dataset
    # data_module = load_dataset(tokenizer=tokenizer)
    from finetune_qwen.datasets.load_dataset_private_qa import load_dataset
    data_module = load_dataset(tokenizer=tokenizer)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()


if __name__ == "__main__":
    train()