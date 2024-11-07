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


# import logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


'''
https://github.com/QwenLM/Qwen/blob/main/finetune.py

https://github.com/QwenLM/Qwen/blob/main/recipes/finetune/deepspeed/finetune_fullparameter_single_gpu.ipynb

# lora
https://github.com/datawhalechina/self-llm/blob/master/models/Qwen2/05-Qwen2-7B-Instruct%20Lora%20%E5%BE%AE%E8%B0%83.md
'''


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

    base_model_path = "/data/Qwen/Qwen2-7B-Instruct"

    training_args = TrainingArguments(
        bf16=True,
        num_train_epochs=6,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=4,
        eval_strategy='no',
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=10,
        learning_rate=1e-4,
        weight_decay=0.1,
        adam_beta1=0.1,
        adam_beta2=0.95,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=10,
        report_to='none',
        # gradient_checkpointing=True,
        # output_dir="output_qwen2_7b_int8_ft_lora_huatuo2_mgpu",
        # output_dir="output_qwen2_0.5_qlora_prv_qa_exp3",
        output_dir="output_qwen2_7_qlora_prv_qa_exp6",
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
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        config=config,
        cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        quantization_config=bnb_config,
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
    # # tokenizer.pad_token_id = tokenizer.eod_id
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    #
    # tokenizer.im_start_token = '<|im_start|>'
    # tokenizer.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    # tokenizer.im_end_token = '<|im_end|>'
    # tokenizer.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    lora_config = LoraConfig(
        r=128,
        lora_alpha=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias='none',
        task_type="CAUSAL_LM",
    )
    # if lora_args.q_lora:
    #     model = prepare_model_for_kbit_training(
    #         model, use_gradient_checkpointing=training_args.gradient_checkpointing
    #     )

    # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5_qlora_prv_qa_exp4/checkpoint-759'
    lora_path = None
    if lora_path is None:
        model = get_peft_model(model, lora_config)
    else:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
    # model = get_peft_model(model, lora_path)

    # Print peft trainable params
    model.print_trainable_parameters()

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()

    # # Load data
    # data_module = make_supervised_data_module(
    #     tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    # )
    # from finetune_qwen.datasets.load_dataset import load_dataset
    # data_module = load_dataset()
    # from finetune_qwen.datasets.load_dataset_med import load_dataset
    # data_module = load_dataset()
    # from finetune_qwen.datasets.load_dataset_med_v2 import load_dataset
    # data_module = load_dataset(tokenizer=tokenizer)
    from finetune_qwen.datasets.load_dataset_private_qa import load_dataset
    data_module = load_dataset(tokenizer=tokenizer)

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()

    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                output_dir=training_args.output_dir,
    #                                bias=lora_args.lora_bias)


if __name__ == "__main__":
    train()
