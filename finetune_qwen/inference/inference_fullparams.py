from dataclasses import dataclass, field
import json
import math
import logging
import os
import sys

sys.path.append('..')

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

# import logging
# logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


'''
https://github.com/QwenLM/Qwen/blob/main/finetune.py

https://github.com/QwenLM/Qwen/blob/main/recipes/finetune/deepspeed/finetune_fullparameter_single_gpu.ipynb

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
    use_lora: bool = False


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        # default="/data/Qwen/Qwen2-0.5B-Instruct"
        # default="/data/expGPT/finetune_qwen/output_qwen2_0.5b_fullparam_private1/checkpoint-144"
        # default="/data/expGPT/finetune_qwen/output_qwen2_0.5b_fullparam_private1_f/checkpoint-144",
        default="/data/expGPT/finetune_qwen/output_qwen2_0.5b_fullparam_huatuodata_modify/checkpoint-441"
    )


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["c_attn", "c_proj", "w1", "w2"]
    )
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False


def train():
    model_args = ModelArguments()
    # training_args = TrainingArguments(
    #     bf16=True,
    #     num_train_epochs=2,
    #     per_device_train_batch_size=64,
    #     per_device_eval_batch_size=64,
    #     gradient_accumulation_steps=4,
    #     evaluation_strategy='no',
    #     save_strategy='steps',
    #     save_steps=1000,
    #     save_total_limit=1000,
    #     learning_rate=1e-5,
    #     weight_decay=0.1,
    #     adam_beta1=0.1,
    #     adam_beta2=0.95,
    #     warmup_ratio=0.0,
    #     lr_scheduler_type='cosine',
    #     logging_steps=1,
    #     report_to='none',
    #     gradient_checkpointing=True,
        # output_dir="output_qwen2_0.5b_fullparam_huatuodata",
        # output_dir="output_qwen2_0.5b_fullparam_private1",
        # output_dir="output_qwen2_0.5b_fullparam_private1_f",
        # log_level='debug',
        # deepspeed="./ds_config_zero2.json",
    # )

    # global local_rank
    #
    # # This serves for single-gpu qlora.
    # if getattr(training_args, 'deepspeed', None) and int(os.environ.get("WORLD_SIZE", 1)) == 1:
    #     training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
    #
    # local_rank = training_args.local_rank

    device_map = None
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    # if lora_args.q_lora:
    #     device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else "auto"
    #     if len(training_args.fsdp) > 0 or deepspeed.is_deepspeed_zero3_enabled():
    #         logging.warning(
    #             "FSDP or ZeRO3 are incompatible with QLoRA."
    #         )

    # is_chat_model = 'chat' in model_args.model_name_or_path.lower()
    # if (
    #         training_args.use_lora
    #         and not lora_args.q_lora
    #         and deepspeed.is_deepspeed_zero3_enabled()
    #         and not is_chat_model
    # ):
    #     raise RuntimeError("ZeRO3 is incompatible with LoRA when finetuning on base model.")

    model_load_kwargs = {
        'low_cpu_mem_usage': not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # Set RoPE scaling factor
    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        trust_remote_code=True,
    )
    config.use_cache = False

    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        # cache_dir=training_args.cache_dir,
        device_map=device_map,
        trust_remote_code=True,
        # quantization_config=GPTQConfig(
        #     bits=4, disable_exllama=True
        # )
        # if training_args.use_lora and lora_args.q_lora else None,
        **model_load_kwargs,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        # cache_dir=training_args.cache_dir,
        # model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    # tokenizer.pad_token_id = tokenizer.eod_id
    tokenizer.pad_token_id = tokenizer.eos_token_id

    tokenizer.im_start_token = '<|im_start|>'
    tokenizer.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    tokenizer.im_end_token = '<|im_end|>'
    tokenizer.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

    # if training_args.use_lora:
    #     if lora_args.q_lora or is_chat_model:
    #         modules_to_save = None
    #     else:
    #         modules_to_save = ["wte", "lm_head"]
    #     lora_config = LoraConfig(
    #         r=lora_args.lora_r,
    #         lora_alpha=lora_args.lora_alpha,
    #         target_modules=lora_args.lora_target_modules,
    #         lora_dropout=lora_args.lora_dropout,
    #         bias=lora_args.lora_bias,
    #         task_type="CAUSAL_LM",
    #         modules_to_save=modules_to_save  # This argument serves for adding new tokens.
    #     )
    #     if lora_args.q_lora:
    #         model = prepare_model_for_kbit_training(
    #             model, use_gradient_checkpointing=training_args.gradient_checkpointing
    #         )
    #
    #     model = get_peft_model(model, lora_config)
    #
    #     # Print peft trainable params
    #     model.print_trainable_parameters()
    #
    #     if training_args.gradient_checkpointing:
    #         model.enable_input_require_grads()

    # # Load data
    # data_module = make_supervised_data_module(
    #     tokenizer=tokenizer, data_args=data_args, max_len=training_args.model_max_length
    # )
    # from finetune_qwen.datasets.load_dataset import load_dataset
    # data_module = load_dataset()
    # from finetune_qwen.datasets.load_dataset_med import load_dataset
    # data_module = load_dataset()
    # from finetune_qwen.datasets.load_dataset_private import load_dataset
    # data_module = load_dataset()
    #
    #
    # # Start trainner
    # trainer = Trainer(
    #     model=model, tokenizer=tokenizer, args=training_args, **data_module
    # )
    #
    # trainer.train()
    # trainer.save_state()

    # safe_save_model_for_hf_trainer(trainer=trainer,
    #                                output_dir=training_args.output_dir,
    #                                bias=lora_args.lora_bias)

    # prompt = "Hey, are you conscious? Can you talk to me?"
    # inputs = tokenizer(prompt, return_tensors="pt")
    #
    # # Generate
    # generate_ids = model.generate(inputs.input_ids,
    #                               temperature=0.01,
    #                               top_k=10,
    #                               max_length=512)
    # print(generate_ids)
    # generate_ids = [
    #     gen[len(input):]
    #     for input, gen in zip(inputs.input_ids, generate_ids)
    # ]
    # result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    # print(result)

    messages = [
        # {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
        {"role": "system", "content": "You are a helpful assistant chatbot who is always politely and warmly."},
        # {"role": "user", "content": "Who are you?"},
        {"role": "user", "content": "伊春森工的林地面积是多少？"},
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    outputs = model.generate(
        input_ids,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.6,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    response = outputs[0][input_ids.shape[-1]:]
    print(tokenizer.decode(response, skip_special_tokens=True))

    return model, tokenizer


if __name__ == "__main__":
    train()
