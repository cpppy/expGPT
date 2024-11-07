# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import os
import sys

sys.path.append('../../')

from typing import List, Union

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader

import time
import math
from pathlib import Path

from tqdm import tqdm

import sys

import torch.distributed as dist
import argparse
import torch.multiprocessing as mp

import fire
import torch
import transformers
from datasets import load_dataset
import os.path as osp
from tqdm import tqdm

# Unused imports removed
# from utils import fsdp_auto_wrap_policy
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    default_data_collator,
    BitsAndBytesConfig
)
import torch.distributed as dist

# Unused imports removed
from utils.train_utils import (
    set_tokenizer_params,
    # train,
    evaluation,
    freeze_transformer_layers,
    check_frozen_layers_peft_model,
    setup,
    setup_environ_flags,
    cleanup,
    clear_gpu_cache,
    get_parameter_dtypes,
    print_model_size,
    get_policies
)
from recon.finetune_v2.train_utils_modify import train

from utils.dataset_utils import get_preprocessed_dataset

from utils.config_utils import (
    update_config,
    generate_peft_config,
    generate_dataset_config,
)
from peft import get_peft_model, TaskType, prepare_model_for_int8_training
import configs
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
)
from torch.utils.data import DistributedSampler
import policies
from policies import AnyPrecisionAdamW
from configs import fsdp_config, train_config
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

import torch
import torch.cuda.nccl as nccl
import torch.distributed as dist
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

import time
import json


def main_proc(gpu, args):
    local_rank = gpu
    rank = args.nr * args.gpus + gpu
    world_size = args.world_size
    torch.distributed.init_process_group(backend='nccl',
                                         init_method='env://',
                                         rank=rank,
                                         world_size=world_size
                                         )

    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    print('############# rank:{}/worldsize:{}'.format(rank, world_size))

    logger.info(f'##################### training task: start ########################')

    ############################ CONFIG ##########################
    kwargs = dict(
        dataset='alpaca_dataset',
        use_peft=True,
        peft_method='lora',
        quantization=True,
        # model_name='/data/Meta-Llama-3-8B-Instruct',
        model_name='/data/data/bloomz-560m',
        output_dir='/data/output/bloomz560m_ft_lora_function_call_llama3v1.1_v4',
        data_path='../ft_datasets/alpaca_data.json',
        batch_size_training=32,
        micro_batch_size=4,
        num_epochs=1,
        use_fp16=False,
        one_gpu=False,
        enable_fsdp=True,
    )

    # Update the configuration for the training and sharding process
    update_config((train_config, fsdp_config), **kwargs)
    # update_config((fsdp_config,), **kwargs)

    # train_config = {
    #     "batch_size_training": 32,
    #     "dataset": "alpaca_dataset",
    #     "dist_checkpoint_folder": "fine-tuned",
    #     "dist_checkpoint_root_folder": "PATH/to/save/FSDP/model",
    #     "enable_fsdp": False,
    #     "freeze_layers": False,
    #     "gamma": 0.85,
    #     "lr": 0.0001,
    #     "micro_batch_size": 4,
    #     "mixed_precision": True,
    #     "model_name": "/data/data/bloomz-560m",
    #     "num_epochs": 1,
    #     "num_freeze_layers": 1,
    #     "num_workers_dataloader": 1,
    #     "one_gpu": False,
    #     "output_dir": "/data/output/bloomz560m_ft_lora_function_call_llama3v1.1_v4",
    #     "peft_method": "lora",
    #     "quantization": True,
    #     "run_validation": True,
    #     "save_model": True,
    #     "save_optimizer": False,
    #     "seed": 42,
    #     "use_fp16": True,
    #     "use_peft": True,
    #     "val_batch_size": 1,
    #     "weight_decay": 0.0
    # }


    # print(f'exp_save_path: {CFG.ckpt_dir}')
    logger.info(f'\n{"#" * 30} {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {"#" * 30}\n')
    cfg_dict = {x: val for x, val in train_config.__dict__.items() if not x.startswith('__')}
    logger.info(f'###################### {train_config.__name__} ########################')
    logger.info(json.dumps(cfg_dict, sort_keys=True, indent=4, separators=(', ', ': '), ensure_ascii=False))

    cfg_dict = {x: getattr(fsdp_config(), x) for x in fsdp_config.__dict__.keys() if not x.startswith('__')}
    logger.info(f'###################### {fsdp_config.__name__} ########################')
    for k, v in cfg_dict.items():
        logger.info(f'{k}: {v}')

    ############################# MODEL ##############################

    # Set the seeds for reproducibility
    torch.cuda.manual_seed(train_config.seed)
    torch.manual_seed(train_config.seed)

    # Calculate gradient accumulation steps
    gradient_accumulation_steps = train_config.batch_size_training // train_config.micro_batch_size

    # Load the tokenizer and add special tokens
    # tokenizer = LlamaTokenizer.from_pretrained(train_config.model_name)
    tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})

    # Load the pre-trained model and setup its configuration
    # model = LlamaForCausalLM.from_pretrained(
    #     train_config.model_name,
    #     load_in_8bit=True,
    #     device_map="auto",
    # )

    from transformers import BloomForCausalLM
    model = BloomForCausalLM.from_pretrained(
        train_config.model_name,
        # load_in_8bit=False,
        torch_dtype=torch.bfloat16,
        # device_map='auto',
    )

    # model = AutoModelForCausalLM.from_pretrained('/data/Meta-Llama-3-8B-Instruct',
    #                                              device_map="auto",
    #                                              torch_dtype=torch.bfloat16)

    print_model_size(model=model, config=train_config, rank=0)

    # for k, v in model.named_parameters():
    #     print(k, v.shape)
    # exit(0)

    # Prepare the model for int8 training if quantization is enabled
    if train_config.quantization:
        model = prepare_model_for_int8_training(model)

    # if train_config.use_peft:
    #     peft_config = generate_peft_config(train_config, kwargs)
    #     model = get_peft_model(model, peft_config)
    #     model.print_trainable_parameters()
    #
    # if not train_config.quantization and not train_config.enable_fsdp:
    #     model.to("cuda")

    from peft import LoraConfig, TaskType, get_peft_model
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        # target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        # target_modules=["q_proj", "k_proj", "v_proj"],
        target_modules=['query_key_value', 'dense'],
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )

    # if init new peft model
    model = get_peft_model(model, lora_config)

    # # load from prev peft model
    # from peft import PeftModel
    # model = PeftModel.from_pretrained(model, train_config.output_dir, is_trainable=True)

    model.print_trainable_parameters()
    model.train()

    ######################### DATASET #########################

    dataset_config = generate_dataset_config(train_config, kwargs)
    print(f'dataset_config: {dataset_config()}')

    # Load and preprocess the dataset for training and validation
    # dataset_train = get_preprocessed_dataset(
    #     tokenizer,
    #     dataset_config,
    #     split="train",
    # )
    from recon.ft_func.ft_datasets.function_call_llama3_dataset import FunctionCallingDataset
    dataset_train = FunctionCallingDataset(
        data_path='/data/data/agent_tuning_data/function-calling-llama-3-format-v1.1/train_filtered.jsonl',
        tokenizer=tokenizer,
        max_words=512
    )

    print(f"--> Training Set Length = {len(dataset_train)}")

    # dataset_val = get_preprocessed_dataset(
    #     tokenizer,
    #     dataset_config,
    #     split="test",
    # )
    dataset_val = FunctionCallingDataset(
        data_path='/data/data/agent_tuning_data/function-calling-llama-3-format-v1.1/valid_filtered.jsonl',
        tokenizer=tokenizer,
        max_words=512,
    )

    print(f"--> Validation Set Length = {len(dataset_val)}")

    # exit(0)

    # train_sampler = None
    # val_sampler = None
    from torch.utils.data.distributed import DistributedSampler
    train_sampler = DistributedSampler(dataset_train,
                                       num_replicas=world_size,
                                       rank=rank,
                                       shuffle=True,
                                       seed=12,
                                       drop_last=True)
    val_sampler = DistributedSampler(dataset_val,
                                     num_replicas=world_size,
                                     rank=rank,
                                     shuffle=False,
                                     seed=12,
                                     drop_last=True)

    # Create DataLoaders for the training and validation dataset
    train_dataloader = torch.utils.data.DataLoader(dataset_train,
                                                   batch_size=train_config.batch_size_training,
                                                   num_workers=train_config.num_workers_dataloader,
                                                   pin_memory=True,
                                                   shuffle=False,
                                                   sampler=train_sampler if train_sampler else None,
                                                   drop_last=True,
                                                   collate_fn=default_data_collator,
                                                   )

    if train_config.run_validation:
        eval_dataloader = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=train_config.val_batch_size,
            num_workers=train_config.num_workers_dataloader,
            pin_memory=True,
            sampler=val_sampler if val_sampler else None,
            drop_last=True,
            collate_fn=default_data_collator,
        )

    # Initialize the optimizer and learning rate scheduler
    if fsdp_config.pure_bf16 and fsdp_config.optimizer == "anyprecision":
        optimizer = AnyPrecisionAdamW(
            model.parameters(),
            lr=train_config.lr,
            momentum_dtype=torch.bfloat16,
            variance_dtype=torch.bfloat16,
            use_kahan_summation=False,
        )
    else:
        optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=0.0,
        )
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

    # DDP
    model.to(local_rank)
    model = FSDP(model)
    # model = torch.nn.parallel.DistributedDataParallel(module=model,
    #                                                   # broadcast_buffers=False,
    #                                                   # find_unused_parameters=True,
    #                                                   device_ids=[local_rank],
    #                                                   # output_device=local_rank
    #                                                   )

    if local_rank == 0:
        for k, v in model.named_parameters():
            print(f'k: {k}, v: {v.shape}, device: {v.device}')

    # Start the training process
    results = train(
        model,
        train_dataloader,
        eval_dataloader,
        tokenizer,
        optimizer,
        scheduler,
        gradient_accumulation_steps,
        train_config,
        None,
        None,
        None,
    )
    [print(f'Key: {k}, Value: {v}') for k, v in results.items()]

    # from transformers import DataCollatorForSeq2Seq, TrainingArguments, Trainer
    # args = TrainingArguments(
    #     output_dir="/data/output/llama3",
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=4,
    #     logging_steps=10,
    #     num_train_epochs=3,
    #     save_steps=100,
    #     learning_rate=1e-4,
    #     save_on_each_node=True,
    #     gradient_checkpointing=True
    # )
    #
    # trainer = Trainer(
    #     model=model,
    #     args=args,
    #     # train_dataset=tokenized_id,
    #     # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    #     train_dataset=dataset_train,
    #     eval_dataset=dataset_val,
    #     data_collator=default_data_collator,
    # )
    #
    # trainer.train()
    #
    # peft_model_id = "/data/output/llama3_lora"
    # os.makedirs(peft_model_id, exist_ok=True)
    # trainer.model.save_pretrained(peft_model_id)
    # tokenizer.save_pretrained(peft_model_id)


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    n_cuda_device = torch.cuda.device_count()

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=n_cuda_device, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    args = parser.parse_args()

    # #########################################################
    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '2001'

    try:
        mp.spawn(main_proc, nprocs=args.gpus, args=(args,))
    except KeyboardInterrupt or Exception:
        print('### catch CTRL+C operation, now destroy process group')
        dist.destroy_process_group()
        torch.cuda.empty_cache()
    #########################################################
