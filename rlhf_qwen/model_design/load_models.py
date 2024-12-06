import argparse
import math

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    SchedulerType,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.accelerator import get_accelerator

from dschat.utils.model.model_utils import create_critic_model
from dschat.utils.data.data_utils import create_prompt_dataset, DataCollatorReward
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, \
    only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible


def main():
    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    end_of_conversation_token = "<|endoftext|>"
    add_eot_token = True
    model_name_or_path = '/data/Qwen/Qwen2.5-0.5B-Instruct'
    num_padding_at_beginning = 1
    dropout = 0
    zero_stage = 0
    compute_fp32_loss = False

    ds_config = get_train_ds_config(offload=False,
                                    dtype='bf16',
                                    stage=zero_stage,
                                    enable_tensorboard=False,
                                    tb_path='./',
                                    tb_name="step2_model")
    ds_config[
        'train_micro_batch_size_per_gpu'] = 2
    ds_config[
        'train_batch_size'] = 8

    additional_special_tokens = end_of_conversation_token if add_eot_token else None
    tokenizer = load_hf_tokenizer(model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)
    rm_model = create_critic_model(model_name_or_path,
                                   tokenizer,
                                   ds_config,
                                   num_padding_at_beginning,
                                   dropout=dropout,
                                   zero_stage=zero_stage,
                                   compute_fp32_loss=compute_fp32_loss)


if __name__=='__main__':

    main()
















