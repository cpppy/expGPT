import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import deepspeed
import argparse

import math

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    SchedulerType,
    default_data_collator,
    get_scheduler,
)

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed import get_accelerator

import sys

sys.path.append('..')

# from dschat.utils.data.data_utils import create_prompt_dataset
from dschat.utils.utils import print_rank_0, to_device, save_hf_format, set_random_seed, get_all_reduce_mean, \
    get_optimizer_grouped_parameters, save_zero_three_model, load_hf_tokenizer
from dschat.utils.ds_utils import get_train_ds_config
from dschat.utils.module.lora import convert_linear_layer_to_lora, convert_lora_to_linear_layer, \
    only_optimize_lora_parameters, make_model_gradient_checkpointing_compatible
from dschat.utils.model.model_utils import create_hf_model, causal_lm_model_to_fp32_loss
from dschat.utils.perf import print_throughput

import logging

# logging.basicConfig(level=logging.INFO)
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def parse_args():
    parser = argparse.ArgumentParser(
        description=
        "Finetune a transformers model on a causal language modeling task")
    parser.add_argument('--data_path',
                        nargs='*',
                        # default=['Dahoas/rm-static'],
                        # default=["wangrui6/Zhihu-KOL"],
                        default=[
                            # "/data/data/llm_data/Zhihu-KOL",
                        ],
                        help='Path to the training dataset. Accepted format:'
                             '1) a single data path, 2) multiple datasets in the'
                             'form: dataset1-path dataset2-path ...')
    parser.add_argument('--data_split',
                        type=str,
                        default='2,4,4',
                        help='Comma-separated list of proportions for training'
                             'phase 1, 2, and 3 data. For example the split `6,2,2`'
                             'will use 60%% of data for phase 1, 20%% for phase 2'
                             'and 20%% for phase 3.')
    parser.add_argument(
        '--sft_only_data_path',
        nargs='*',
        default=[
            "/data/data/llm_data/Zhihu-KOL",
        ],
        help='Path to the dataset for only using in SFT phase.')
    parser.add_argument(
        '--data_output_path',
        type=str,
        default='/mnt2/data/dsp_data_files2',
        help=
        'Where to store the data-related files such as shuffle index. This needs to be on a local storage of a node (not on a shared storage)'
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        # default="/data/Qwen2.5-0.5B-Instruct",
        default="/data/Qwen/Qwen2.5-0.5B-Instruct",
        help=
        "Path to pretrained model or model identifier from huggingface.co/models.",
        # required=True,
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the evaluation dataloader.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=256,
        help="The maximum sequence length.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-3,
        help=
        "Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument("--weight_decay",
                        type=float,
                        default=0.,
                        help="Weight decay to use.")
    parser.add_argument("--num_train_epochs",
                        type=int,
                        default=1,
                        help="Total number of training epochs to perform.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help=
        "Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=SchedulerType,
        default="cosine",
        help="The scheduler type to use.",
        choices=[
            "linear", "cosine", "cosine_with_restarts", "polynomial",
            "constant", "constant_with_warmup"
        ],
    )
    parser.add_argument(
        "--num_warmup_steps",
        type=int,
        default=0,
        help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--output_dir",
                        type=str,
                        default='/mnt2/output/dsp_model_output2',
                        help="Where to store the model.")
    parser.add_argument("--seed",
                        type=int,
                        default=1234,
                        help="A seed for reproducible training.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        # default=0,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--gradient_checkpointing',
                        action='store_true',
                        help='Enable HF gradient checkpointing for model.')
    parser.add_argument(
        "--dropout",
        type=float,
        default=None,
        help="If dropout configured, use it. "
             "Otherwise, keep the default dropout configuration of the model.")
    # deepspeed features
    parser.add_argument('--offload',
                        action='store_true',
                        # type=bool,
                        # default=True,
                        help='Enable ZeRO Offload techniques.')
    parser.add_argument('--dtype',
                        type=str,
                        # default='fp16',
                        default='bf16',
                        choices=['fp16', 'bf16'],
                        help='Training data type')
    parser.add_argument(
        '--zero_stage',
        type=int,
        # default=0,
        # default=2,
        default=2,
        help='ZeRO optimization stage for Actor model (and clones).')
    ## LoRA for efficient training setting
    parser.add_argument("--lora_dim",
                        type=int,
                        default=0,
                        help="If > 0, use LoRA for efficient training.")
    parser.add_argument("--lora_module_name",
                        type=str,
                        default="decoder.layers.",
                        help="The scope of LoRA.")
    parser.add_argument('--only_optimize_lora',
                        action='store_true',
                        help='Only optimize the LoRA parameters.')
    parser.add_argument(
        "--lora_learning_rate",
        type=float,
        default=5e-4,
        help=
        "Initial LoRA learning rate (after the potential warmup period) to use."
    )
    ## low precision
    parser.add_argument(
        '--compute_fp32_loss',
        action='store_true',
        help='Relevant for low precision dtypes (fp16, bf16, etc.). '
             'If specified, loss is calculated in fp32.')
    ## Tensorboard logging
    parser.add_argument('--enable_tensorboard',
                        # action='store_true',
                        type=bool,
                        default=True,
                        help='Enable tensorboard logging')
    parser.add_argument('--tensorboard_path',
                        type=str,
                        default="/mnt2/output/dsp_model_output2_tensorboard")
    ## Tokenizer
    parser.add_argument(
        "--add_eot_token",
        action='store_true',
        help="Add `eot_token` as additional special token to tokenizer")
    parser.add_argument(
        "--eot_token",
        type=str,
        default="<|endoftext|>",
        help="Specify the format of the `eot_token`",
    )
    ## Print loss
    parser.add_argument('--print_loss',
                        # action='store_true',
                        type=bool,
                        default=True,
                        help='Prints loss at each step.')
    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()

    return args


'''
/home/ubuntu/.cache/huggingface/datasets/Dahoas___rm-static/

/home/ubuntu/.cache/huggingface/datasets/wangrui6___zhihu-kol

'''


def main():
    args = parse_args()

    print(f'args: {args}')
    # exit(0)

    if args.local_rank == -1:
        device = torch.device(get_accelerator().device_name())
    else:
        get_accelerator().set_device(args.local_rank)
        device = torch.device(get_accelerator().device_name(), args.local_rank)
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        # torch.distributed.init_process_group(backend='nccl')
    deepspeed.init_distributed()

    args.global_rank = torch.distributed.get_rank()

    ds_config = get_train_ds_config(offload=args.offload,
                                    dtype=args.dtype,
                                    stage=args.zero_stage,
                                    enable_tensorboard=args.enable_tensorboard,
                                    tb_path=args.tensorboard_path,
                                    tb_name="step2_model",
                                    )
    ds_config[
        'train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
    ds_config[
        'train_batch_size'] = args.per_device_train_batch_size * torch.distributed.get_world_size(
    ) * args.gradient_accumulation_steps

    ds_config['steps_per_print'] = 1e4

    # If passed along, set the training seed now.
    set_random_seed(args.seed)

    torch.distributed.barrier()

    # load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
    additional_special_tokens = args.eot_token if args.add_eot_token else None
    tokenizer = load_hf_tokenizer(args.model_name_or_path,
                                  fast_tokenizer=True,
                                  add_special_tokens=additional_special_tokens)

    # base_model_path = "/data/Qwen2.5-0.5B-Instruct"
    # tokenizer = AutoTokenizer.from_pretrained(
    #     pretrained_model_name_or_path=base_model_path,
    #     model_max_length=256,
    #     padding_side="right",
    #     use_fast=False,
    #     trust_remote_code=True,
    # )

    model_save_path = '/mnt2/output/dsp_model_output2'
    if os.path.exists(model_save_path):
        args.model_name_or_path = model_save_path
    model = create_hf_model(AutoModelForCausalLM,
                            args.model_name_or_path,
                            tokenizer,
                            ds_config,
                            dropout=args.dropout)

    # from transformers import AutoModelForCausalLM

    # model_path = '/data/Qwen/Qwen2.5-0.5B-Instruct'
    # from transformers import Qwen2Config
    # model_config = Qwen2Config.from_pretrained(model_path)
    # model = AutoModelForCausalLM.from_pretrained(model_path,
    #                                              # device_map="auto",
    #                                              # torch_dtype=torch.bfloat16,
    #                                              # device='auto',
    #                                              # quantization_config=bnb_config,
    #                                              config=model_config
    #                                              )

    # from transformers import LlamaConfig, LlamaForCausalLM
    # config = LlamaConfig(
    #     # vocab_size=16000,
    #     vocab_size=tokenizer.vocab_size,
    #     # hidden_size=512,
    #     hidden_size=896,
    #     num_hidden_layers=16,
    #     intermediate_size=1024,
    #     # intermediate_size=512,
    #     num_attention_heads=8,
    #     # bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
    #     # eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    #     # pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    #     # max_position_embeddings=2 * SEQ_LENGTH,
    # )
    #
    # model = LlamaForCausalLM(config)
    #
    # model.config.end_token_id = tokenizer.eos_token_id
    # model.config.pad_token_id = model.config.eos_token_id
    # model.resize_token_embeddings(int(
    #     8 *
    #     math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
    #
    # embed_weight = torch.load('/mnt2/output/qwen2_emb.pt')
    # for k, v in model.named_parameters():
    #     if 'embed' in k:
    #         v.data = embed_weight[k].data
    #         print(f'REPLACED [{k}] by qwen_embed_layer_w: {v.shape}')

    # if args.compute_fp32_loss:
    #     print_rank_0(
    #         f"Using model {model.__class__.__name__} with loss in fp32",
    #         args.global_rank)
    #     causal_lm_model_to_fp32_loss(model)
    #
    # if args.lora_dim > 0:
    #     model = convert_linear_layer_to_lora(model, args.lora_module_name,
    #                                          args.lora_dim)
    #     if args.only_optimize_lora:
    #         model = only_optimize_lora_parameters(model)
    #         model = make_model_gradient_checkpointing_compatible(model)

    # if torch.distributed.get_rank() in [-1, 0]:
    #     for k, v in model.named_parameters():
    #         print(k, v.shape)
    #
    for k, v in model.named_parameters():
        if 'embed_tokens' in k:
            v.requires_grad = False
            print(f'FROZEN weight: [{k}]')

    # train_phase = 1

    # # from dschat.utils.data.data_utils import create_prompt_dataset
    # from finetune_dsp.datasets_dsp.create_custom_dataset import create_prompt_dataset
    # train_dataset, eval_dataset = create_prompt_dataset(
    #     local_rank=args.local_rank,
    #     data_path=args.data_path,
    #     data_split=args.data_split,
    #     output_path=args.data_output_path,
    #     train_phase=train_phase,
    #     seed=args.seed,
    #     tokenizer=tokenizer,
    #     max_seq_len=args.max_seq_len,
    #     end_of_conversation_token=tokenizer.eos_token,
    #     sft_only_data_path=args.sft_only_data_path)

    # from finetune_dsp.datasets.load_zhihukol_adv import build_dataset
    # tokenizer.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    # tokenizer.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    # train_dataset, eval_dataset = build_dataset(tokenizer=tokenizer, max_len=args.max_seq_len)

    from finetune_dsp.dataset_cache.lmdb_dataloader import CustomInstructDataset
    train_dataset = CustomInstructDataset(
        db_path='/mnt2/data/dsp_data_files2/zhihu-kol_train_tokenizerQwen25_cache_20241126.lmdb',
        max_len=args.max_seq_len,
    )
    eval_dataset = CustomInstructDataset(
        db_path='/mnt2/data/dsp_data_files2/zhihu-kol_eval_tokenizerQwen25_cache_20241126.lmdb',
        max_len=args.max_seq_len,
    )

    logger.info(f'train_dataset_len: {len(train_dataset)}')
    logger.info(f'eval_dataset_len: {len(eval_dataset)}')

    # for s in train_dataset:
    #     print(s)
    #     for k, v in s.items():
    #         print(k, v.shape)
    #     print(f'====================== input_ids ======================')
    #     print(tokenizer.decode(s['input_ids'][s['attention_mask'] == 1], skip_special_tokens=False))
    #     print(f'====================== labels ======================')
    #     # print(tokenizer.decode(s['labels'][s['labels'] >= 0], skip_special_tokens=False))
    #     print(tokenizer.decode(s['labels'][s['attention_mask'] == 1], skip_special_tokens=False))
    #     exit(0)

    # ################# unsupervised dataset ################
    # from dschat.utils.data.data_utils import get_unsupervised_data
    # train_dataset = get_unsupervised_data(args, tokenizer)
    # for s in train_dataset:
    #     print(s)
    #     for k, v in s.items():
    #         print(k, v.shape)
    #
    #     exit(0)

    # DataLoaders creation:
    # if args.local_rank == -1:
    if torch.distributed.get_rank() == -1:
        train_sampler = RandomSampler(train_dataset)
        eval_sampler = SequentialSampler(eval_dataset)
    else:
        train_sampler = DistributedSampler(train_dataset)
        eval_sampler = DistributedSampler(eval_dataset)
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=default_data_collator,
                                  sampler=train_sampler,
                                  batch_size=args.per_device_train_batch_size,
                                  num_workers=4,
                                  pin_memory=True)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=default_data_collator,
                                 sampler=eval_sampler,
                                 batch_size=args.per_device_eval_batch_size,
                                 num_workers=4,
                                 pin_memory=True)

    def evaluation(model, eval_dataloader):
        model.eval()
        losses = 0
        pbar = tqdm(total=len(eval_dataloader), desc='[EVAL]')
        for step, batch in enumerate(eval_dataloader):
            batch['labels'] = batch['labels'].type(torch.int64)
            batch = to_device(batch, device)
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            losses += loss.float()
            pbar.update(1)
        pbar.close()

        losses = losses / len(eval_dataloader)
        try:
            losses = get_all_reduce_mean(losses)
        except:
            pass
        try:
            perplexity = torch.exp(losses).item()
        except OverflowError:
            perplexity = float("inf")
        return perplexity, losses.item()

    # Split weights in two groups, one with weight decay and the other not.
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(
        model, args.weight_decay, args.lora_learning_rate)

    AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam
    optimizer = AdamOptimizer(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              betas=(0.9, 0.95))

    ### lr_scheduler from transformers.optimizations
    # num_update_steps_per_epoch = math.ceil(
    #     len(train_dataloader) / args.gradient_accumulation_steps)
    # lr_scheduler = get_scheduler(
    #     name=args.lr_scheduler_type,
    #     optimizer=optimizer,
    #     num_warmup_steps=args.num_warmup_steps,
    #     num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
    # )

    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.8,
        patience=5000,
        threshold=0.01,
        threshold_mode='rel',
        cooldown=0,
        min_lr=1e-4,
        eps=1e-08,
    )

    model, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        args=args,
        config=ds_config,
        lr_scheduler=lr_scheduler,
        dist_init_required=True)

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # ###################### EVAL #####################
    # # Train!
    # if torch.distributed.get_rank() in [-1, 0]:
    #     print_rank_0("***** Running training *****", args.global_rank)
    #     print_rank_0(
    #         f"***** Evaluating perplexity, Epoch {0}/{args.num_train_epochs} *****",
    #         args.global_rank)
    #     perplexity, eval_loss = evaluation(model, eval_dataloader)
    #     print_rank_0(f"ppl: {perplexity}, loss: {eval_loss}", args.global_rank)

    from utils.metrics import AverageMetric
    m_loss = AverageMetric()
    # m_throughput = AverageMetric()
    for epoch in range(args.num_train_epochs):
        print_rank_0(
            f"Beginning of Epoch {epoch + 1}/{args.num_train_epochs}, Total Micro Batches {len(train_dataloader)}",
            args.global_rank)
        model.train()
        import time
        for step, batch in enumerate(train_dataloader):
            start = time.time()
            # batch['labels'] = batch['labels'].type(torch.int64)
            batch = to_device(batch, device)
            outputs = model(**batch, use_cache=False)
            loss = outputs.loss
            m_loss.update(loss.item(), batch['input_ids'].shape[0])
            if args.print_loss and step % 100 == 0:
                _lr = lr_scheduler.get_last_lr()
                print(
                    f"Epoch: {epoch}, Step: {step}, Rank: {torch.distributed.get_rank()}, lr = {round(_lr[0], 6)}, loss = {m_loss.avg}"
                )
            model.backward(loss)
            model.step()
            end = time.time()
            # m_throughput.update(
            #     batch['input_ids'] * args.max_seq_len / (end - start),
            #     batch['input_ids'].shape[0])
            if torch.distributed.get_rank() in [-1, 0] and step % 1000 == 0:
                # print_throughput(model.model, args, end - start,
                #                  args.global_rank)
                _throughput = batch['input_ids'].shape[0] * args.max_seq_len / (end - start)
                _num_samples = batch['input_ids'].shape[0] / (end - start)
                print(
                    f"Rank: {torch.distributed.get_rank()}, tokens/sec={round(_throughput, 4)}, samples/sec={round(_num_samples, 4)}"
                )

        # Evaluate perplexity on the validation set.
        print_rank_0(
            f"***** Evaluating perplexity, Epoch {epoch + 1}/{args.num_train_epochs} *****",
            args.global_rank)
        perplexity, eval_loss = evaluation(model, eval_dataloader)
        print_rank_0(f"ppl: {perplexity}, loss: {eval_loss}", args.global_rank)
        model.tput_timer.update_epoch_count()

        # save model each epoch
        if args.output_dir is not None:
            print_rank_0('saving the final model ...', args.global_rank)
            model = convert_lora_to_linear_layer(model)

            if args.global_rank == 0:
                save_hf_format(model, tokenizer, args)

            if args.zero_stage == 3:
                # For zero stage 3, each gpu only has a part of the model, so we need a special save function
                save_zero_three_model(model,
                                      args.global_rank,
                                      args.output_dir,
                                      zero_stage=args.zero_stage)

if __name__ == '__main__':
    main()




'''
python3.10 -m torch.distributed.run --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 train_qwen2_from_scratch_pretrain.py

# for 4090*2
NCCL_P2P_DISABLE="1" NCCL_IB_DISABLE="1" python3.11 -m torch.distributed.run --nproc_per_node 2 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6601 example_v3_dataslow.py

'''


