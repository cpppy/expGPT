import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader

import time
import math
from pathlib import Path

from loguru import logger
from tqdm import tqdm

import sys

import torch.distributed as dist
import argparse
import torch.multiprocessing as mp


# learning rate decay scheduler (cosine with warmup)
def get_lr(it, cfg):
    # 1) linear warmup for warmup_iters steps
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / cfg.warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - cfg.warmup_iters) / (cfg.lr_decay_iters - cfg.warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def print_model_size(model, config, rank: int = 0) -> None:
    """
    Print model name, the number of trainable parameters and initialization time.

    Args:
        model: The PyTorch model.
        model_name (str): Name of the model.
        init_time_start (float): Initialization start time.
        init_time_end (float): Initialization end time.
        rank (int, optional): Current process's rank. Defaults to 0.
    """
    if rank == 0:
        print(f"--> Model {config.model_name}")
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n--> {config.model_name} has {total_params / 1e6} Million params\n")



class CFG:
    # Hyperparameters
    # model
    llama_size = '410M'
    model_name = f'Llama_{llama_size}'

    # dataset
    tokenizer_path = '/data/expGPT/tokenizer_team/llama_tokenizer/tokenizer.model'
    train_datas = [
        dict(
            data_dir=f'/data/data/llm_data/pretrain_{dataset_tag}',
            prefix=f'{dataset_tag}.train',
        )
        for dataset_tag in [
            'webtext',
            # 'small-117M',
            # 'small-117M-k40',
            # 'medium-345M',
            # 'medium-345M-k40',
            # 'large-762M',
            # 'large-762M-k40',
            # 'xl-1542M',
            # 'xl-1542M-k40',
        ]
    ]

    block_size = 512

    seed = 1234
    learning_rate = 3e-4
    batch_size = 64
    micro_batch_size = 32
    max_iters = 100000000  # num_epochs * (epoch_size // micro_batch_size) // devices
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    decay_lr = True
    warmup_iters = 2000
    lr_decay_iters = max_iters
    min_lr = 6e-5

    ckpt_dir = '/data/output/expGPT_pretrain_gpt2Dataset_experiment1'
    save_interval = 5000
    eval_interval = 1000
    eval_iters = 100
    log_interval = 20

    # resume = '/data/output/expGPT_pretrain_gpt2Dataset_experiment1/iter-009999-ckpt.pth'
    # resume = '/data/output/expGPT_pretrain_gpt2Dataset_experiment1/iter-00024999-ckpt.pth'
    # resume = '/data/output/expGPT_pretrain_gpt2Dataset_experiment1/iter-00259999-ckpt.pth'
    # resume = '/data/output/expGPT_pretrain_gpt2Dataset_experiment1/iter-00279999-ckpt.pth'
    resume = None
    # resume = '/data/output/expGPT_pretrain_gpt2Dataset_experiment1/iter-00094999-ckpt.pth'

def train():
    # local_rank = gpu
    # rank = args.nr * args.gpus + gpu
    # world_size = args.world_size
    # torch.distributed.init_process_group(backend='nccl',
    #                                      init_method='env://',
    #                                      rank=rank,
    #                                      world_size=world_size
    #                                      )
    #
    # torch.cuda.set_device(local_rank)
    # rank = dist.get_rank()
    # world_size = dist.get_world_size()
    # print('############# rank:{}/worldsize:{}'.format(rank, world_size))

    logger.info(f'##################### training task: start ########################')

    cfg = CFG()
    logger.info(f'exp_save_path: {cfg.ckpt_dir}')
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    logger.info(f'\n{"#" * 30} {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {"#" * 30}\n')
    cfg_dict = {x: getattr(CFG(), x) for x in dir(CFG()) if not x.startswith('__')}
    logger.info(f'CFG: {cfg_dict}')

    ################## DATASET #################
    from tokenizer_team.llama_tokenizer import Tokenizer
    tokenizer_path = Path(cfg.tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)

    from pretrain.datasets.dataset_webtext import build_dataset
    from pretrain.datasets.packed_dataset import PackedDataset, CombinedDataset
    datasets = [
        build_dataset(data_dir=d['data_dir'],
                      prefix=d['prefix'],
                      block_size=cfg.block_size + 1,
                      shuffle=True,
                      # world_size=world_size,
                      # global_rank=rank,
                      seed=cfg.seed)
        for d in cfg.train_datas
    ]

    combined_dataset = CombinedDataset(datasets, seed=cfg.seed)

    # train_dataloader = DataLoader(combined_dataset,
    #                               batch_size=cfg.batch_size,
    #                               shuffle=False,
    #                               pin_memory=False)
    # print(f'n_batch: {len(train_dataloader)}')

    data_dir = '/data/data/llm_data/raw_data_gpt2'
    # data_files = glob.glob(f'{data_dir}/*.jsonl')

    from datasets import load_dataset
    gpt2_dataset = load_dataset(
        path=data_dir,
        # extension='jsonl',
        # data_files=data_files,
        cache_dir='/data/data/cache',
        # **dataset_args,
    )

    data_module = dict(train_dataset=gpt2_dataset, eval_dataset=None)

    # logger.info(f'[DATASET] n_train_samples: {len(combined_dataset)}')

    # train_sampler = torch.utils.data.distributed.DistributedSampler(combined_dataset, shuffle=True, rank=rank)
    # train_dataloader = torch.utils.data.DataLoader(dataset=combined_dataset,
    #                                                shuffle=False,
    #                                                sampler=train_sampler,
    #                                                batch_size=cfg.micro_batch_size,
    #                                                num_workers=cfg.n_workers,
    #                                                # collate_fn=dataset.collate_fn
    #                                                )
    # logger.info(f'[DATASET] n_train_bs in rank_{dist.get_rank()}: {len(train_dataloader)}')

    ################## MODEL ###################
    from model_design.lit_llama_model import LLaMA, LLaMAConfig
    config = LLaMAConfig.from_name(cfg.llama_size)
    config.block_size = cfg.block_size
    config.vocab_size = tokenizer.vocab_size

    model = LLaMA(config)
    model.apply(model._init_weights)
    # torch.set_default_dtype(torch.float32)

    # print_model_size(model=model, config=cfg, rank=dist.get_rank())

    model.train()
    model.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        # weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        foreach=False,
    )

    ################ resume from checkpoint #################
    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        prev_iter = checkpoint['iter_num']
        logger.info(f'prev_training_config: {checkpoint["config"]}')
        del checkpoint
    else:
        prev_iter = 0

    # # DDP
    # model = torch.nn.parallel.DistributedDataParallel(module=model,
    #                                                   # broadcast_buffers=False,
    #                                                   # find_unused_parameters=True,
    #                                                   device_ids=[dist.get_rank()])

    # compile = True
    # if compile:
    #     model = torch.compile(model)

    # devices = world_size
    devices = 1
    process_batch_size = cfg.batch_size // devices
    grad_accum_steps = process_batch_size // cfg.micro_batch_size
    logger.info(f'grad_accum_steps: {grad_accum_steps}')

    iter_num = prev_iter

    step_time = 0.0
    tokens = 0
    tokens_sec = 0.0
    prev_t1 = time.time()

    # scaler = torch.cuda.amp.GradScaler(enabled=True)
    #
    # while True:
    #     for train_data in train_dataloader:
    #         t0 = time.time()
    #
    #         iter_num += 1
    #
    #         train_data = train_data.cuda(non_blocking=True)
    #         # print(train_data.shape)
    #
    #         # determine and set the learning rate for this iteration
    #         lr = get_lr(iter_num, cfg) if cfg.decay_lr else cfg.learning_rate
    #         for param_group in optimizer.param_groups:
    #             param_group["lr"] = lr
    #
    #         input_ids = train_data[:, 0: cfg.block_size].contiguous()
    #         targets = train_data[:, 1: cfg.block_size + 1].contiguous()
    #
    #         is_accumulating = (iter_num + 1) % grad_accum_steps != 0
    #
    #         # with fabric.no_backward_sync(model, enabled=is_accumulating):
    #         with torch.cuda.amp.autocast():
    #             logits = model(input_ids)
    #             loss = torch.nn.functional.cross_entropy(
    #                 logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
    #             )
    #         if rank != -1:
    #             loss *= world_size
    #         # fabric.backward(loss / grad_accum_steps)
    #         # loss.backward()
    #         scaler.scale(loss).backward()
    #
    #         t1 = time.time()
    #
    #         if not is_accumulating:
    #             # fabric.clip_gradients(model, optimizer, max_norm=grad_clip)
    #
    #             # optimizer.step()
    #             # optimizer.zero_grad()
    #
    #             # TODO add grad clip
    #
    #             scaler.step(optimizer)
    #             scaler.update()
    #             optimizer.zero_grad(set_to_none=True)
    #
    #             t1 = time.time()
    #
    #             # if val_dataloader is not None and step_count % eval_interval == 0:
    #             #     val_loss = validate(fabric, model, val_dataloader)
    #             #     fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
    #             #     fabric.barrier()
    #             #     fabric.log_dict(
    #             #         {"iter": iter_num, "val_loss": val_loss, "step": step_count, "lr": lr}
    #             #     )
    #             #
    #             if dist.get_rank() in [0, -1] and (iter_num + 1) % cfg.save_interval == 0:
    #                 checkpoint = {
    #                     'model': model.module.state_dict(),
    #                     'optimizer': optimizer.state_dict(),
    #                     # 'model_args': model_args,
    #                     'iter_num': iter_num,
    #                     # 'best_val_loss': best_val_loss,
    #                     'config': cfg_dict,
    #                 }
    #                 save_path = os.path.join(cfg.ckpt_dir, f"iter-{iter_num:08d}-ckpt.pth")
    #                 logger.info(f"saving checkpoint to {save_path}")
    #                 torch.save(checkpoint, save_path)
    #
    #         dt = t1 - t0
    #
    #         tokens += cfg.micro_batch_size * cfg.block_size
    #         step_time += t1 - prev_t1
    #         prev_t1 = t1
    #
    #         if dist.get_rank() in [0, -1] and iter_num % cfg.log_interval == 0:
    #             tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"
    #
    #             # fabric.log_dict(
    #             #     {"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr}
    #             # )
    #             logger.info(
    #                 f"iter {iter_num}: loss {loss.item():.4f}, time: {dt * 1000:.2f}ms, speed: {tokens_sec_str} toks/device"
    #             )
    #
    #         if not is_accumulating:
    #             tokens = 0
    #             step_time = 0.0
    #
    #         # if iter_num > cfg.max_iters:
    #         #     break
    #
    # torch.cuda.empty_cache()

    from dataclasses import dataclass, field
    import transformers
    from transformers import Trainer
    from typing import Optional

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

    training_args = TrainingArguments(
        bf16=True,
        max_steps=1000,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        evaluation_strategy='no',
        save_strategy='steps',
        save_steps=1000,
        save_total_limit=1000,
        learning_rate=1e-5,
        weight_decay=0.1,
        adam_beta1=0.1,
        adam_beta2=0.95,
        warmup_ratio=0.01,
        lr_scheduler_type='cosine',
        logging_steps=1,
        report_to='none',
        # gradient_checkpointing=True,
        output_dir="output_llama",
        log_level='debug',
        # deepspeed="./ds_config_zero2.json",
    )

    # Start trainner
    trainer = Trainer(
        model=model, tokenizer=tokenizer, args=training_args, **data_module
    )

    trainer.train()
    trainer.save_state()


if __name__ == '__main__':

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
    # n_cuda_device = torch.cuda.device_count()
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--nodes', default=1,
    #                     type=int, metavar='N')
    # parser.add_argument('-g', '--gpus', default=n_cuda_device, type=int,
    #                     help='number of gpus per node')
    # parser.add_argument('-nr', '--nr', default=0, type=int,
    #                     help='ranking within the nodes')
    # args = parser.parse_args()
    #
    # # #########################################################
    # args.world_size = args.gpus * args.nodes
    # os.environ['MASTER_ADDR'] = '127.0.0.1'
    # os.environ['MASTER_PORT'] = '2001'
    #
    # try:
    #     mp.spawn(train, nprocs=args.gpus, args=(args,))
    # except KeyboardInterrupt or Exception:
    #     print('### catch CTRL+C operation, now destroy process group')
    #     dist.destroy_process_group()
    #     torch.cuda.empty_cache()
    # #########################################################

    train()
