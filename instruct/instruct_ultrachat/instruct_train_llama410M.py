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
import glob
import numpy as np


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
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"{config.model_name} has {total_params / 1e6} Million params.")


def loss_fn(logits, targets):
    # shift the targets such that output n predicts token n+1
    logits = logits[..., :-1, :].contiguous()
    targets = targets[..., 1:].contiguous()
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
    return loss


class CFG:
    # Hyperparameters
    # model
    llama_size = '410M'
    model_name = f'Llama_{llama_size}'

    # dataset
    tokenizer_path = '/data/llm_models/llama_tokenizer/tokenizer.model'
    train_datas = [
        dict(
            raw_path='/data/data/ultrachat_data/ultrachat_material_release_230412.json',
        ),
        dict(
            raw_path='/data/data/ultrachat_data/ultrachat_material_release_230417.json',
        ),
        dict(
            raw_path='/data/data/ultrachat_data/ultrachat_existent_material_release_230420.json',
        ),
        dict(
            raw_path='/data/data/ultrachat_data/ultrachat_release_230407.json',
        ),
    ]

    block_size = 1024

    seed = 1234
    learning_rate = 1e-3
    batch_size = 4
    micro_batch_size = 2
    max_iters = 6000000  # num_epochs * (epoch_size // micro_batch_size) // devices
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    decay_lr = True
    warmup_iters = 2000
    lr_decay_iters = max_iters
    min_lr = 6e-5

    ckpt_dir = '/data/output/expGPT_instruct_ultrachat_experiment3'
    save_interval = 20000
    eval_interval = 1000
    eval_iters = 100
    log_interval = 20

    # resume = '/data/output/expGPT_pretrain_gpt2Dataset_experiment1/iter-009999-ckpt.pth'
    # resume = '/data/output/expGPT_pretrain_gpt2Dataset_experiment1/iter-00024999-ckpt.pth'
    # resume = '/data/output/expGPT_pretrain_gpt2Dataset_experiment1/iter-00259999-ckpt.pth'
    # resume = '/data/output/expGPT_pretrain_gpt2Dataset_experiment1/iter-00459999-ckpt.pth'
    # resume = '/data/output/expGPT_instruct_ultrachat_experiment3/iter-01054999-ckpt.pth'
    resume = None


def train(gpu, args):
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
    logger.info('############# rank:{}/worldsize:{}'.format(rank, world_size))

    logger.remove()
    if rank == 0:
        logger.add(sys.stderr, level="INFO")
    else:
        logger.add(sys.stderr, level="ERROR")

    logger.info(f'##################### training task: start ########################')

    cfg = CFG()
    logger.info(f'exp_save_path: {cfg.ckpt_dir}')
    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    logger.info(f'\n{"#" * 30} {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())} {"#" * 30}\n')
    cfg_dict = {x: getattr(CFG(), x) for x in dir(CFG()) if not x.startswith('__')}
    logger.info(f'CFG: {cfg_dict}')

    from utils.fix_seed import seed_everything
    seed_everything(cfg.seed)

    ################## DATASET #################
    from tokenizer_team.llama_tokenizer import Tokenizer
    tokenizer_path = Path(cfg.tokenizer_path)
    tokenizer = Tokenizer(tokenizer_path)

    from instruct.instruct_ultrachat.ultrachat_dataset import load_raw_data, PromptIterableDataset
    raw_dataset = load_raw_data([d['raw_path'] for d in cfg.train_datas])
    dataset = PromptIterableDataset(raw_dataset,
                                    tokenizer=tokenizer,
                                    max_seq_length=cfg.block_size,
                                    teacher_forcing=True,
                                    rank_id=rank,
                                    world_size=world_size,
                                    )
    logger.info(f'dataset_len: {len(dataset)}')
    train_dataloader = DataLoader(dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=False,
                                  num_workers=2,
                                  pin_memory=False,
                                  drop_last=True,
                                  collate_fn=dataset.collate_fn)
    # print(f'n_batch: {len(train_dataloader)}')

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

    print_model_size(model=model, config=cfg, rank=dist.get_rank())

    model.train()
    model.cuda()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        foreach=False,
    )

    ################ resume from checkpoint #################
    # assign resume
    if cfg.resume is None:
        ckpt_fpath_list = glob.glob(f'{cfg.ckpt_dir}/iter-*.pth')
        if len(ckpt_fpath_list) > 0:
            modify_time_list = [os.path.getmtime(fpath) for fpath in ckpt_fpath_list]
            latest_fpath_idx = np.argmax(modify_time_list)
            latest_ckpt_fpath = ckpt_fpath_list[int(latest_fpath_idx)]
            logger.info(f'latest_ckpt_fpath: {latest_ckpt_fpath}')
            cfg.resume = latest_ckpt_fpath
    if cfg.resume is not None:
        checkpoint = torch.load(cfg.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        prev_iter = checkpoint['iter_num']
        logger.info(f'prev_training_config: {checkpoint["config"]}')
        del checkpoint
    else:
        prev_iter = 0

    # DDP
    model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                      # broadcast_buffers=False,
                                                      # find_unused_parameters=True,
                                                      device_ids=[dist.get_rank()])

    # compile = True
    # if compile:
    #     model = torch.compile(model)

    devices = world_size
    process_batch_size = cfg.batch_size // devices
    grad_accum_steps = process_batch_size // cfg.micro_batch_size
    logger.info(f'grad_accum_steps: {grad_accum_steps}')

    iter_num = prev_iter

    step_time = 0.0
    tokens = 0
    tokens_sec = 0.0
    prev_t1 = time.time()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    for input_ids, targets in train_dataloader:
        t0 = time.time()

        iter_num += 1

        # train_data = train_data.cuda(non_blocking=True)
        # print(train_data.shape)

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, cfg) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # input_ids = train_data[:, 0: cfg.block_size].contiguous()
        # targets = train_data[:, 1: cfg.block_size + 1].contiguous()
        input_ids = input_ids.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)

        is_accumulating = (iter_num + 1) % grad_accum_steps != 0

        # with fabric.no_backward_sync(model, enabled=is_accumulating):
        with torch.cuda.amp.autocast():
            # logits = model(input_ids)
            # loss = torch.nn.functional.cross_entropy(
            #     logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            # )
            logits = model(input_ids)
            loss = loss_fn(logits, targets)
        if rank != -1:
            loss *= world_size
        # fabric.backward(loss / grad_accum_steps)
        # loss.backward()
        scaler.scale(loss).backward()

        t1 = time.time()

        if not is_accumulating:
            # fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

            # optimizer.step()
            # optimizer.zero_grad()

            # TODO add grad clip

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            t1 = time.time()

            # if val_dataloader is not None and step_count % eval_interval == 0:
            #     val_loss = validate(fabric, model, val_dataloader)
            #     fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            #     fabric.barrier()
            #     fabric.log_dict(
            #         {"iter": iter_num, "val_loss": val_loss, "step": step_count, "lr": lr}
            #     )
            #
            if dist.get_rank() in [0, -1] and (iter_num + 1) % cfg.save_interval == 0:
                checkpoint = {
                    'model': model.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'model_args': model_args,
                    'iter_num': iter_num,
                    # 'best_val_loss': best_val_loss,
                    'config': cfg_dict,
                }
                save_path = os.path.join(cfg.ckpt_dir, f"iter-{iter_num:08d}-ckpt.pth")
                logger.info(f"saving checkpoint to {save_path}")
                torch.save(checkpoint, save_path)

        dt = t1 - t0

        tokens += cfg.micro_batch_size * cfg.block_size
        step_time += t1 - prev_t1
        prev_t1 = t1

        if dist.get_rank() in [0, -1] and iter_num % cfg.log_interval == 0:
            tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"

            # fabric.log_dict(
            #     {"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr}
            # )
            logger.info(
                f"iter {iter_num}: loss {loss.item():.4f}, time: {dt * 1000:.2f}ms, speed: {tokens_sec_str} toks/device"
            )

        if not is_accumulating:
            tokens = 0
            step_time = 0.0

        if iter_num > cfg.max_iters:
            break

    torch.cuda.empty_cache()


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
        mp.spawn(train, nprocs=args.gpus, args=(args,))
    except KeyboardInterrupt or Exception:
        print('### catch CTRL+C operation, now destroy process group')
        dist.destroy_process_group()
        torch.cuda.empty_cache()
    #########################################################