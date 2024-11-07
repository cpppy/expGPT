import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data.dataloader import DataLoader

import time
import math
from pathlib import Path


# import torch._dynamo
# torch._dynamo.config.suppress_errors = True


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


class CFG:
    # Hyperparameters
    learning_rate = 6e-4
    batch_size = 10
    micro_batch_size = 5
    max_iters = 600000  # num_epochs * (epoch_size // micro_batch_size) // devices
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0
    decay_lr = True
    warmup_iters = 2000
    lr_decay_iters = max_iters
    min_lr = 6e-5

    out_dir = "out/training"
    save_interval = 1000
    eval_interval = 1000
    eval_iters = 100
    log_interval = 1


def train():
    cfg = CFG()

    ################## DATASET #################
    from tokenizer_team.llama_tokenizer import Tokenizer
    tokenizer_path = Path('/data/llm_models/llama_tokenizer/tokenizer.model')
    tokenizer = Tokenizer(tokenizer_path)

    from pretrain.datasets.dataset_webtext import build_dataset
    from pretrain.datasets.packed_dataset import PackedDataset, CombinedDataset
    dataset = build_dataset(data_dir='/data/data/llm_data/pretrain_webtext',
                            prefix='webtext.train')
    print(f'n_data_block: {len(dataset._filenames)}')

    combined_dataset = CombinedDataset([dataset], seed=1234)
    print(f'dataset_len: {len(dataset)}')

    train_dataloader = DataLoader(combined_dataset,
                                  batch_size=cfg.batch_size,
                                  shuffle=False,
                                  pin_memory=False)
    print(f'n_batch: {len(train_dataloader)}')

    ################## MODEL ###################
    from model_design.lit_llama_model import LLaMA, LLaMAConfig
    config = LLaMAConfig.from_name("410M")
    config.block_size = 512
    config.vocab_size = tokenizer.vocab_size

    model = LLaMA(config)
    model.apply(model._init_weights)
    torch.set_default_dtype(torch.float32)

    model.train()
    model.cuda()

    # compile = True
    # if compile:
    #     model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        betas=(cfg.beta1, cfg.beta2),
        foreach=False,
    )

    devices = 1
    process_batch_size = cfg.batch_size // devices
    grad_accum_steps = process_batch_size // cfg.micro_batch_size
    print(f'grad_accum_steps: {grad_accum_steps}')

    step_count = 0

    step_time = 0.0
    tokens = 0
    tokens_sec = 0.0
    prev_t1 = time.time()

    for iter_num, train_data in enumerate(train_dataloader):
        t0 = time.time()

        train_data = train_data.cuda(non_blocking=True)

        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num, cfg) if cfg.decay_lr else cfg.learning_rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        input_ids = train_data[:, 0: model.config.block_size].contiguous()
        targets = train_data[:, 1: model.config.block_size + 1].contiguous()

        is_accumulating = (iter_num + 1) % grad_accum_steps != 0

        # with fabric.no_backward_sync(model, enabled=is_accumulating):
        logits = model(input_ids)
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
        )
        # fabric.backward(loss / grad_accum_steps)
        loss.backward()

        t1 = time.time()

        if not is_accumulating:
            # fabric.clip_gradients(model, optimizer, max_norm=grad_clip)

            optimizer.step()
            optimizer.zero_grad()
            step_count += 1

            t1 = time.time()

            # if val_dataloader is not None and step_count % eval_interval == 0:
            #     val_loss = validate(fabric, model, val_dataloader)
            #     fabric.print(f"step {iter_num}: val loss {val_loss:.4f}")
            #     fabric.barrier()
            #     fabric.log_dict(
            #         {"iter": iter_num, "val_loss": val_loss, "step": step_count, "lr": lr}
            #     )
            #
            # if step_count % save_interval == 0:
            #     fabric.print(f"Saving checkpoint to {out_dir}")
            #     save_model_checkpoint(
            #         fabric, model, os.path.join(out_dir, f"iter-{iter_num:06d}-ckpt.pth")
            #     )

        dt = t1 - t0

        tokens += cfg.micro_batch_size * model.config.block_size
        step_time += t1 - prev_t1
        prev_t1 = t1

        if iter_num % cfg.log_interval == 0:
            tokens_sec_str = f"{tokens / step_time:.0f}" if not is_accumulating else "-"

            # fabric.log_dict(
            #     {"iter": iter_num, "train_loss": loss, "step": step_count, "lr": lr}
            # )
            print(
                f"iter {iter_num}: loss {loss.item():.4f}, time: {dt * 1000:.2f}ms, speed: {tokens_sec_str} toks/s/device"
            )

        if not is_accumulating:
            tokens = 0
            step_time = 0.0

        if iter_num > cfg.max_iters:
            break


if __name__ == '__main__':
    train()
