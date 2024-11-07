import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import os
import numpy as np
import inspect
from tqdm import tqdm

def main():

    ######################### Dataset #########################
    dataset = 'shakespeare'

    block_size = 1024
    batch_size = 16
    device_type = 'cpu'
    device = 'cpu'
    split = 'train'

    # data_dir = os.path.join('/data/nanoGPT/data', dataset)
    # train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    # val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    # print(f'train_data: {train_data.shape}')
    # print(f'val_data: {val_data.shape}')

    # def get_batch(split):
    #     data = train_data if split == 'train' else val_data
    #     ix = torch.randint(len(data) - block_size, (batch_size,))
    #     x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    #     y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    #     if device_type == 'cuda':
    #         # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
    #         x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    #     else:
    #         x, y = x.to(device), y.to(device)
    #     return x, y

    from recon.dataset.baike_corpus import BaikeCorpus
    dataset = BaikeCorpus(max_length=128)
    print(f'dataset_len: {len(dataset)}')

    # dataset.counter_text_length()
    # return

    train_batch_size = 16
    eval_batch_size = 4
    workers = 4
    is_training = True

    if is_training == "train":
        # define data_generator will help experiment reproducibility.
        data_generator = torch.Generator()
        data_generator.manual_seed(1234)
        data_sampler = RandomSampler(dataset, generator=data_generator)
        batch_size = train_batch_size
    else:
        data_sampler = SequentialSampler(dataset)
        batch_size = eval_batch_size

    dataloader = DataLoader(dataset=dataset,
                            sampler=data_sampler,
                            batch_size=batch_size,
                            num_workers=workers)


    ############################# MODEL ##############################
    from model import GPTConfig, GPT
    model_args = {
        'n_layer': 12,
        'n_head': 12,
        'n_embd': 768,
        'block_size': 128,
        'bias': False,
        'vocab_size': 20001,
        'dropout': 0.0
    }

    # meta_vocab_size = None
    # if meta_vocab_size is None:
    #     print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    # model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)


    ########################## OPTIMIZER ###########################
    # adamw optimizer
    learning_rate = 6e-4  # max learning rate
    max_iters = 600000  # total number of training iterations
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_clip = 1.0  # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True  # whether to decay the learning rate
    warmup_iters = 2000  # how many steps to warm up for
    lr_decay_iters = 600000  # should be ~= max_iters per Chinchilla
    min_lr = 6e-5  # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
    print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == 'cuda'
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2), **extra_args)
    print(f"using fused AdamW: {use_fused}")

    ############################# TRAIN #########################
    model.cuda()
    model.train()

    from recon.checkpoint_mgr.checkpoint_mgr import CheckpointMgr
    # ckpt_dir = '/data/output/nanoGPT_train_v1'
    ckpt_dir = '/data/output/nanoGPT_pretrain_baikecorpus_v2'
    checkpoint_op = CheckpointMgr(ckpt_dir)
    checkpoint_op.load_checkpoint(model)

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    from recon.checkpoint_mgr.metrics import AverageMeter
    m_loss = AverageMeter()
    n_epoch = 10

    for ep_i in range(n_epoch):
        print(('\n' + '%10s' * 3) % ('Epoch', 'lr', 'loss'))
        pbar = tqdm(total=len(dataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        for iter, batch in enumerate(dataloader):
            X, Y = batch
            X = X.cuda(non_blocking=True)
            Y = Y.cuda(non_blocking=True).long()
            with torch.cuda.amp.autocast():
                logits, loss = model(X, Y)

            # if (iter+1) % 1000 == 0:
            #     print(f'ep: {ep+1}, iter: {iter+1}, loss: {loss.item()}')

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            bs = X.shape[0]
            m_loss.update(loss.item(), bs)

            _lr = optimizer.param_groups[0]['lr']
            pbar.set_description(('%10s' * 2 + '%10.4g' * 1) % (
                f'{ep_i + 1}/{n_epoch}',
                round(_lr, 6),
                round(m_loss.avg, 5)))
            pbar.update(1)

            # if (iter+1) % 1000 == 0:
        checkpoint_op.save_checkpoint(model, verbose=True)












if __name__=='__main__':

    main()







