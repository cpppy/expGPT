import os

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import sys

sys.path.append('..')

from recon.dataset.data_augment_v4 import TrainTransform

from recon.checkpoint_mgr.metrics import AverageMeter

from loguru import logger
from tqdm import tqdm

import torch.distributed as dist
import argparse
import torch.multiprocessing as mp


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
    print('############# rank:{}/worldsize:{}'.format(rank, world_size))

    logger.info(f'##################### training task: start ########################')

    from recon.utils.fix_seed import seed_everything
    seed_everything(1234)

    update_cache = False
    img_size = (2048, 2048)
    batch_size = 6
    n_workers = 4

    ##################### DATASET #####################
    from recon.utils.dist_util import torch_distributed_zero_first

    from recon.dataset_lbs.dataset_yimu_detail_with_ignore import YimuDataset
    if dist.get_rank() not in [0, -1]:
        update_cache = False
    with torch_distributed_zero_first(rank):
        dataset = YimuDataset(
            paths=[
            ],
            img_size=img_size,
            preproc=None,
            neg_k=2.0,
            lbs_mode='only',
        )

    # from yolox.data.datasets import MosaicDetection
    from recon.dataset.mosaicdetection import MosaicDetection
    dataset = MosaicDetection(
        dataset=dataset,
        mosaic=False,
        img_size=img_size,

        # aug_v4
        preproc=TrainTransform(
            max_labels=50,
            horizontal_flip_prob=0.5,
            vertical_flip_prob=0.5,
            # hsv_prob=0.5,
            # affine params
            degrees=30.0,
            translate=0.2,
            scales=0.1,
            # shear=10.0,
        ),

        degrees=15.0,
        translate=0.1,
        mosaic_scale=(0.8, 1.2),
        mixup_scale=(0.5, 1.5),
        shear=2.0,
        enable_mixup=False,
        mosaic_prob=0.1,
        mixup_prob=1.0,
    )

    logger.info(f'[DATASET] n_train_samples: {len(dataset)}')

    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True, rank=rank)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,
                                             shuffle=False,
                                             sampler=train_sampler,
                                             batch_size=batch_size,
                                             num_workers=n_workers,
                                             # collate_fn=dataset.collate_fn
                                             )

    logger.info(f'[DATASET] n_train_bs in rank_{dist.get_rank()}: {len(dataloader)}')

    eval_each_epoch = True
    if dist.get_rank() in [0, -1] and eval_each_epoch:
        ### val loader
        ##################### DATASET #####################
        val_dataset = YimuDataset(
            paths=[
            ],
            img_size=img_size,
            preproc=TrainTransform(
                max_labels=50,
            ),
            lbs_mode='prior'
        )

        ######################### visual ##########################
        logger.info(f'[EVAL] dataset_length: {len(dataset)}')

        val_dataloader = DataLoader(dataset=val_dataset,
                                    batch_size=batch_size * 2,
                                    num_workers=2,
                                    sampler=None,
                                    batch_sampler=None,
                                    shuffle=False,
                                    pin_memory=False,
                                    drop_last=False)

        from recon.evaluate.val_draft import Eval
        eval_tool = Eval(dataset=val_dataset, val_loader=val_dataloader)
    else:
        eval_tool = None

    #################### MODEL #####################
    from recon.model_design.yolox_swinT_ds8x import yolox_swinT
    model = yolox_swinT(num_classes=1, phi='s')
    from recon.model_design_swinT.yolo_training import weights_init
    weights_init(model)

    ################### CHECKPOINT ##################
    from recon.checkpoint_mgr.checkpoint_mgr import CheckpointMgr
    ckpt_dir = '/data/output/yolox_swinT_small_ds8x_s2048_s3_experiment42'  # lbs_mode='only', neg_k=2.0, drop ss

    checkpoint_op = CheckpointMgr(
        ckpt_dir=ckpt_dir,
        max_remain=3,
    )
    checkpoint_op.load_checkpoint(model,
                                  ckpt_fpath='/data/output/yolox_swinT_small_ds8x_s2048_s3_experiment42/backup_ckpt_mAP0.7649526596069336.pth'
                                  )

    model.to(dist.get_rank())

    freeze_pattern = [
    ]
    n_freeze = 0
    for k, v in model.named_parameters():
        # if 'emb' in k: continue
        for p in freeze_pattern:
            if p in k:
                v.requires_grad = False
                n_freeze += 1
    logger.info(f'[FREEZE] freeze_pattern: {freeze_pattern}, n_freeze_params: {n_freeze}')

    pg0, pg1, pg2 = [], [], []  # optimizer parameter groups
    for k, v in model.named_parameters():
        if not v.requires_grad:
            continue
        if k.endswith('bias'):
            pg2.append(v)  # biases
        elif '.bn' in k and k.endswith('weight'):
            pg0.append(v)  # bn_weights no_decay
        elif k.endswith('weight'):
            pg1.append(v)  # apply decay
        else:
            print(f'no_match_param_name:{k}')
            pg0.append(v)  # other_weights no_decay

    lr = 2e-4
    weight_decay = 1e-4

    nbs = 64  # nominal batch size
    accumulate = max(round(nbs / batch_size), 1)  # accumulate loss before optimizing
    weight_decay *= batch_size * accumulate / nbs  # scale weight_decay
    logger.info(f'[OPTIMIZER] accumulate = {accumulate}')
    logger.info(f"[OPTIMIZER] Scaled weight_decay = {weight_decay}")
    optimizer = torch.optim.SGD(pg0, lr=lr, momentum=0.937, nesterov=True)
    # optimizer = torch.optim.AdamW(pg0, lr=lr, weight_decay=0.0)
    optimizer.add_param_group(
        {"params": pg1, "weight_decay": weight_decay}
    )  # add pg1 with weight_decay
    optimizer.add_param_group({"params": pg2})
    logger.info('[Optimizer] groups: %g bias(no_decay), %g weight(l2_decay), %g other(no_decay)' % (
        len(pg2), len(pg1), len(pg0)))
    del pg0, pg1, pg2

    # DDP
    model = torch.nn.parallel.DistributedDataParallel(module=model,
                                                      # broadcast_buffers=False,
                                                      find_unused_parameters=True,
                                                      device_ids=[dist.get_rank()])
    model.train()
    # model.cuda()

    use_ema = False
    if use_ema and dist.get_rank() == 0:
        # EMA
        from recon.model_design.ema import ModelEMA
        ema = ModelEMA(model=model, decay=0.9999, updates=1e4)

    # SyncBatchNorm
    sync_bn = False
    if sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(dist.get_rank())
        logger.info('Using SyncBatchNorm')

    if dist.get_rank() in [0, -1]:
        m_loss = AverageMeter()
        m_conf = AverageMeter()
        m_bbox = AverageMeter()
        m_cls = AverageMeter()

    scaler = torch.cuda.amp.GradScaler(enabled=True)

    logger.info(f'[TRAIN] start training...')

    # if dist.get_rank() in [0, -1]:
    #     eval_tool(model.module)
    # exit(0)

    n_epoch = 1000
    for ep_i in range(n_epoch):
        dataset.random_sample_ids()
        if dist.get_rank() in [-1, 0]:
            print(('\n' + '%10s' * 6) % ('Epoch', 'lr', 'loss', 'conf', 'bbox', 'cls'))
            pbar = tqdm(total=len(dataloader), bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')  # progress bar
        dataloader.sampler.set_epoch(ep_i)
        for idx, batch in enumerate(dataloader):
            imgs, labels, img_infos, ids = batch
            imgs = imgs.cuda(non_blocking=True) / 255.0
            labels = labels.cuda(non_blocking=True)

            with torch.cuda.amp.autocast(enabled=True):
                output = model(imgs, labels)
            loss = output['total_loss']
            iou_loss = output['iou_loss']
            conf_loss = output['conf_loss']
            cls_loss = output['cls_loss']
            if rank != -1:
                loss *= world_size
            steps = (idx + 1 + ep_i * len(dataloader))
            scaler.scale(loss).backward()
            if steps % accumulate == 0:
                scaler.step(optimizer)  # optimizer.step
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            if dist.get_rank() in [0, -1]:
                # update metrics
                bs = imgs.shape[0]
                m_loss.update(loss.item(), bs)
                m_conf.update(conf_loss.item(), bs)
                m_bbox.update(iou_loss.item(), bs)
                m_cls.update(cls_loss.item(), bs)

                _lr = optimizer.param_groups[0]['lr']
                pbar.set_description(('%10s' * 2 + '%10.4g' * 4) % (
                    f'{ep_i + 1}/{n_epoch}',
                    round(_lr, 6),
                    round(m_loss.avg, 5),
                    round(m_conf.avg, 5),
                    round(m_bbox.avg, 5),
                    round(m_cls.avg, 5)))
                pbar.update(1)
                if use_ema:
                    ema.update(model)

        if dist.get_rank() in [0, -1]:
            pbar.close()
            checkpoint_op.save_checkpoint(model=model.module, verbose=True)

            m_loss.reset()
            m_conf.reset()
            m_bbox.reset()
            m_cls.reset()

            if eval_each_epoch:
                if not use_ema:
                    eval_metrics = eval_tool(model.module)
                else:
                    eval_metrics = eval_tool(ema.ema)
                mAP = eval_metrics[0]
                if mAP >= 0.70:
                    checkpoint_op.save_checkpoint(
                        model=ema.ema if use_ema else (model.module),
                        ckpt_fpath=f'backup_ema_mAP{mAP}.pth' if use_ema else f'backup_ckpt_mAP{mAP}.pth',
                        verbose=True)

        torch.cuda.empty_cache()
    torch.cuda.empty_cache()


if __name__ == '__main__':

    logger.remove()
    logger.add(sys.stderr, level="INFO")

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
