import torch
import torch.nn as nn

from dataclasses import dataclass

import deepspeed

import time

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys
sys.path.append('..')

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

from utils.metrics import LossAverage

def train():

    @dataclass
    class args:
        base_model_path="/data/Qwen2.5-0.5B-Instruct"
        output_dir='/data/output/dsp_demo_saved_2'
        train_batch_size=4

    from utils.fix_seed import seed_everything
    seed_everything(12)

    deepspeed.init_distributed()


    # base_model_path = '/data/data/babyllama-100m-2024'
    # base_model_path = '/data/data/mt0-small'

    import transformers
    # Load model and tokenizer
    model = transformers.AutoModelForCausalLM.from_pretrained(
        # pretrained_model_name_or_path=args.base_model_path,
        pretrained_model_name_or_path='/data/output/dsp_demo_saved',
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,

    )

    # from transformers import AutoModelForSeq2SeqLM
    # model = AutoModelForSeq2SeqLM.from_pretrained(base_model_path,
    #                                               torch_dtype="auto",
    #                                               device_map="auto"
    #                                               )

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=args.base_model_path,
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    # optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    from transformers.trainer_pt_utils import get_parameter_names

    # training_args = TrainingArguments(per_device_train_batch_size=4,
    #                                   **default_args
    #                                   )

    # decay_parameters = get_parameter_names(model, [nn.LayerNorm])
    # decay_parameters = [name for name in decay_parameters if "bias" not in name]
    # optimizer_grouped_parameters = [
    #     {
    #         "params": [p for n, p in model.named_parameters() if n in decay_parameters],
    #         "weight_decay": 0.1,
    #     },
    #     {
    #         "params": [p for n, p in model.named_parameters() if n not in decay_parameters],
    #         "weight_decay": 0.0,
    #     },
    # ]

    cmd_args = {}

    model_engine, optimizer, _, _ = deepspeed.initialize(args=cmd_args,
                                                         model=model,
                                                         model_parameters=model.parameters(),
                                                         # config='./ds_config_zero2.json'
                                                         config='./dsp_config.json',
                                                         )

    print(f'model_device: {model_engine.device}')

    # # load checkpoint

    os.makedirs(args.output_dir, exist_ok=True)
    # _, client_sd = model_engine.load_checkpoint(load_dir, tag=None)
    # step = client_sd['step']

    # from finetune_qwen.datasets.load_dataset_private_qa import load_dataset
    # data_module = load_dataset(tokenizer=tokenizer)

    from finetune_dsp.datasets import load_medmcq
    data_module = {
        'train_dataset': load_medmcq.main()
    }

    from torch.utils.data.dataloader import DataLoader
    from finetune_dsp.load_dataset import collate_fn
    from functools import partial
    dataloader = DataLoader(data_module['train_dataset'],
                            batch_size=args.train_batch_size,
                            collate_fn=partial(collate_fn, device=model.device),
                            )
    print(f'n_train_batch: {len(dataloader)}')

    m_loss = LossAverage()
    for step, batch in enumerate(dataloader):
        # print(batch['input_ids'].device)
        # exit(0)
        t0 = time.time()

        # forward() method
        loss = model_engine(**batch)['loss']
        # print(loss.keys())
        # exit(0)

        # runs backpropagation
        model_engine.backward(loss)

        # weight update
        model_engine.step()

        m_loss.update(loss.item(), args.train_batch_size)

        t1 = time.time()
        throughput = (batch['input_ids'].shape[0] * batch['input_ids'].shape[1]) / (t1 - t0)


        if step %100 == 0:
            print(f'step: {step}, loss: {m_loss.avg}, throughput_per_device: {round(throughput, 3)}')

        # # save checkpoint
        # if step > 0 and step % 100 == 0:
        #     # # client_sd['step'] = step
        #     # ckpt_id = loss.item()
        #     # model_engine.save_checkpoint(load_dir, ckpt_id,
        #     #                              # client_sd=client_sd
        #     #                              )

    from dschat.utils.utils import save_hf_format
    save_hf_format(model, tokenizer, args)
    tokenizer.save_pretrained(args.output_dir)




if __name__ == '__main__':
    train()
