import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


def collate_fn(batch, device):
    # input_ids, labels = zip(*batch)  # transposed
    input_ids = [b['input_ids'] for b in batch]
    labels = [b['labels'].to(torch.int64) for b in batch]
    attention_mask = [b['attention_mask'] for b in batch]

    # max_len = max(len(s) for s in input_ids)
    #
    # def pad_right(x, pad_id):
    #     # pad right based on the longest sequence
    #     n = max_len - len(x)
    #     return torch.cat((x, torch.full((n,), pad_id, dtype=x.dtype)))
    #
    # x = torch.stack([pad_right(x, pad_id=0) for x in input_ids])
    # y = torch.stack([pad_right(x, pad_id=-1) for x in labels])

    return {
        'input_ids': torch.stack(input_ids).to(device),
        'labels': torch.stack(labels).to(device),
        'attention_mask': torch.stack(attention_mask).to(device),
    }


def main():
    base_model_path = "/data/data/Qwen2-0.5B-Instruct"

    import transformers

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=base_model_path,
        model_max_length=1024,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )

    from finetune_qwen.datasets.load_dataset_private_qa import load_dataset
    data_module = load_dataset(tokenizer=tokenizer)

    from torch.utils.data.dataloader import DataLoader
    from transformers.data.data_collator import default_data_collator
    dataloader = DataLoader(data_module['train_dataset'],
                            batch_size=2,
                            collate_fn=partial(collate_fn, device=torch.device('cuda:0')),
                            )
    print(f'n_train_batch: {len(dataloader)}')

    for step, batch in enumerate(dataloader):
        print(batch['input_ids'].device)

        for k, v in batch.items():
            print(k, v.shape)


        exit(0)


if __name__=='__main__':

    main()

