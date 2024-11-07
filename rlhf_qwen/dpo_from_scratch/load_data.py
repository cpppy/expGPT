'''

https://github.com/rasbt/LLMs-from-scratch/blob/main/ch07/04_preference-tuning-with-dpo/dpo-from-scratch.ipynb

'''
import json
import jsonlines


def load_data():

    file_path = "../data/dpo_zh_500.jsonl"

    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        data = [json.loads(x) for x in lines]
        data = [
            dict(
                instruction=s['question'],
                input='',
                output=s['response_chosen'],
                chosen=s['response_chosen'],
                rejected=s['response_rejected'],
            )
            for s in data
        ]

    print(data[0])
    print("Number of entries:", len(data))
    return data


def format_input(entry):
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text



import torch
from torch.utils.data import Dataset

class PreferenceDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data

        # Pre-tokenize texts
        self.encoded_texts = []
        for entry in data:
            prompt = format_input(entry)
            rejected_response = entry["rejected"]
            chosen_response = entry["chosen"]

            prompt_tokens = tokenizer.encode(prompt)
            chosen_full_text = f"{prompt}\n\n### Response:\n{chosen_response}"
            rejected_full_text = f"{prompt}\n\n### Response:\n{rejected_response}"
            chosen_full_tokens = tokenizer.encode(chosen_full_text)
            rejected_full_tokens = tokenizer.encode(rejected_full_text)

            self.encoded_texts.append({
                "prompt": prompt_tokens,
                "chosen": chosen_full_tokens,
                "rejected": rejected_full_tokens,
            })

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)

def custom_collate_fn(
    batch,
    pad_token_id=50256,
    allowed_max_length=None,
    mask_prompt_tokens=True,
    device="cpu"
):
    # Initialize lists to hold batch data
    batch_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
        "rejected_mask": [],
        "chosen_mask": []

    }

    # Determine the longest sequence to set a common padding length
    max_length_common = 0
    if batch:
        for key in ["chosen", "rejected"]:
            current_max = max(len(item[key])+1 for item in batch)
            max_length_common = max(max_length_common, current_max)

    # Process each item in the batch
    for item in batch:
        prompt = torch.tensor(item["prompt"])
        batch_data["prompt"].append(prompt)

        for key in ["chosen", "rejected"]:
            # Adjust padding according to the common maximum length
            sequence = item[key]
            padded = sequence + [pad_token_id] * (max_length_common - len(sequence))
            mask = torch.ones(len(padded)).bool()

            # Set mask for all padding tokens to False
            mask[len(sequence):] = False

            # Set mask for all input tokens to False
            # +2 sets the 2 newline ("\n") tokens before "### Response" to False
            if mask_prompt_tokens:
                mask[:prompt.shape[0]+2] = False

            batch_data[key].append(torch.tensor(padded))
            batch_data[f"{key}_mask"].append(mask)

    # Final processing
    for key in ["chosen", "rejected", "chosen_mask", "rejected_mask"]:
        # Stack all sequences into a tensor for the given key
        tensor_stack = torch.stack(batch_data[key])

        # Optionally truncate to maximum sequence length
        if allowed_max_length is not None:
            tensor_stack = tensor_stack[:, :allowed_max_length]

        # Move to the specified device
        batch_data[key] = tensor_stack.to(device)

    return batch_data


def build_dataloader():

    from importlib.metadata import version

    pkgs = [
        "tiktoken",    # Tokenizer
        "torch",       # Deep learning library
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")

    ############## PROCESS DATASET #############
    data = load_data()
    train_portion = int(len(data) * 0.85)  # 85% for training
    test_portion = int(len(data) * 0.1)  # 10% for testing
    val_portion = len(data) - train_portion - test_portion  # Remaining 5% for validation

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))

    from functools import partial

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    customized_collate_fn = partial(
        custom_collate_fn,
        device=device,  # Put the data directly on a GPU if available
        mask_prompt_tokens=True,  # This is optional
        allowed_max_length=1024  # The supported context length of the model
    )

    import tiktoken
    from torch.utils.data import DataLoader

    tokenizer = tiktoken.get_encoding("gpt2")

    example_dataset = PreferenceDataset(train_data, tokenizer)

    example_dataloader = DataLoader(
        example_dataset,
        batch_size=2,
        collate_fn=customized_collate_fn,
        shuffle=True
    )

    for batch in example_dataloader:
        print("batch.keys:", batch.keys())
        for k, v in batch.items():
            if isinstance(v, list):
                print(f'{k}, {[x.shape for x in v]}')
            else:
                print(f'{k}, {v.shape}')
        break

    def decode_tokens_from_batch(token_ids, tokenizer):
        ids_in_python_list = token_ids.flatten().tolist()
        return tokenizer.decode(ids_in_python_list)

    batch = next(iter(example_dataloader))
    text = decode_tokens_from_batch(
        token_ids=batch["prompt"][0],  # [0] for the first entry in the batch
        tokenizer=tokenizer,
    )
    print(repr(text))
    text = decode_tokens_from_batch(
        token_ids=batch["chosen"][0][batch["chosen_mask"][0]],
        tokenizer=tokenizer,
    )
    print(repr(text))
    text = decode_tokens_from_batch(
        token_ids=batch["rejected"][0][batch["rejected_mask"][0]],
        tokenizer=tokenizer,
    )
    print(repr(text))

    from torch.utils.data import DataLoader

    num_workers = 0
    batch_size = 8

    torch.manual_seed(123)

    train_dataset = PreferenceDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    val_dataset = PreferenceDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    test_dataset = PreferenceDataset(test_data, tokenizer)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader


if __name__=='__main__':

    build_dataloader()

