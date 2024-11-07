import json
import glob
import os
from pathlib import Path
import sys

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import numpy as np
from tqdm import tqdm

from tokenizer_team.llama_tokenizer import Tokenizer
from pretrain.datasets import packed_dataset

#
# filenames_sample = [
#     "arxiv_sample.jsonl",
#     "book_sample.jsonl",
#     "c4_sample.jsonl",
#     "cc_2019-30_sample.jsonl",
#     "cc_2020-05_sample.jsonl",
#     "cc_2021-04_sample.jsonl",
#     "cc_2022-05_sample.jsonl",
#     "cc_2023-06_sample.jsonl",
#     "github_sample.jsonl",
#     "stackexchange_sample.jsonl",
#     "wikipedia_sample.jsonl",
# ]
#
# filename_sets = {
#     "arxiv": "arxiv/arxiv*",
#     "book": "book/book*",
#     "c4": "c4/c4-train*",
#     "common_crawl": "common_crawl/*",
#     "github": "github/filtered*",
#     "stackexchange": "stackexchange/stackexchange*",
#     "wikipedia": "wikipedia/wiki*",
# }


def prepare_sample(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    match = ""
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained (i.e. we reuse LLaMA's tokenizer model)."""
    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    filepath = source_path

    if not filepath.is_file():
        raise RuntimeError(
            f"Input file not found at {filepath}. \n"
            "Make sure you download the data, e.g. wget -i https://data.together.xyz/redpajama-data-1T/v1.0.0/urls.txt or through \n"
            "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T \n"
            "https://huggingface.co/datasets/togethercomputer/RedPajama-Data-1T-Sample \n"
        )

    name = filepath.name
    prefix, _ = os.path.splitext(name)
    print(f'prefix: {prefix}')

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=prefix,
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )

    print(f"Processing {name}")

    with open(filepath, encoding="utf-8") as f:
        for row in tqdm(f):
            text = json.loads(row)["text"]
            text_ids = tokenizer.encode(text)
            builder.add_array(np.array(text_ids, dtype=builder.dtype))

    builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 2049 * 1024,  # 2048 block size + 1 for causal (from LLama), 1024 blocks
    sample: bool = False,
    match: str = "",
) -> None:
    """Prepare the "Red Pajama" dataset. We assume tokenizer has been trained (i.e. we reuse LLaMA's tokenizer model)."""
    if sample:
        prepare_sample(
            source_path=source_path,
            tokenizer_path=tokenizer_path,
            destination_path=destination_path,
            chunk_size=chunk_size,
            match=match,
        )


if __name__ == "__main__":


    '''
    large-762M-k40.train.jsonl   small-117M.test.jsonl
large-762M-k40.valid.jsonl   small-117M.train.jsonl
large-762M.train.jsonl	     webtext.test.jsonl
large-762M.valid.jsonl	     webtext.train.jsonl
medium-345M-k40.train.jsonl  xl-1542M-k40.test.jsonl
medium-345M-k40.valid.jsonl  xl-1542M-k40.train.jsonl
medium-345M.train.jsonl      xl-1542M.test.jsonl
medium-345M.valid.jsonl      xl-1542M.train.jsonl
small-117M-k40.train.jsonl
    
    '''

    tags = [
        'webtext',
        'small-117M',
        'small-117M-k40',
        'medium-345M',
        'medium-345M-k40',
        'large-762M',
        'large-762M-k40',
        'xl-1542M',
        'xl-1542M-k40',
    ]


    # source_path = Path("/data/expGPT/pretrain/data/webtext.train.jsonl")
    # source_path = Path("/data/expGPT/pretrain/data/webtext.test.jsonl")
    # destination_path = Path("/data/data/llm_data/pretrain_webtext")

    # source_path = Path("/data/data/llm_data/raw_data_gpt2/medium-345M.train.jsonl")
    # destination_path = Path("/data/data/llm_data/pretrain_medium-345M")


    for tag in tags:
        print(f'### process: {tag}')
        source_path = Path(f"/data/data/llm_data/raw_data_gpt2/{tag}.train.jsonl")
        destination_path = Path(f"/data/data/llm_data/pretrain_{tag}")

        tokenizer_path = Path("/data/llm_models/llama_tokenizer/tokenizer.model")
        chunk_size = 2049 * 1024  # 2048 block size + 1 for causal (from LLama), 1024 blocks

        prepare_sample(
            source_path=source_path,
            tokenizer_path=tokenizer_path,
            destination_path=destination_path,
            chunk_size=chunk_size,
            match=''
        )
