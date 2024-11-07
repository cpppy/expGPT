import torch
import torch.nn as nn
import torch.nn.functional as F

from model_design.lit_llama_model import LLaMA, LLaMAConfig

import sys
from loguru import logger

max_seq_length = 256  # see scripts/prepare_alpaca.py

torch.backends.cuda.enable_flash_sdp(False)

torch.set_float32_matmul_precision('medium')


def main():

    config = LLaMAConfig.from_name("410M")
    config.block_size = max_seq_length

    print(config)

    # fabric = L.Fabric(accelerator="cuda", devices=2, precision="bf16-true")
    # fabric.launch()
    # fabric.seed_everything(1337 + fabric.global_rank)

    # if fabric.global_rank == 0:
    #     os.makedirs(out_dir, exist_ok=True)

    model = LLaMA(config)
    print(model)






if __name__=='__main__':

    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # main()
    import numpy as np
    a = np.ones((2, 4), dtype=np.int32)
    print(a)
    a = np.reshape(a, (-1,))
    print(a)

    b = memoryview(a)
    print(b)
    b[2:4] = 2
    print(a)