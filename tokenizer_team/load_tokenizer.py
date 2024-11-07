from tokenizer_team.llama_tokenizer import Tokenizer

from pathlib import Path


def main():

    # tokenizer_path = Path('/data/lit-llama/recon/checkpoints/lit-llama/tokenizer.model')
    # tokenizer_path = Path('/data/alpaca-lora/recon/ch_data/chinese_sp.model')
    # tokenizer_path = Path('/data/llm_models/merged_tokenizer_hf/tokenizer.model')
    tokenizer_path = Path('/data/llm_models/llama_tokenizer/tokenizer.model')
    tokenizer = Tokenizer(tokenizer_path)
    print(f'vocab_size: {tokenizer.vocab_size}')

    print(f'bos_id: {tokenizer.bos_id}')
    print(f'eos_id: {tokenizer.eos_id}')
    print(f'pad_id: {tokenizer.pad_id}')
    print(f'unk_id: {tokenizer.processor.unk_id()}')
    print(tokenizer.processor.IdToPiece(0))

    text = '''白日依山尽，黄河入海流。欲穷千里目，更上一层楼。
    The primary use of LLaMA is research on large language models, including'''
    print("Test text:\n", text)
    print()
    print(f"Tokenized by tokenizer:{tokenizer.encode(text)}")
    print(f'n_token: {tokenizer.encode(text).shape}')
    print(f'decode result: {tokenizer.decode(tokenizer.encode(text))}')


if __name__=='__main__':

    main()


