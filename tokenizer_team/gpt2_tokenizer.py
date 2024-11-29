import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'


def main():
    from transformers import GPT2Tokenizer

    text = "This is a sequence"

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print(f'n_vocab: {tokenizer.vocab_size}')

    x = tokenizer(text, truncation=True, max_length=2)

    print(len(x))  # 2


if __name__=='__main__':

    main()

