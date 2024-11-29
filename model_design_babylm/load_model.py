import transformers


def main():

    SEQ_LENGTH = 128

    # # tokenizer_path = PATH / "models/gpt-clean-16000.json"
    # tokenizer_path = './vocab.json'
    # tokenizer = GPT2TokenizerFast(tokenizer_file=str(tokenizer_path))

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        # pretrained_model_name_or_path="/data/Qwen2.5-0.5B-Instruct",
        pretrained_model_name_or_path="/data/Qwen/Qwen2-0.5B-Instruct",
        # cache_dir=training_args.cache_dir,
        # model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    print(f'n_vocab: {tokenizer.vocab_size}')

    tokenizer.bos_token = "<s>"
    tokenizer.eos_token = "</s>"
    tokenizer.pad_token = "<pad>"

    from transformers import LlamaConfig, LlamaForCausalLM
    config = LlamaConfig(
        # vocab_size=16000,
        vocab_size=tokenizer.vocab_size,
        # hidden_size=512,
        hidden_size=896,
        num_hidden_layers=16,
        intermediate_size=1024,
        # intermediate_size=512,
        num_attention_heads=8,
        bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
        eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        max_position_embeddings=2 * SEQ_LENGTH,
    )

    student = LlamaForCausalLM(config)
    print(student)


    n_params = 0
    for k, v in student.named_parameters():
        if 'embed' in k or 'lm_head' in k:
            continue
        print(k, v.shape)
        n_params += v.numel()
    print(f'n_params: {n_params / 1e6} M')

    '''
    if tokenizer.vocab is 16000, the model size is 58M
    if tokenizer.vocab is 151643, the model size is 197M, 41M if not contain embedding layer
    '''


if __name__=='__main__':

    main()

