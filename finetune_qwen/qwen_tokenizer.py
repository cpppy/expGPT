import transformers


if __name__=='__main__':


    tokenizer = transformers.AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path='/data/data/Qwen2-0.5B-Instruct',
        # cache_dir=training_args.cache_dir,
        # model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    # tokenizer.pad_token_id = tokenizer.eod_id
    print(tokenizer.eos_token)
    print(tokenizer.eos_token_id)

    print(tokenizer.bos_token)
    print(tokenizer.bos_token_id)
    print(tokenizer.all_special_tokens)
    print(tokenizer.all_special_tokens_extended)
    print(tokenizer.special_tokens_map_extended)
    tokenizer.im_start_token = '<|im_start|>'
    tokenizer.im_start_id = tokenizer.convert_tokens_to_ids("<|im_start|>")
    tokenizer.im_end_token = '<|im_end|>'
    tokenizer.im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")

