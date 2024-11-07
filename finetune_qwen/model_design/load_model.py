def main():
    model_load_kwargs = {
        'low_cpu_mem_usage': True,  # not deepspeed.is_deepspeed_zero3_enabled(),
    }

    # # Set RoPE scaling factor
    # config = transformers.AutoConfig.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     trust_remote_code=True,
    # )
    # config.use_cache = False
    #
    # # Load model and tokenizer
    # model = transformers.AutoModelForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    #     device_map=device_map,
    #     trust_remote_code=True,
    #     # quantization_config=GPTQConfig(
    #     #     bits=4, disable_exllama=True
    #     # )
    #     # if training_args.use_lora and lora_args.q_lora else None,
    #     **model_load_kwargs,
    # )
    # tokenizer = transformers.AutoTokenizer.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    #     model_max_length=training_args.model_max_length,
    #     padding_side="right",
    #     use_fast=False,
    #     trust_remote_code=True,
    # )

    from finetune_qwen.model_design.modeling_qwen2 import Qwen2ForCausalLM, Qwen2Config

    config = Qwen2Config.from_dict(
        {
            "architectures": [
                "Qwen2ForCausalLM"
            ],
            "attention_dropout": 0.0,
            "bos_token_id": 151643,
            "eos_token_id": 151645,
            "hidden_act": "silu",
            "hidden_size": 896,
            "initializer_range": 0.02,
            "intermediate_size": 4864,
            "max_position_embeddings": 32768,
            "max_window_layers": 24,
            "model_type": "qwen2",
            "num_attention_heads": 14,
            "num_hidden_layers": 24,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "sliding_window": 32768,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "transformers_version": "4.40.1",
            "use_cache": False,
            "use_sliding_window": False,
            "vocab_size": 151936,
            "attn_implementation": "flash_attention_2",
            "rope_scaling": {'rope_type': 'default'}
        }

    )
    config.update(model_load_kwargs)

    # config.use_cache = False
    # config.rope_scaling = {'rope_type': 'default'}
    # config.torch_dtype = torch.bfloat16

    # model = Qwen2ForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     config=config,
    #     cache_dir=training_args.cache_dir,
    #     device_map=device_map,
    #     trust_remote_code=True,
    #     # quantization_config=GPTQConfig(
    #     #     bits=4, disable_exllama=True
    #     # )
    #     # if training_args.use_lora and lora_args.q_lora else None,
    #     **model_load_kwargs,
    # )

    model_path = '/data/expGPT/finetune_qwen/output_qwen2_scratch/checkpoint-10000'

    # model = Qwen2ForCausalLM(config=config)
    model = Qwen2ForCausalLM.from_pretrained(model_path, device_map='cuda:0')

    from transformers.models.qwen2.tokenization_qwen2 import Qwen2Tokenizer
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path,
                                               # cache_dir=training_args.cache_dir,
                                               model_max_length=1024,
                                               # padding_side="right",
                                               use_fast=False,
                                               # trust_remote_code=True,
                                               )

    text = 'today is a nice day, is there some good place to'
    model_inputs = tokenizer([text], return_tensors="pt").to('cuda:0')

    print(f'input_ids: {model_inputs}')

    from transformers.generation import GenerationConfig
    # gen_config = GenerationConfig.from_dict(
    #     {
    #         "_from_model_config": True,
    #         "bos_token_id": 151643,
    #         "eos_token_id": 151645,
    #         "transformers_version": "4.44.2",
    #         "use_cache": False
    #     }
    # )
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=32,
        # generation_config=gen_config,
    )
    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(response)


if __name__ == '__main__':
    main()
