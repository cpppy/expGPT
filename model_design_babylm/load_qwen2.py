
import math

if __name__=='__main__':

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = '/data/Qwen/Qwen2.5-0.5B-Instruct'
    model = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map="auto",
                                                 torch_dtype=torch.bfloat16,
                                                 # quantization_config=bnb_config,
                                                 )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    saved = {}
    n_total = 0
    n_wo_emb = 0
    for k, v in model.named_parameters():
        n_total += v.numel()
        if 'embed' in k:
            print(k, v.shape)
            saved[k] = v
            continue
        n_wo_emb += v.numel()

    print(f'model size: {n_total/1e6}M, wo emb: {n_wo_emb/1e6}')

    # torch.save(saved, '/mnt2/output/qwen2_emb.pt')



