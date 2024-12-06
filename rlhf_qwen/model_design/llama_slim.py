import math
import torch

from transformers import AutoTokenizer
from dschat.utils.model.reward_model import RewardModel

def main():
    model_path = '/data/Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path)

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
        # bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
        # eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
        # pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
        # max_position_embeddings=2 * SEQ_LENGTH,
    )

    model = LlamaForCausalLM(config)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id
    model.resize_token_embeddings(int(
        8 *
        math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8

    embed_weight = torch.load('/mnt2/output/qwen2_emb.pt')
    for k, v in model.named_parameters():
        if 'embed' in k:
            v.data = embed_weight[k].data
            print(f'REPLACED [{k}] by qwen_embed_layer_w: {v.shape}')

    num_padding_at_beginning = 1
    compute_fp32_loss = False
    critic_model = RewardModel(
        model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning,
        compute_fp32_loss=compute_fp32_loss)



if __name__=='__main__':

    main()

