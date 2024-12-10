import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import math

from transformers import AutoTokenizer

# from dschat.utils.model.reward_model import RewardModel
from rlhf_qwen.model_design.reward_model_mod import RewardModel
from dschat.utils.data.data_utils import DataCollatorReward
from dschat.utils.utils import to_device


def train():
    # --------------------- DATASET -----------------------
    from rlhf_qwen.datasets.lmdb_dataloader import CustomDPODataset
    train_dataset = CustomDPODataset(
        # db_path='/mnt2/data/dsp_data_files2/dpo_zh_500_train_train_tokenizerQwen25_cache_20241204.lmdb',
        db_path='/mnt2/data/dsp_data_files2/dpo_en_zh_20k_train_tokenizerQwen25_cache_20241204.lmdb',
        max_len=512
    )
    eval_dataset = CustomDPODataset(
        # db_path='/mnt2/data/dsp_data_files2/dpo_zh_500_eval_eval_tokenizerQwen25_cache_20241204.lmdb'
        db_path='/mnt2/data/dsp_data_files2/dpo_en_zh_20k_eval_tokenizerQwen25_cache_20241204.lmdb',
        max_len=512
    )

    per_device_train_batch_size = 2
    per_device_eval_batch_size = 2

    # DataLoaders creation:
    data_collator = DataCollatorReward()
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
 
    train_dataloader = DataLoader(train_dataset,
                                  collate_fn=data_collator,
                                  sampler=train_sampler,
                                  batch_size=per_device_train_batch_size)
    eval_dataloader = DataLoader(eval_dataset,
                                 collate_fn=data_collator,
                                 sampler=eval_sampler,
                                 batch_size=per_device_eval_batch_size)

    # --------------------- MODEL --------------------
    model_path = '/data/Qwen/Qwen2.5-0.5B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # from transformers import LlamaConfig, LlamaForCausalLM
    # config = LlamaConfig(
    #     # vocab_size=16000,
    #     vocab_size=tokenizer.vocab_size,
    #     # hidden_size=512,
    #     hidden_size=896,
    #     num_hidden_layers=16,
    #     intermediate_size=1024,
    #     # intermediate_size=512,
    #     num_attention_heads=8,
    #     # bos_token_id=tokenizer.convert_tokens_to_ids("<s>"),
    #     # eos_token_id=tokenizer.convert_tokens_to_ids("</s>"),
    #     # pad_token_id=tokenizer.convert_tokens_to_ids("<pad>"),
    #     # max_position_embeddings=2 * SEQ_LENGTH,
    # )
    #
    # model = LlamaForCausalLM(config)

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # print(model.device, model.dtype)
    # exit(0)

    model.config.end_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = model.config.eos_token_id

    print(f'model_config: {model.config}')

    print(f'tokenizer n_vocab: {tokenizer.vocab_size}')

    # ----------- resize embedding size ------------
    # model.resize_token_embeddings(int(
    #     8 *
    #     math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8
    #
    # embed_weight = torch.load('/mnt2/output/qwen2_emb.pt')
    # for k, v in model.named_parameters():
    #     if 'embed' in k:
    #         v.data = embed_weight[k].data
    #         print(f'REPLACED [{k}] by qwen_embed_layer_w: {v.shape}')

    num_padding_at_beginning = 1
    compute_fp32_loss = False
    rm_model = RewardModel(
        model,
        tokenizer,
        num_padding_at_beginning=num_padding_at_beginning,
        compute_fp32_loss=compute_fp32_loss)

    # device = torch.device('cuda:0')
    device = model.device
    # rm_model.to(device)

    for step, batch in enumerate(train_dataloader):
        batch = to_device(batch, device)

        for k, v in batch.items():
            print(k, v.shape)

        outputs = rm_model(**batch, use_cache=False)
        loss = outputs["loss"]
        print(f'loss: {loss}')

        loss.backward()

        # rm_model.backward(loss)
        # rm_model.step()


        exit(0)

        # mean_loss += loss.item()
        # total_micro_steps += 1
        # gas_boundary = (total_micro_steps %
        #                 args.gradient_accumulation_steps == 0)
        # total_steps = total_micro_steps // args.gradient_accumulation_steps
        # if args.eval_interval and gas_boundary and (
        #         total_steps % args.eval_interval == 0):
        #     print_rank_0(f"Iter {total_steps}: Evaluating reward",
        #                  args.global_rank)
        #     reward_score, reject_score, acc = evaluation_reward(
        #         rm_model, eval_dataloader, args.eval_iters)
        #     print_rank_0(
        #         f"Iter {total_steps}: c_scores: {reward_score}, r_scores: {reject_score}, "
        #         f"diff: {reward_score - reject_score}, acc: {acc}",
        #         args.global_rank)
        #     rm_model.train()








if __name__=='__main__':

    train()

