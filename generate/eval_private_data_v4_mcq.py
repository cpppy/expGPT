import json
from argparse import ArgumentParser
import tiktoken
from os.path import exists, join
from time import sleep
import jsonlines

import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import sys

sys.path.append('..')

# from transformers import pipeline, AutoTokenizer
# import torch

# from utils.tothepoint_io import ttp_retrieve, ttp_renew_token, ttp_validate_token

import openai
from tqdm import tqdm

# import tiktoken
#
# enc = tiktoken.get_encoding("cl100k_base")

'''
https://github.com/To-The-Point-Tech/medqa-chatgpt-evaluation/blob/main/utils/tothepoint_io.py

'''


class AnswerBot:

    def __init__(self):

        # from llama_index.llms.openai_like import OpenAILike
        # from medqa_test.qwen2_system_prompt import completion_to_prompt, messages_to_prompt
        # self.llm = OpenAILike(
        #
        #     # model='qwen2:72b-instruct',
        #     # api_base="http://192.168.100.24:11434/v1",
        #
        #     # model='qwen2.5:72b-instruct',
        #     # api_base="http://192.168.100.24:11434/v1",
        #
        #     # model='qwen2:7b-instruct-fp16',
        #     # api_base="http://192.168.100.24:11434/v1",
        #
        #     # model='qwen2.5:7b-instruct-fp16',
        #     # api_base="http://192.168.100.24:11434/v1",
        #
        #     model='qwen2.5:0.5b-instruct-fp16',
        #     api_base="http://192.168.100.24:11434/v1",
        #
        #     # model='Qwen2.5-72B-Instruct',
        #     # api_base="http://192.168.100.27:8000/v1",
        #
        #     # model='Qwen2-0.5B-Instruct',
        #     # api_base="http://192.168.100.27:8000/v1",
        #
        #     messages_to_prompt=messages_to_prompt,
        #     completion_to_prompt=completion_to_prompt,
        #     api_key='EMPTY',
        #     temperature=0,
        #     max_tokens=2048,
        #     timeout=600,
        # )
        # from finetune_qwen import inference
        self.model, self.tokenizer = self._load_model()

    def _load_model(self):

        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
        from peft import PeftModel

        # ###################### qwen2-0.5B ###################
        model_path = '/data/Qwen2.5-0.5B-Instruct'
        # # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5b_ft_lora_huatuo2/checkpoint-110'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5b_ft_lora_huatuo2_test/checkpoint-40'

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(
            # model_path,
            # '/data/output/dsp_demo_saved',
            '/data/output/dsp_demo_saved_2',
            device_map="auto",
            torch_dtype=torch.float16)
        # model = PeftModel.from_pretrained(model, model_id=lora_path)

        model.eval()
        return model, tokenizer

    def inference(self, messages):
        from llama_index.core.llms import ChatMessage, MessageRole, ChatResponse
        _messages = []
        for m in messages:
            if m['role'] == 'system':
                _messages.append(ChatMessage(role=MessageRole.SYSTEM, content=m['content']))
            elif m['role'] == 'user':
                _messages.append(ChatMessage(role=MessageRole.USER, content=m['content']))
            if m['role'] == 'assistant':
                _messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=m['content']))

        # response = self.llm.app(_messages)
        # response_str = response.message.content
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            # tokenize=False,
        )
        # print(input_ids)
        # exit(0)


        outputs = self.model.generate(
            inputs=input_ids.cuda(),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            top_k=3,
            top_p=0.6,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = outputs[0][input_ids.shape[-1]:]
        response_str = self.tokenizer.decode(response, skip_special_tokens=True)
        # exit(0)

        # print(response_str)
        response_str = response_str[0:8].strip()
        return response_str

    def completion_inference(self, prompt):
        # from llama_index.core.llms import ChatMessage, MessageRole, ChatResponse
        # _messages = []
        # for m in messages:
        #     if m['role'] == 'system':
        #         _messages.append(ChatMessage(role=MessageRole.SYSTEM, content=m['content']))
        #     elif m['role'] == 'user':
        #         _messages.append(ChatMessage(role=MessageRole.USER, content=m['content']))
        #     if m['role'] == 'assistant':
        #         _messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=m['content']))

        # response = self.llm.app(_messages)
        # response_str = response.message.content
        input_ids = self.tokenizer.encode(
            prompt,
            # add_generation_prompt=True,
            return_tensors="pt",
            # tokenize=False,
        )
        # print(input_ids)
        # exit(0)

        outputs = self.model.generate(
            inputs=input_ids.cuda(),
            max_new_tokens=256,
            do_sample=True,
            temperature=0.1,
            top_k=3,
            top_p=0.6,
            pad_token_id=self.tokenizer.eos_token_id
        )

        response = outputs[0][input_ids.shape[-1]:]
        response_str = self.tokenizer.decode(response, skip_special_tokens=True)
        # exit(0)

        # print(response_str)
        response_str = response_str[0:8].strip()
        return response_str



# def to_llama_format(message: list):
#     system_message, *other_messages, user_message = message
#     system_promt = system_message["content"]
#     user_prompt = user_message["content"]
#     return f"""<s>[INST] <<SYS>>
#     {system_promt}
#     <</SYS>>
#     {user_prompt} [/INST]"""


def create_optimized_prompt(row):
    # 构造解释性更强、指示性更明确的格式
    # prompt = (f"Read the question and choose the correct option based on the options given.\n\n"
    #           f"Question: {row['question']}\n"
    #           f"Options:\n"
    #           f"A. {row['A']}\n"
    #           f"B. {row['B']}\n"
    #           f"C. {row['C']}\n"
    #           f"D. {row['D']}\n"
    #           f"\nThe correct option is: ")
    prompt = (f"Read the question and choose the correct option based on the options given.\n\n"
              f"Question: {row['question']}\n"
              f"Options:\n"
              f"A. {row['options']['A']}\n"
              f"B. {row['options']['B']}\n"
              f"C. {row['options']['C']}\n"
              f"D. {row['options']['D']}\n"
              f"\nYou should select only one option from the options above.\n"
              f"Then, the option you selected must be in ['A', 'B', 'C', 'D'], you should choose only one from this list."
              f"\nSo, The correct option is:")
    return prompt



def run_evaluation(data_path: str, with_rag=False):
    data = []
    with jsonlines.open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    print(data[0])

    data = data[::8]
    print(f'n_sample: {len(data)}')

    bot = AnswerBot()

    # data = data[0:75]
    n_correct = 0
    n_total = 0
    for _data in tqdm(data):
        try:
            # question = _data['question']
            # options = _data['options']
            answer = _data['answer']
            prompt = create_optimized_prompt(_data)
            # print(prompt)
            # exit(0)

            messages = []
            messages.append({"role": "user", "content": prompt})
            # messages.append({"role": "assistant", "content": answer})

            # response = bot.inference(messages)
            response = bot.completion_inference(prompt)

            n_total += 1
            if answer == response:
                n_correct += 1
            print(f'ANSWER: {answer} ---> PREDICTION: {response}, ACC: {n_correct / n_total}')
        except Exception as e:
            print(e)
        # exit(0)
    print(f'final_acc: {n_correct / len(data)}')

    exit(0)

    # token = ""
    # with open(join("data", task, f"{model}-{split}.json"), "a") as f:
    #     for line in tqdm(data):
    #         id_ = line["id"]
    #         if id_ in done:
    #             continue
    #
    #         question = line["sent1"]
    #         n_classes = len([k for k in line if "ending" in k])
    #         answers = [line[f'ending{i}'] for i in range(n_classes)]
    #
    #         try:
    #             if "gpt" in model:
    #                 if model == "tothepoint-chatgpt":
    #                     if not ttp_validate_token(token=token):
    #                         token = ttp_renew_token()
    #                     try:
    #                         context = {
    #                             i: [e['summary'] for e in ttp_retrieve(answer, token=token, k=3)]
    #                             for i, answer in enumerate(answers)
    #                         }
    #                     except:
    #                         continue
    #                 elif model == "chatgpt":
    #                     context = None
    #                 elif model == "gpt4":
    #                     context = None
    #                 else:
    #                     raise ValueError(f"Unsupported model {model}")
    #                 message = generate_message(question=question, possible_answers=answers, additional_context=context)
    #                 response = openai.ChatCompletion.create(
    #                     model=openai_model_name,
    #                     messages=message,
    #                     temperature=0.2,
    #                     n=1,
    #                     max_tokens=3,
    #                 )
    #                 class_ = [e["message"]["content"].strip() for e in response["choices"]][0]
    #                 sleep(2)
    #             elif "llama" in model:
    #                 context = None
    #                 message = generate_message(question=question, possible_answers=answers, additional_context=context)
    #                 message = to_llama_format(message)
    #                 sequences = pipe(
    #                     message,
    #                     do_sample=True,
    #                     top_k=10,
    #                     num_return_sequences=1,
    #                     eos_token_id=tokenizer.eos_token_id,
    #                     max_length=4096,
    #                     temperature=0.2,
    #                     max_new_tokens=3,
    #                     return_full_text=False,
    #                 )
    #                 class_ = sequences[0]["generated_text"]
    #             else:
    #                 raise ValueError(f"Unsupported model {model}")
    #             print(class_)
    #         except:
    #             print("Fail")
    #             continue
    #         outputs = {
    #             "answer": class_,
    #             "id": id_,
    #         }
    #         f.write(json.dumps(outputs) + "\n")
    #
    # id2answer = {}
    # with open(join("data", task, f"{model}-{split}.json"), "r") as f:
    #     for line in f:
    #         jsn = json.loads(line)
    #         try:
    #             answer = int(jsn["answer"].replace(":", "").split(" ")[0])
    #             id2answer[jsn["id"]] = answer
    #         except:
    #             continue
    #
    # correct, total = 0, 0
    # for line in data:
    #     id_, label = line["id"], line["label"]
    #     if id_ in id2answer:
    #         answer = id2answer[id_]
    #         if label == answer:
    #             correct += 1
    #         total += 1
    # acc = round(100 * correct / total, 2)
    # print(f"Acctuacy of model {model} on {task} is {acc}%")


if __name__ == "__main__":
    # parser = ArgumentParser()
    # parser.add_argument("--model", choices=[
    #     "chatgpt", "gpt4",
    #     "llama-2-7b", "llama-2-13b",
    #     "llama-2-7b-chat", "llama-2-13b-chat",
    #     "tothepoint-chatgpt",
    # ])
    # parser.add_argument("--task", choices=["medmcqa", "medqa", "mmlu", "medcoding"])
    # parser.add_argument("--split", choices=["test", "dev"], default="test")
    # args_ = parser.parse_args()

    from utils.fix_seed import seed_everything

    seed_everything(1234)

    # run_evaluation(model=args_.model, task=args_.task, split=args_.split)

    # run_evaluation(
    #     data_path='./single_choice_samples_cardiovascular_diseases_20240925.jsonl',
    #     # with_rag=True,
    # )

    run_evaluation(
        data_path='./samples_from_baiduyidian_20240919.jsonl',
        # with_rag=True
    )
