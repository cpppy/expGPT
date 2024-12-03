import json
from argparse import ArgumentParser
import os
import tiktoken
from os.path import exists, join
from time import sleep
import jsonlines

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
        # model_path = '/data/data/Qwen2-0.5B-Instruct'
        # # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5b_ft_lora_huatuo2/checkpoint-110'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5b_ft_lora_huatuo2_test/checkpoint-40'

        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.bfloat16)
        # model = PeftModel.from_pretrained(model, model_id=lora_path)

        # ###################### qwen2-7B ###################
        # model_path = '/data/data/Qwen2-7B-Instruct'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_7b_int8_ft_lora_huatuo2/checkpoint-8800'
        #
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModelForCausalLM.from_pretrained(model_path,
        #                                              device_map="auto",
        #                                              # torch_dtype=torch.bfloat16,
        #                                              load_in_8bit=True,
        #                                              )
        # model = PeftModel.from_pretrained(model, model_id=lora_path)
        #
        # # model = AutoModelForCausalLM.from_pretrained(model_path,
        # #                                              device_map="auto",
        # #                                              torch_dtype=torch.bfloat16,
        # #                                              # load_in_8bit=True,
        # #                                              )

        # ###################### qwen2-7B ###################
        # model_path = '/data/data/Qwen2-7B-Instruct'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_7b_nf4_ft_lora_huatuo2/checkpoint-5600'
        #
        # from transformers import BitsAndBytesConfig
        # nf4_config = BitsAndBytesConfig(load_in_4bit=True,
        #                                 bnb_4bit_quant_type="nf4",
        #                                 bnb_4bit_compute_dtype=torch.bfloat16)
        #
        # tokenizer = AutoTokenizer.from_pretrained(model_path)
        # model = AutoModelForCausalLM.from_pretrained(model_path,
        #                                              device_map="auto",
        #                                              # torch_dtype=torch.bfloat16,
        #                                              quantization_config=nf4_config,
        #                                              )
        # model = PeftModel.from_pretrained(model, model_id=lora_path)

        ###################### qwen2-7B ###################
        # model_path = '/data/data/Qwen2-7B-Instruct'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_7b_int8_ft_lora_huatuo2_mgpu/checkpoint-2649'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_7b_int8_ft_lora_huatuo2_mgpu/checkpoint-32290'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5_qlora_prv_qa_exp4/checkpoint-759'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5_qlora_prv_qa_exp3/checkpoint-8'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5_qlora_prv_qa_exp3/checkpoint-400'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5_qlora_prv_qa_exp5_adamw/checkpoint-2000'
        # lora_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_7_qlora_prv_qa_exp6/checkpoint-1000'

        # model_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5b_fullp_prv_qa_exp7_plus/checkpoint-300'
        # model_path = '/mnt2/expGPT/finetune_qwen/output_qwen2_0.5b_fullp_prv_qa2_exp8_plus/checkpoint-561'
        model_path = '/mnt2/output/dsp_model_medical_ft1'

        tokenizer = AutoTokenizer.from_pretrained('/data/Qwen/Qwen2-0.5B-Instruct')

        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(model_path,
                                                     device_map="auto",
                                                     torch_dtype=torch.bfloat16,
                                                     # quantization_config=bnb_config,
                                                     )
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
        ).cuda()

        outputs = self.model.generate(
            inputs=input_ids,
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

        print(response_str)
        response_str = response_str[0:1]
        return response_str


def _merge_with_index(answers):
    return "\n".join([f"{i}) {answer}" for i, answer in enumerate(answers)])


def generate_message(question: str,
                     possible_answers: list[str],
                     # additional_context: dict = None,
                     additional_context: str = None,
                     ):
    possible_answers_example_prompt = _merge_with_index([
        "Decreased ability to copy", "Decreased execution", "Deficit of expression by gesture",
        "Deficit of fluent speech"
    ])
    message = [{
        "role": "system",
        "content": f"""
        Answer the question provided you by the user. For the question you will have few possible answers.
        Make sure to answer the question accurately, as any mistakes could have serious consequences for the patient.
        The response options are: {', '.join([str(i) for i in range(len(possible_answers))])}. Other options may be partially correct, but there is only ONE BEST answer. For example:
        For the question 'Hypomimia is ?', possible answers are: {possible_answers_example_prompt}
        We expect you to return just a single number: 2. Additional information may be provided at times.
        """,
    }]

    possible_answers_prompt = _merge_with_index(possible_answers)
    if additional_context is not None:
        # message += [{
        #     "role": "assistant",
        #     "content": f"Here the knowledge that I have. May it could help me to answer the question",
        # }]
        # for i, context in additional_context.items():
        #     message += [{
        #         "role": "assistant",
        #         "content": f"Few facts about answer {i}) {possible_answers[i]}:\n" + "\n - ".join(
        #             [c[:4096] for c in context]),
        #     }]

        message += [{
            "role": "assistant",
            "content": f"Here the knowledge that I have. May it could help me to answer the question. \nContext: {additional_context} \n\n\n",
        }]

    message += [
        {
            "role": "user",
            "content": f"Question: {question}\n\n" +
                       f"Possible answers: \n{possible_answers_prompt}\n\n" +
                       f"Choose only one of the answers, and return the number of {', '.join([str(i) for i in range(len(possible_answers))])} correspomdimg to the correct snswer." +
                       "No need to add any natural language explanation, only the number of the right answer"
        }
    ]
    return message


def run_evaluation(data_path: str, with_rag=False):
    data = []
    with jsonlines.open(data_path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    # data = []
    # with jsonlines.open(data_path, "r") as f:
    #     for line in f:
    #         _data = json.loads(line)
    #         # if _data['level'] == '困难':
    #         #     data.append(_data)
    #         if _data['level'] == '中等':
    #             data.append(_data)

    # if "gpt4" in model:
    #     openai_model_name = "gpt-4"
    # elif "chatgpt" in model:
    #     openai_model_name = "gpt-3.5-turbo"
    # elif "llama" in model:
    #     tokenizer = AutoTokenizer.from_pretrained(f"models/{model}-hf")
    #     pipe = pipeline(
    #         "text-generation",
    #         model=f"models/{model}-hf",
    #         torch_dtype=torch.float16,
    #         device_map="auto",
    #         device=0
    #     )
    # else:
    #     raise ValueError(f"Unknown model {model}")

    # done = []
    # if exists(join("data", task, f"{model}-{split}.json")):
    #     with open(join("data", task, f"{model}-{split}.json"), "r") as f:
    #         for line in f:
    #             jsn = json.loads(line)
    #             done.append(jsn["id"])

    print(data[0])

    data = data[::8]
    print(f'n_sample: {len(data)}')
    # exit(0)

    bot = AnswerBot()

    # data = data[0:75]
    n_correct = 0
    n_total = 0
    for _data in tqdm(data):
        try:
            question = _data['question']
            options = _data['options']
            answer_idx = _data['answer']
            # print(data)
            option_ids = sorted(_data['options'].keys())
            # print(option_ids)
            # if with_rag:
            #     from medqa_test.generate_only_rag import search_from_rag
            #     chunks = search_from_rag(query=question, score_thr=0.5)
            #     context_str = '\n'.join([x['page_content'] for x in chunks])
            # else:
            context_str = None
            messages = generate_message(question=question,
                                        possible_answers=[options[x] for x in option_ids],
                                        additional_context=context_str)
            # print(messages)
            answer_idx = option_ids.index(answer_idx)
            # answer_idx = answer_idx
            # print(f'answer_idx: {answer_idx}')

            response = bot.inference(messages)

            n_total += 1
            if str(answer_idx) == response:
                n_correct += 1
            print(f'ANSWER: {answer_idx} ---> PREDICTION: {response}, ACC: {n_correct / n_total}')
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
