
import math


def main():

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = '/data/Qwen/Qwen2.5-0.5B-Instruct'
    # model = AutoModelForCausalLM.from_pretrained(model_path,
    #                                              device_map="auto",
    #                                              torch_dtype=torch.bfloat16,
    #                                              # quantization_config=bnb_config,
    #                                              )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    # model.resize_token_embeddings(int(
    #     8 *
    #     math.ceil(len(tokenizer) / 8.0)))  # make the vocab size multiple of 8\

    model_save_path = '/mnt2/output/dsp_model_medical_ft1'
    # model_save_path = model_path

    model = AutoModelForCausalLM.from_pretrained(model_save_path,
                                                 )

    # Convert prompt to tokens

    prompt_template = """
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
${query}<|im_end|>
<|im_start|>assistant
"""

    # query = 'I have a headache, can you give me some suggestion?'
    query = '我有点头痛，请问可能是什么原因导致的？'
    prompt = prompt_template.replace('${query', query)

    prompt = """
Question: 上火在中医理论中，主要与哪些因素相关？
Options:
A. 饮食不节、饮水不足
B. 熬夜失眠、心理压力
C. 天气燥热
D. 以上都是

现在从ABCD四个选项选一个，请直接告诉我答案
"""

    prompt = """
Question: 与抗生素相比，热淋清片在治疗尿路感染时的主要优势是什么？
Options:
A. 热淋清片可以治疗所有类型的尿路感染
B. 热淋清片具有更广泛的抗菌谱
C. 热淋清片可以减少抗生素的耐药性问题
D. 热淋清片无需进行细菌培养和药敏试验即可使用

现在从ABCD四个选项选一个，请直接告诉我答案
"""

# You should select only one option from the options above.
# Then, the option you selected must be in ['A', 'B', 'C', 'D'], you should choose only one from this list.
# So, The correct option is"""


    tokens = tokenizer(
        prompt,
        return_tensors='pt'
    ).input_ids

    # Generate output
    generation_output = model.generate(
        tokens,
        # streamer=streamer,
        # max_length=2048,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_p=0.3,
    )

    print(generation_output)

    # response = tokenizer.decode(generation_output[0][tokens.shape[-1]:], skip_special_tokens=False)
    response = tokenizer.decode(generation_output[0], skip_special_tokens=False)
    print(response)


if __name__=='__main__':

    main()

