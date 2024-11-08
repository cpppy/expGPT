import torch


def main():

    # pip install -q transformers accelerate
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    # checkpoint = "/data/data/mt0-small"
    # tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    # model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")


    checkpoint = '/data/output/dsp_demo_saved'
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=checkpoint,
        device_map='auto',
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,

    )

    tokenizer = AutoTokenizer.from_pretrained(
        # pretrained_model_name_or_path=checkpoint,
        pretrained_model_name_or_path="/data/Qwen2.5-0.5B-Instruct",
        model_max_length=512,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )



    inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))


if __name__=='__main__':

    main()


