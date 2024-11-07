

def main():

    # pip install -q transformers accelerate
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    checkpoint = "/data/data/mt0-small"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint, torch_dtype="auto", device_map="auto")

    inputs = tokenizer.encode("Translate to English: Je t’aime.", return_tensors="pt").to("cuda")
    outputs = model.generate(inputs)
    print(tokenizer.decode(outputs[0]))


if __name__=='__main__':

    main()


