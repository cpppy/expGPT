import openai

openai.api_key = "sk-NYsoG3VBKDiTuvdtC969F95aFc4f45379aD3854a93602327"
openai.api_base = "https://key.wenwen-ai.com/v1"

# completion = openai.ChatCompletion.create(
#   model="gpt-3.5-turbo",
#   messages=[
#     {"role": "system", "content": "You are a helpful assistant."},
#     {"role": "user", "content": "Hello!"}
#   ]
# )

# print(completion.choices[0].message)

import os
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key="sk-NYsoG3VBKDiTuvdtC969F95aFc4f45379aD3854a93602327",  # os.environ.get("OPENAI_API_KEY"),
    base_url="https://key.wenwen-ai.com/v1",
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Say this is a test",
            # "content": "tell me your name",
        }
    ],
    model="gpt-3.5-turbo",
)

print(chat_completion.choices[0].message.content)
