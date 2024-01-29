from dotenv import load_dotenv
from langchain.chat_models import ChatLiteLLM


def main():
    load_dotenv()  # loads .env file
    llm = ChatLiteLLM(temperature=0.2,
                      # model_name="gpt-3.5-turbo-instruct"
                      model_name="gpt-3.5-turbo",
                      )


if __name__ == '__main__':
    main()
