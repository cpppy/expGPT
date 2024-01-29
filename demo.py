import os
from salesgpt.agents import SalesGPT
from langchain.chat_models import ChatLiteLLM

from dotenv import load_dotenv


def main():

    API_SECRET_KEY = "sk-NYsoG3VBKDiTuvdtC969F95aFc4f45379aD3854a93602327"
    BASE_URL = "https://key.wenwen-ai.com/v1"
    os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
    os.environ["OPENAI_API_BASE"] = BASE_URL
    load_dotenv()  # make sure you have .env file with your API keys, eg., OPENAI_API_KEY=sk-xxx

    # select your model - we support 50+ LLMs via LiteLLM https://docs.litellm.ai/docs/providers
    llm = ChatLiteLLM(temperature=0.4, model_name="gpt-3.5-turbo")

    # from langchain_community.chat_models.openai import ChatOpenAI
    # llm = ChatOpenAI(temperature=0.4, model_name="gpt-3.5-turbo")

    sales_agent = SalesGPT.from_llm(llm, use_tools=True, verbose=False,
                                    product_catalog="../examples/sample_product_catalog.txt",
                                    salesperson_name="Ted Lasso",
                                    salesperson_role="Sales Representative",
                                    company_name="Sleep Haven",
                                    company_business='''Sleep Haven 
                                is a premium mattress company that provides
                                customers with the most comfortable and
                                supportive sleeping experience possible. 
                                We offer a range of high-quality mattresses,
                                pillows, and bedding accessories 
                                that are designed to meet the unique 
                                needs of our customers.'''
                                    )
    sales_agent.seed_agent()
    sales_agent.determine_conversation_stage()  # optional for demonstration, built into the prompt
    # agent
    sales_agent.step()

    # user
    user_input = input('Your response: ')  # Yea, sure
    sales_agent.human_step(user_input)

    # agent
    sales_agent.determine_conversation_stage()  # optional for demonstration, built into the prompt
    sales_agent.step()

    # user
    user_input = input('Your response: ')  # What pricing do you have for your mattresses?
    sales_agent.human_step(user_input)

    # agent
    sales_agent.determine_conversation_stage()  # optional for demonstration, built into the prompt
    sales_agent.step()

if __name__=='__main__':

    main()

