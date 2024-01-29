import os

from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma

# from salesgpt.tools import get_tools, setup_knowledge_base

from loguru import logger


def setup_knowledge_base(product_catalog: str = None):
    """
    We assume that the product catalog is simply a text string.
    """
    # load product catalog
    with open(product_catalog, "r") as f:
        product_catalog = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=10, chunk_overlap=0)
    texts = text_splitter.split_text(product_catalog)
    for _t in texts:
        logger.debug(f'text: {_t}')

    llm = OpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()
    docsearch = Chroma.from_texts(
        texts, embeddings, collection_name="product-knowledge-base"
    )

    knowledge_base = RetrievalQA.from_chain_type(
        llm=llm, chain_type="stuff", retriever=docsearch.as_retriever()
    )
    return knowledge_base


def get_tools(knowledge_base):
    # we only use one tool for now, but this is highly extensible!
    tools = [
        Tool(
            name="ProductSearch",
            func=knowledge_base.run,
            description="useful for when you need to answer questions about product information",
        )
    ]
    return tools


def main():
    API_SECRET_KEY = "sk-NYsoG3VBKDiTuvdtC969F95aFc4f45379aD3854a93602327"
    BASE_URL = "https://key.wenwen-ai.com/v1"
    os.environ["OPENAI_API_KEY"] = API_SECRET_KEY
    os.environ["OPENAI_API_BASE"] = BASE_URL

    product_catalog = "../examples/sample_product_catalog.txt"
    knowledge_base = setup_knowledge_base(product_catalog)
    tools = get_tools(knowledge_base)

    tool_names = [tool.name for tool in tools]
    print(f'tool_names: {tool_names}')


if __name__ == '__main__':
    import sys

    logger.remove()
    logger.add(sys.stderr, level="DEBUG")

    main()
