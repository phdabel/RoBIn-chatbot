import os
from elasticsearch import Elasticsearch
from langchain_elasticsearch.retrievers import ElasticsearchRetriever
from langchain_elasticsearch.client import create_elasticsearch_client
from typing import Dict
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

ROBIN_QA_MODEL = os.getenv('ROBIN_QA_MODEL')
GPT_MODE = int(os.getenv("GPT_MODE"))
GPT_MODEL = os.getenv("GPT_MODEL")
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

es_client = create_elasticsearch_client(
    url=os.getenv("ES_BASE_URL"),
    api_key=os.getenv("ES_ACCESS_TOKEN")
)

def bm25_query(search_query: str) -> Dict:
    return {
        "query": {
            "match": {"text": search_query}
        }
    }

bm25_retriever = ElasticsearchRetriever(
    es_client=es_client,
    index_name=os.getenv("ES_INDEX_NAME"),
    body_func=bm25_query,
    content_field="text"
)

article_template = """Your job is to use PubMed articles to answer questions
about randomization, blinding, and other aspects of clinical trials. Use the
following context to answer questions. Be as detailed as possible, but don't
make up any information that's not from the context. If you don't know an
answer, say you don't know.
{context}
"""

article_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=article_template)
)

article_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [article_system_prompt, article_human_prompt]

article_prompt = ChatPromptTemplate(
    input_variables=["context", "question"], messages=messages
)

if GPT_MODE == 1:
    model = ChatOpenAI(model=GPT_MODEL,
                       temperature=GPT_TEMPERATURE,
                       api_key=OPENAI_API_KEY)
else:
    model = OllamaLLM(model=ROBIN_QA_MODEL, 
                      temperature=0, 
                      base_url=os.getenv('OLLAMA_BASE_URL', default='http://localhost:11434'))

article_retrieval_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=bm25_retriever,
)

article_retrieval_chain.combine_documents_chain.llm_chain.prompt = article_prompt