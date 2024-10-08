import os
# from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_community.vectorstores import Neo4jVector
from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)

ROBIN_QA_MODEL = os.getenv("ROBIN_QA_MODEL")
GPT_MODE = int(os.getenv("GPT_MODE"))
GPT_MODEL = os.getenv("GPT_MODEL")
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

neo4j_vector_index = Neo4jVector.from_existing_graph(
    embedding=OllamaEmbeddings(model="mxbai-embed-large", base_url=os.getenv('OLLAMA_BASE_URL', default='http://localhost:11434')),
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
    index_name="support",
    node_label="Evaluation",
    text_node_properties=[
        "result",
        "rob_judgment",
        "support_judgment"
    ],
    embedding_node_property="embedding",
)

review_template = """Your job is to use systematic reviews evaluations
to answer questions about risk of bias in clinical trials. Use
the following context to answer questions. Be as detailed as possible, but
don't make up any information that's not from the context. If you don't know
an answer, say you don't know.
{context}
"""

review_system_prompt = SystemMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["context"], template=review_template)
)

review_human_prompt = HumanMessagePromptTemplate(
    prompt=PromptTemplate(input_variables=["question"], template="{question}")
)
messages = [review_system_prompt, review_human_prompt]

review_prompt = ChatPromptTemplate(
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

reviews_vector_chain = RetrievalQA.from_chain_type(
    llm=model,
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)

reviews_vector_chain.combine_documents_chain.llm_chain.prompt = review_prompt