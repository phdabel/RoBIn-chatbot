import os
from typing import ClassVar
from typing import Optional, Dict, Any
from pydantic import PrivateAttr
from langchain.chains.base import Chain
from langchain_community.vectorstores import Chroma
# from langchain.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.language_models.llms import BaseLLM
from langchain_core.embeddings import Embeddings
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_openai import ChatOpenAI
import logging

from agents.robin_rag_agent import robin_rag_agent_executor

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)


class FileQAChain(Chain):
    
    _llm_model: BaseLLM = PrivateAttr()
    _embeddings: Embeddings = PrivateAttr()
    input_keys: ClassVar[list[str]] = ["query_text", "uploaded_file"]
    output_keys: ClassVar[list[str]] = ["answer"]
        
    def __init__(self, llm_model: BaseLLM, embeddings: Embeddings):
        super().__init__()
        self._llm_model = llm_model
        self._embeddings = embeddings

    def _call(self, inputs: Dict[str, Any], run_manager: Optional[Any]=None):
        
        query_text = inputs["query_text"]
        uploaded_file = inputs["uploaded_file"]
        filename = inputs["filename"]
        
        if uploaded_file is not None:
            try:
                if filename.endswith(".pdf"):
                    texts = PyMuPDFLoader(f"/tmp/{filename}").load()
                else:
                    document = uploaded_file.read().decode()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=64, add_start_index=True)
                    texts = text_splitter.create_documents([document])
            except UnicodeDecodeError:
                return {"answer": "Error decoding file. Ensure the file is in UTF-8 format."}
            
            db = Chroma.from_documents(documents=texts, embedding=self._embeddings) #, persist_directory=os.getenv('CHROMA_DB_PATH'))

            qa_chain = MultiQueryRetriever.from_llm(
                llm=self._llm_model,
                retriever=db.as_retriever(k=5)
            )
            
            answer = qa_chain.invoke(query_text)

            if filename.endswith(".pdf"):
                os.unlink(f"/tmp/{filename}")
                answers = {i: doc.page_content for i, doc in enumerate(answer)}
            else:
                answers = {doc.metadata['start_index']: doc.page_content for doc in answer}
            context = "\n".join([answers[i] for i in list(answers.keys())])
            prompt = f"""Context:
{context}

Instructions:
{query_text}

Given the context of the provided file, answer the question and follow the instructions provided by the user.
Be as detailed as possible and do no make assumptions about the context. If you are unsure about the answer, please state so."""
            # answer = self._llm_model.invoke(prompt)

            # return {"answer": {"output": answer, "intermediate_steps": []}}
            answer = robin_rag_agent_executor.invoke({'input': prompt})
            answer["intermediate_steps"] = [
                str(s) for s in answer["intermediate_steps"]
            ]

            return {"answer": answer}
        else:
            return {"answer": {"output": "No file uploaded", "intermediate_steps": []}}

ROBIN_FILE_CHAIN_MODEL = os.getenv("ROBIN_FILE_CHAIN_MODEL")
GPT_MODE = int(os.getenv("GPT_MODE"))
GPT_MODEL = os.getenv("GPT_MODEL")
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=os.getenv('OLLAMA_BASE_URL', default='http://localhost:11434'))#base_url=os.getenv('OLLAMA_BASE_URL', default='http://host.docker.internal:11434'))

if GPT_MODE == 1:
    model = ChatOpenAI(model=GPT_MODEL,
                       temperature=GPT_TEMPERATURE,
                       api_key=OPENAI_API_KEY)
else:
    model = OllamaLLM(model=ROBIN_FILE_CHAIN_MODEL,
                      temperature=0, 
                      base_url=os.getenv('OLLAMA_BASE_URL', default='http://localhost:11434'))

file_qa_chain = FileQAChain(llm_model=model, embeddings=embeddings)
    