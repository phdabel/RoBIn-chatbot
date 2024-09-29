import os
from typing import ClassVar
from typing import Optional, Dict, Any
from pydantic import PrivateAttr
from langchain.schema.runnable import Runnable
from langchain.chains.base import Chain
from langchain.vectorstores import Chroma
from ast import literal_eval
# from langchain.text_splitter import CharacterTextSplitter
# from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_core.language_models.llms import BaseLLM
from langchain_core.embeddings import Embeddings
from langchain.chains import RetrievalQA
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from agents.robin_rag_agent import robin_rag_agent_executor


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
        
        if uploaded_file is not None:
            try:
                document = uploaded_file.read().decode()
            except UnicodeDecodeError:
                return {"answer": "Error decoding file. Ensure the file is in UTF-8 format."}

            # text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            # texts = text_splitter.split_text(document)
            # texts = text_splitter.create_documents(texts)

            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100, add_start_index=True)
            texts = text_splitter.create_documents([document])
            
            # text_splitter = SemanticChunker(embeddings=self._embeddings, number_of_chunks=100)
            # texts = text_splitter.create_documents([document])

            db = Chroma.from_documents(documents=texts, embedding=self._embeddings)#, persist_directory=os.getenv('CHROMA_DB_PATH'))
            qa_chain = MultiQueryRetriever.from_llm(
                llm=self._llm_model,
                retriever=db.as_retriever(k=16)
            )
            # qa_chain = RetrievalQA.from_chain_type(llm=self._llm_model,
            #                                        chain_type="stuff",
            #                                        retriever=db.as_retriever(k=100))
            
            answer = qa_chain.invoke(query_text)
            answers = {doc.metadata['start_index']: doc.page_content for doc in answer}
            context = "\n".join([answers[i] for i in sorted(answers.keys())])
            prompt = f"""Given the context of the provided file, answer the question and follow the instructions provided by the user.
Be as detailed as possible and do no make assumptions about the context. If you are unsure about the answer, please state so.


Context:
{context}

Instructions:
{query_text}"""
            # answer = self._llm_model.invoke(prompt)

            # return {"answer": answer}
            answer = robin_rag_agent_executor.invoke({'input': prompt})
            answer["intermediate_steps"] = [
                str(s) for s in answer["intermediate_steps"]
            ]

            return {"answer": answer}
        else:
            return {"answer": {"output": "No file uploaded", "intermediate_steps": []}}
        

embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url=os.getenv('OLLAMA_BASE_URL', default='http://localhost:11434'))#base_url=os.getenv('OLLAMA_BASE_URL', default='http://host.docker.internal:11434'))
ollama_model = OllamaLLM(model="gemma2",
                         temperature=0,
                         base_url=os.getenv('OLLAMA_BASE_URL', default='http://localhost:11434'))#default='http://host.docker.internal:11434'))
file_qa_chain = FileQAChain(llm_model=ollama_model, embeddings=embeddings)

    