from fastapi import FastAPI, UploadFile
from agents.robin_rag_agent import robin_rag_agent_executor
from chains.file_qa_chain import file_qa_chain
from models.robin_rag_query import RoBInQueryInput, RoBInQueryOutput, RoBInFileOutput
from utils.async_utils import async_retry

app = FastAPI(
    title="RoBIn Chatbot",
    description="Endpoints for RoBIn system RAG chatbot",
)

@async_retry(max_retries=3, delay=1)
async def invoke_agent_with_retry(query: str, session: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues
    to external APIs.
    """
    return await robin_rag_agent_executor.ainvoke({"input": query}, config={"configurable": {"session_id": session}})


@async_retry(max_retries=3, delay=1)
async def invoke_file_agent_with_retry(query_text: str, uploaded_file: UploadFile):
    return await file_qa_chain.ainvoke({"query_text": query_text, "uploaded_file": uploaded_file.file})

@app.get("/")
async def get_status():
    return {"status": "running"}


@app.post("/robin-rag-agent")
async def query_robin_agent(query: RoBInQueryInput) -> RoBInQueryOutput:
    query_response = await invoke_agent_with_retry(query.text, query.session)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response

@app.post("/robin-file-agent")
async def query_robin_file_agent(query_text: str, uploaded_file: UploadFile) -> RoBInFileOutput:
    query_response = await invoke_file_agent_with_retry(query_text, uploaded_file)
    return query_response