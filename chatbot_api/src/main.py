from fastapi import FastAPI
from agents.robin_rag_agent import robin_rag_agent_executor
from models.robin_rag_query import RoBInQueryInput, RoBInQueryOutput
from utils.async_utils import async_retry

app = FastAPI(
    title="RoBIn Chatbot",
    description="Endpoints for RoBIn system RAG chatbot",
)

@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues
    to external APIs.
    """
    return await robin_rag_agent_executor.ainvoke({"input": query})

@app.get("/")
async def get_status():
    return {"status": "running"}

@app.post("/robin-rag-agent")
async def query_robin_agent(query: RoBInQueryInput) -> RoBInQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response