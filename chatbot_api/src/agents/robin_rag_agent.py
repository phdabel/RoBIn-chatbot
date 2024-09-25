import os
from langchain_ollama import OllamaLLM
from langchain_core.tools import render_text_description
from langchain.agents import (
    create_react_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub
# from chatbot_api.src.chains.review_study_chain import reviews_vector_chain
# from chatbot_api.src.chains.pubmed_article_chain import article_retrieval_chain
from chains.cochrane_cypher_chain import cochrane_cypher_chain
from chains.review_study_chain import reviews_vector_chain
from chains.pubmed_article_chain import article_retrieval_chain


ROBIN_AGENT_MODEL = os.getenv("ROBIN_AGENT_MODEL", default="gemma2")

robin_agent_prompt = hub.pull("hwchase17/react")

tools = [
    Tool(
        name="Graph",
        func=cochrane_cypher_chain.invoke,
        description="""Useful for answering questions about studies and references, 
        systematic reviews, evaluations, and bias types using a Neo4j graph database. 
        Use the entire prompt as input to the tool. For instance, if the prompt is 
        "How many systematic reviews there are in the graph.", the input should be 
        "How many systematic reviews there are in the graph.".
        """,
    ),
    Tool(
        name="RoB",
        func=reviews_vector_chain.invoke,
        description="""Useful when you need to answer questions
        about risk of bias in clinical trials using systematic reviews
        evaluations as ground truth. Not useful for answering questions
        that require information from PubMed articles. Use the entire 
        prompt as input to the tool. For instance, if the prompt is
        "Summarize supporting sentences from low risk of bias trials.", the input should be
        "Summarize supporting sentences from low risk of bias trials.".
        """,
    ),
    Tool(
        name="PubMed",
        func=article_retrieval_chain.invoke,
        description="""Useful for answering questions about clinical trial 
        publications from PubMed. Use the entire prompt as input to the tool. 
        For instance, if the prompt is "What interventions are studied in clinical 
        trials about COVID-19 treatment?", the input should be "What interventions 
        are studied in clinical trials about COVID-19 treatment?".
        """,
    ),
]

robin_agent_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools])
)

chat_model = OllamaLLM(
    base_url=os.getenv('OLLAMA_BASE_URL', default='http://host.docker.internal:11434'),
    model=ROBIN_AGENT_MODEL,
    temperature=0,
)

robin_rag_agent = create_react_agent(
    llm=chat_model,
    prompt=robin_agent_prompt,
    tools=tools,
)

robin_rag_agent_executor = AgentExecutor(
    agent=robin_rag_agent,
    tools=tools,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    verbose=True,
)