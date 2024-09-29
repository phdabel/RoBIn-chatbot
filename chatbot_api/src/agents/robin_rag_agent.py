import os
from langchain_ollama import OllamaLLM
from langchain_core.prompts.prompt import PromptTemplate

from langchain_core.tools import render_text_description
from langchain.agents import (
    create_react_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub

# from chatbot_api.src.memory.BaseMemory import ModifiedConversationBufferMemory
# from chatbot_api.src.chains.cochrane_cypher_chain import cochrane_cypher_chain
# from chatbot_api.src.chains.review_study_chain import reviews_vector_chain
# from chatbot_api.src.chains.pubmed_article_chain import article_retrieval_chain

from memory.BaseMemory import ModifiedConversationBufferMemory
from chains.cochrane_cypher_chain import cochrane_cypher_chain
from chains.review_study_chain import reviews_vector_chain
from chains.pubmed_article_chain import article_retrieval_chain


ROBIN_AGENT_MODEL = os.getenv("ROBIN_AGENT_MODEL", default="gemma2")

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
        name="ArticleIndex",
        func=article_retrieval_chain.invoke,
        description="""Useful for answering questions about clinical trial 
        publications, when searching for articles, papers or studies. 
        Use the entire prompt as input to the tool. For instance, if the prompt is 
        "What interventions are studied in clinical trials about COVID-19 treatment?", 
        the input should be "What interventions are studied in clinical trials about 
        COVID-19 treatment?". If the prompt is "Search for articles about cardiac diseases.",
        the input should be "Search for articles about cardiac diseases.".
        """,
    ),
]

robin_agent_prompt_template = """RoBIn is a large language model created at INF-UFRGS.

RoBIn is designed to be able to assist with a wide range of tasks, from answering simple questions to providing in-depth explanations and discussions on a wide range of topics. As a language model, RoBIn is able to generate human-like text based on the input it receives, allowing it to engage in natural-sounding conversations and provide responses that are coherent and relevant to the topic at hand.

RoBIn is constantly learning and improving, and its capabilities are constantly evolving. It is able to process and understand large amounts of text, and can use this knowledge to provide accurate and informative responses to a wide range of questions. Additionally, RoBIn is able to generate its own text based on the input it receives, allowing it to engage in discussions and provide explanations and descriptions on a wide range of topics.

Overall, RoBIn is a powerful tool that can help with a wide range of tasks and provide valuable insights and information on a wide range of topics. Whether you need help with a specific question or just want to have a conversation about a particular topic, RoBIn is here to assist.

TOOLS:

------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```

Thought: Do I need to use a tool? Yes

Action: the action to take, should be one of [{tool_names}]

Action Input: the input to the action

Observation: the result of the action

```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```

Thought: Do I need to use a tool? No

Final Answer: [your response here]

```

Begin!

Previous conversation history:
{chat_history}

New input: {input}

{agent_scratchpad}
"""

robin_agent_prompt = PromptTemplate(input_variables=['input', 'chat_history'], template=robin_agent_prompt_template)


# robin_agent_prompt = hub.pull("hwchase17/react-chat")

robin_agent_prompt.partial(
    tools=render_text_description(tools),
    tool_names=", ".join([t.name for t in tools])
)

chat_model = OllamaLLM(
    base_url=os.getenv('OLLAMA_BASE_URL', default='http://host.docker.internal:11434'),
    model=ROBIN_AGENT_MODEL,
    temperature=0,
)

memory = ModifiedConversationBufferMemory(memory_key="chat_history", output_key="output", input_key="input")

robin_rag_agent = create_react_agent(
    chat_model,
    prompt=robin_agent_prompt,
    tools=tools
)

robin_rag_agent_executor = AgentExecutor(
    agent=robin_rag_agent,
    tools=tools,
    handle_parsing_errors=True,
    return_intermediate_steps=True,
    memory=memory,
    verbose=True
)

