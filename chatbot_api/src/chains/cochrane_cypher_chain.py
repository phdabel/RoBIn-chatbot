import os
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain_openai import ChatOpenAI
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate

ROBIN_CYPHER_MODEL = os.getenv("ROBIN_CYPHER_MODEL")

GPT_MODE = int(os.getenv("GPT_MODE", 0))
GPT_MODEL = os.getenv("GPT_MODEL")
GPT_TEMPERATURE = float(os.getenv("GPT_TEMPERATURE", 0))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

graph = Neo4jGraph(
    url=os.getenv("NEO4J_URI"),
    username=os.getenv("NEO4J_USERNAME"),
    password=os.getenv("NEO4J_PASSWORD"),
)

graph.refresh_schema()

cypher_generation_template = """
Task:
Generate Cypher query for a Neo4j graph database.

Instructions:
Use only the provided relationship types and properties in the schema.
Do not use any other relationship types or properties that are not provided.
Do not use group by, order by, or any other aggregation functions.
Risk of Bias are one of: LOW, HIGH, or UNCLEAR.

Schema:
{schema}

The question is:
{question}
"""

cypher_generation_prompt = PromptTemplate(
    input_variables=["schema", "question"], template=cypher_generation_template
)

qa_generation_template = """You are an assistant that takes the results
from a Neo4j Cypher query and forms a human-readable response. The
query results section contains the results of a Cypher query that was
generated based on a user's natural language question. The provided
information is authoritative, you must never doubt it or try to use
your internal knowledge to correct it. Make the answer sound like a
response to the question.

Query Results:
{context}

Question:
{question}

If the provided information is empty, say you don't know the answer.
Empty information looks like this: []

If the information is not empty, you must provide an answer using the
results. If the question involves a time duration, use the unit specified
in the information. If no unit is specified, you should not assume the
unit.

When names are provided in the query results, such as author names,
titles, or other names, beware  of any names that have commas or other punctuation in them.
For instance, 'Piette JD' refers to a single author named 'J.D. Piette'.
'Piette JD, Smith R' refers to two authors, 'J.D. Piette' and 'R. Smith'.
Make sure you return any list of names in a way that isn't ambiguous and allows someone to tell what the full
names are.

Never say you don't have the right information if there is data in
the query results. Always use the data in the query results.

Helpful Answer:
"""

qa_generation_prompt = PromptTemplate(
    input_variables=["context", "question"], template=qa_generation_template
)

if GPT_MODE == 1:
    model = ChatOpenAI(model=GPT_MODEL,
                       temperature=GPT_TEMPERATURE,
                       api_key=OPENAI_API_KEY)
else:
    model = OllamaLLM(model=ROBIN_CYPHER_MODEL, 
                      temperature=0, 
                      base_url=os.getenv('OLLAMA_BASE_URL', default='http://localhost:11434'))

cochrane_cypher_chain = GraphCypherQAChain.from_llm(
    llm=model,
    graph=graph,
    verbose=True,
    qa_prompt=qa_generation_prompt,
    cypher_prompt=cypher_generation_prompt,
    validate_cypher=True,
    top_k=100,
    allow_dangerous_requests=True
)