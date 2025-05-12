# RoBIn Chatbot

RoBIn (Risk of Bias Inference) is an AI‐powered chatbot specialized in:
- Performing file‐based question answering over clinical study documents (TXT, PDF).
- Automatic risk‐of‐bias evaluation for clinical trial descriptions using a trained linear classifier.

> **Note:** Although the full RoBIn system supports additional tools (Neo4j graph queries, PubMed retrieval), this guide focuses on using only the File QA chain and the RoB classifier tool.

## Features

- **File QA**: Upload a study file and ask domain‐specific questions about its contents.
- **Risk‐of‐Bias (RoB) Inference**: Evaluate the risk of bias for a study description with a single API call.

## Prerequisites

- Docker & Docker Compose (recommended)
- (Optional) Python 3.10+ & pip (for local development)
- Pretrained RoB classifier weights:
  - Place your pretrained classifier in `./data/models/linear_classifier_4e-05/pretrained/`, so it is mounted into the container at `/models/linear_classifier_4e-05/pretrained/`.

## Getting Started

1. Clone the repository:
   ```bash
   git clone https://github.com/<your‐org>/robin-chatbot.git
   cd robin-chatbot
   ```
2. Create a `.env` file in the project root with the following variables:
   ```dotenv
   # Choose LLM backend: 0 = Ollama, 1 = OpenAI
   GPT_MODE=0

   # Ollama (local) settings
   OLLAMA_BASE_URL=http://localhost:11434
   ROBIN_FILE_CHAIN_MODEL=gemma2

   # If you want to use OpenAI instead:
   # GPT_MODE=1
   # GPT_MODEL=gpt-4
   # GPT_TEMPERATURE=0.2
   # OPENAI_API_KEY=your_api_key_here
   ```
3. Place the pretrained RoB classifier:
   ```bash
   mkdir -p data/models/linear_classifier_4e-05/pretrained
   # Copy your HuggingFace‐style model checkpoint files into data/models/linear_classifier_4e-05/pretrained/
   ```
4. Start the services via Docker Compose:
   ```bash
   docker-compose up -d
   ```
   This will launch:
   - **robin_neo4j_etl** (loads Cochrane evaluations into Neo4j)
   - **chatbot_api** (FastAPI server with File QA and RoB tool)
   - **chatbot_frontend** (Streamlit UI on port 8501)
   - Other auxiliary services

## Usage

### 1. File Question Answering

Send a `POST` request to `http://localhost:8000/robin-file-agent` with:
- `query_text` (string): Your question about the uploaded document.
- `filename` (string): Name of the file (e.g., `study.pdf` or `study.txt`).
- `uploaded_file`: The file payload (form field).

#### Example (curl)
```bash
curl -X POST "http://localhost:8000/robin-file-agent?query_text=What%20are%20the%20primary%20outcomes%3F&filename=study.txt" \
  -F "uploaded_file=@/path/to/study.txt"
```

#### Sample Response
```json
{
  "answer": {
    "output": "The primary outcomes were...",
    "intermediate_steps": [
      "Thought: ...",
      "Action: FileQAChain",
      "Observation: ...",
      "Final Answer: The primary outcomes were..."
    ]
  }
}
```

### 2. Risk‐of‐Bias Classification

Send a `POST` request to `http://localhost:8000/robin-rag-agent` with JSON:
```json
{
  "text": "Evaluate the risk of bias. Context: The study was randomized with proper allocation concealment.",
  "session": "my-session-id"
}
```

#### Example (curl)
```bash
curl -X POST http://localhost:8000/robin-rag-agent \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Evaluate the risk of bias. Context: The study was randomized with proper allocation concealment.",
    "session": "session123"
  }'
```

#### Sample Response
```json
{
  "output": "Low risk of bias",
  "intermediate_steps": [
    "Thought: ...",
    "Action: RoBIn Classifier",
    "Observation: Low risk of bias",
    "Final Answer: Low risk of bias"
  ]
}
```

## Local Development (Optional)

If you prefer to run the API without Docker:
```bash
cd chatbot_api
pip install -r requirements.txt
uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

Ensure your `.env` is set and the classifier weights are in `data/models/...` as above.

## Further Reading

- Explore the Streamlit front end at `http://localhost:8501` for an interactive UI.
- See `chatbot_api/src/chains/file_qa_chain.py` and `chatbot_api/src/chains/robin_tool_chain.py` for implementation details.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.