import os
import requests
import streamlit as st

CHATBOT_URL = os.getenv("CHATBOT_URL", "http://localhost:8000/robin-rag-agent")
FILE_CHATBOT_URL = os.getenv("FILE_CHATBOT_URL", "http://localhost:8000/robin-file-agent")

with st.sidebar:
    st.header("About")
    st.markdown(
        """
        I am RoBIn (Risk of Bias Inference), an AI
        agent designed to answer questions about PubMed articles indexed in my database,
        and risk of bias evaluation from Cochrane systematic reviews. I make use of a
        retrieval-augment generation (RAG) models to answer questions about
        clinical trials and risk of bias evaluations.
        """
    )

    st.header("Example Questions")
    st.markdown("- Describe the randomization method of this study; if it is not stated in the document do not invent or assume anything about it. Just say, that it is not informed in the text. Use the RoB tool to evaluate the risk of bias in the study, regarding selection bias (random sequence generation), given the resultant description.")
    st.markdown("- What are the main outcomes of the study? Summarize the results of the study.")
    st.markdown("- What are the concerns about randomization in the risk of bias assessment?")
    st.markdown("- What interventions are being studied to treat ADHD?")
    st.markdown("- What are the results of the clinical trial on COVID-19 treatment?")
    st.markdown("- Summarize supporting sentences from low risk of bias trials.")
    st.markdown("- Which interventions are studied in clinical trials about COVID-19 treatment?")


st.title("RoBIn Chatbot")
st.info(
    "Ask me questions about clinical trials, PubMed articles, and risk of bias evaluations."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if "output" in message.keys():
            st.markdown(message["output"])

        if "explanation" in message.keys():
            with st.status("How was this generated", state="complete"):
                st.info(message["explanation"])

prompt = st.chat_input("What do you want to know?")
uploaded_file = st.file_uploader("Upload a file", type=["txt"])

if prompt:

    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append({"role": "user", "output": prompt})

    data = {"text": prompt}

    with st.spinner("Searching for an answer..."):
        if uploaded_file is not None:
            response = requests.post(FILE_CHATBOT_URL, 
                                     params={"query_text": prompt, "filename": uploaded_file.name},
                                     files={"uploaded_file": uploaded_file.getvalue()})
        else:
            response = requests.post(CHATBOT_URL, json=data)

        if response.status_code == 200 and uploaded_file is None:
            output_text = response.json()["output"]
            explanation = response.json()["intermediate_steps"]
        elif response.status_code == 200 and uploaded_file is not None:
            output_text = response.json()["answer"]["output"]
            explanation = response.json()["answer"]["intermediate_steps"]

        else:
            output_text = """An error occurred while processing your message.
            Please try again or rephrase your message."""
            explanation = output_text

    st.chat_message("assistant").markdown(output_text)
    st.status("How was this generated", state="complete").info(explanation)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "output": output_text,
            "explanation": explanation,
        }
    )

# file input
# uploaded_file = st.file_uploader("Upload a file", type=["txt"])
# if uploaded_file and prompt:
#     st.session_state.messages.append({"role": "user", "output": "File uploaded"})

#     data = {"query_text": prompt, "uploaded_file": uploaded_file}

#     with st.spinner("Searching for an answer..."):
#         response = requests.post(FILE_CHATBOT_URL, json=data)
