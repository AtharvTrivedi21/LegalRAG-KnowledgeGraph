from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from rag_chain import answer_query
from config import ACTIVE_CONFIG


st.set_page_config(page_title="BNS RAG", layout="wide")

st.title("BNS Mitra – RAG based Legal Assistant for BNS")

st.markdown(
    f"**Model config:** `{ACTIVE_CONFIG.config_name}` "
    f"(LLM: `{ACTIVE_CONFIG.llm_model_name}`, Embeddings: `{ACTIVE_CONFIG.embedding_model_name}`)"
)

if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Display chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Describe your incident or ask about a BNS provision...")

if user_input:
    # Add user message
    st.session_state["messages"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Analyzing your query using BNS..."):
            result = answer_query(user_input)

        # Show rephrased query
        st.markdown(f"**Formal legal query:** {result['rephrased_query']}")
        st.markdown("---")

        # Show answer
        st.markdown(result["answer"])

        # Show retrieved documents
        with st.expander("Show retrieved BNS sections (context)"):
            for i, d in enumerate(result["docs"]):
                st.markdown(f"**Chunk {i+1}**")
                meta = d.metadata if hasattr(d, "metadata") else {}
                if meta:
                    st.json(meta)
                st.write(d.page_content)
                st.markdown("---")

    # Save assistant answer in history (plain answer text)
    st.session_state["messages"].append(
        {"role": "assistant", "content": result["answer"]}
    )