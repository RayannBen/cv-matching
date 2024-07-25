"""Home page of the Streamlit app."""


import tempfile
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from src.streamlit_app.utils.set_page_config import set_page_config
from src.vdb.assistant import RagAssistant
from src.vdb.load_people import load_people


def main() -> None:
    """Home page of the Streamlit app."""
    set_page_config()
    st.title("Welcome to Expert Connect")
    openai_client = OpenAI()
    try:
        qdrant_client = QdrantClient(
            host="localhost",
            prefer_grpc=True,
        )
    except Exception as e:
        st.error(f"Could not connect to Qdrant: {e}")
        return

    rag_assistant = RagAssistant(openai_client, qdrant_client)
    st.write("Connected to Qdrant")

    if st.button("Load Collection"):
        COLLECTION_NAME = "streamlit_people"
        OPEN_AI_EMBEDDINGS_SIZE = 1536
        if not qdrant_client.collection_exists(COLLECTION_NAME):
            qdrant_client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=OPEN_AI_EMBEDDINGS_SIZE, distance=Distance.COSINE
                ),
            )
            load_people(
                folder_path=Path("data/people"),
                collection_name="streamlit_people",
                qdrant_client=qdrant_client,
            )
        st.write("Collection loaded")

    # openai_client = OpenAI()

    # rag_assistant = RagAssistant(openai_client, qdrant_client)
    # rag_assistant.retrieve("Who is the best suited to work on spacecraft?")

    uploaded_file = st.file_uploader("Upload Demande de prestation", type="pdf")
    if uploaded_file is not None:
        # Load PDF content using PyPDFLoader
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_file_path = tmp_file.name
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        pdf_text = " ".join([doc.page_content for doc in documents])

        # Retrieve top k vectors related to the given question
        best_vectors = rag_assistant.retrieve(
            pdf_text, collection_name="streamlit_people"
        )

        st.write("Best vectors retrieved")
        st.write([vector["source"] for vector in best_vectors])

    input_text = st.text_input("Enter Text")
    if st.button("Retrieve Vectors"):
        # Retrieve top k vectors related to the given text
        best_vectors = rag_assistant.retrieve(
            input_text, collection_name="streamlit_people"
        )

        st.write("Best vectors retrieved")
        st.write([vector["source"] for vector in best_vectors])


if __name__ == "__main__":
    main()
