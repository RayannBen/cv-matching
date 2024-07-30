"""Home page of the Streamlit app."""


import base64
import json
import tempfile
from collections import OrderedDict
from pathlib import Path

import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
# print("Current working directory: ", os.getcwd())
# os.listdir(os.getcwd())
from src.streamlit_app.utils.set_page_config import set_page_config
from src.vdb.assistant import RagAssistant
from src.vdb.load_people import load_people

PEOPLE_INFORMATIONS_PATH = Path("data/people_informations.json")
with open(PEOPLE_INFORMATIONS_PATH) as file:
    people_informations = json.load(file)


def retrieve_name_from_source(source: str) -> str:
    """Retrieve the name of a person from the source."""
    if "/" in source:
        return source.split("/")[-1][:-4]
    return source[:-4]


def display_person_information(person_information: dict[str, str]) -> None:
    """Display the information of a person."""
    name = person_information["name"]
    age = person_information["age"]
    job = person_information["job"]
    st.markdown(f"# {name}")
    st.markdown(f"## Age: {age} - Job: {job}")

    pdf_path = Path(f"data/people/{name}.pdf")
    if pdf_path.exists():
        with st.expander("Preview of PDF"):
            with open(pdf_path, "rb") as pdf_file:
                base64_pdf = base64.b64encode(pdf_file.read()).decode("utf-8")
            pdf_display = f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="700" height="1000" type="application/pdf"></iframe>'
            st.markdown(pdf_display, unsafe_allow_html=True)
    else:
        file_path = Path(f"data/people/{name}.txt")
        if file_path.exists():
            with st.expander("Preview of Text"), open(file_path) as file:
                st.write(file.read())


def select_people(best_vectors: list[dict[str, str]]) -> list[str]:
    """Select the people from the best."""
    return list(
        OrderedDict.fromkeys(
            [retrieve_name_from_source(vector["source"]) for vector in best_vectors]
        )
    )


def display_people(
    rag_assistant: RagAssistant,
    input_text: str,
    people_informations: dict[str, dict[str, str]] = people_informations,
) -> None:
    """Display the people related to the input text."""
    best_vectors = rag_assistant.retrieve(
        input_text, collection_name="streamlit_people"
    )
    ordered_people = select_people(best_vectors)
    ordered_people_informations = [
        people_informations[person] for person in ordered_people
    ]
    for person_information in ordered_people_informations:
        display_person_information(person_information)


def main() -> None:
    """Home page of the Streamlit app."""
    set_page_config()
    st.title("Welcome to CV Matching")
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

    choice = st.radio("Choose an action", ("Upload a File", "Type a Search"))

    if choice == "Upload a File":
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
            display_people(rag_assistant, pdf_text)
    elif choice == "Type a Search":
        input_text = st.text_input("Enter Text")
        if st.button("Retrieve Vectors"):
            st.write("input_text: ", input_text)
            # Retrieve top k vectors related to the given text
            display_people(rag_assistant, input_text)


if __name__ == "__main__":
    main()
