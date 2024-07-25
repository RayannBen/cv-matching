"""Generate and print answer for a given query."""
from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient

from src.vdb.assistant import RagAssistant

load_dotenv()

openai_client = OpenAI()
qdrant_client = QdrantClient(url="http://localhost:6333")

rag_assistant = RagAssistant(openai_client=openai_client, qdrant_client=qdrant_client)


if __name__ == "__main__":
    query = "Who is the best candidate to analyse the safety of a plane project?"
    chunks = rag_assistant.retrieve(query, collection_name="streamlit_people")
    print("chunks: ", chunks)
    print(
        """ \n\nAnswer: """,
        rag_assistant.generate_answer(query, [chunk["content"] for chunk in chunks]),
    )
