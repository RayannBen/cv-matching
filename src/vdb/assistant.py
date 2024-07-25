"""A module to represent a RAG assistant."""
from openai import OpenAI
from qdrant_client import QdrantClient

DEFAULT_COLLECTION_NAME = "people_collection"
TOP_VECTOR_SEARCH = 10


class RagAssistant:
    """A class to represent a RAG assistant."""

    def __init__(
        self,
        openai_client: OpenAI,
        qdrant_client: QdrantClient,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        top_k_vectors: int = TOP_VECTOR_SEARCH,
        model: str = "gpt-3.5-turbo",
    ):
        """Initialize the assistant."""
        self.openai_client = openai_client
        self.qdrant_client = qdrant_client
        self.top_k_vectors = top_k_vectors
        self.model = model

    def retrieve(
        self, query: str, collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> list[dict[str, str]]:
        """Retrieve the top k vectors related to the given question from a Qdrant DB."""
        embedded_query = (
            self.openai_client.embeddings.create(
                input=[query],
                model="text-embedding-ada-002",
            )
            .data[0]
            .embedding
        )
        query_results = self.qdrant_client.search(
            collection_name=collection_name,
            query_vector=embedded_query,
            limit=self.top_k_vectors,
        )
        return [
            vector.payload for vector in query_results if vector.payload is not None
        ]

    def generate_answer(self, query: str, context_chunks: list[str]) -> str | None:
        """Generate an answer using OpenAI with added context base on a Qdrant DB."""
        context = "\n".join(context_chunks)
        response = self.openai_client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": """
                    You are a helpful assistant.
                    You will be provided with a document delimited by "Context:" and a question.
                    Your task is to answer the question using only the provided document.
                    Give the most relevant pieces of information in the document that answer the given question.
                    """,
                },
                {"role": "user", "content": f"Context: {context}"},
                {"role": "user", "content": f"{query}"},
            ],
            temperature=0,
        )
        return response.choices[0].message.content

    def query(
        self, query: str, collection_name: str = DEFAULT_COLLECTION_NAME
    ) -> str | None:
        """Query the assistant."""
        context_chunks = [
            chunk["content"] for chunk in self.retrieve(query, collection_name)
        ]
        answer = self.generate_answer(query, context_chunks)
        return answer
