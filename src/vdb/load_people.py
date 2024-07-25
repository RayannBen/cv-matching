"""Load people data into Qdrant."""
import uuid
from pathlib import Path
from typing import Any

from langchain_community.document_loaders import PyPDFLoader
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models as rest
from qdrant_client.models import Distance, VectorParams

COLLECTION_NAME = "streamlit_people"
OPEN_AI_EMBEDDINGS_SIZE = 1536


def main(
    collection_name: str = COLLECTION_NAME, vector_size: int = OPEN_AI_EMBEDDINGS_SIZE
) -> None:
    """Create a collection in Qdrant."""
    qdrant_client = QdrantClient(
        host="localhost",
        prefer_grpc=True,
    )
    if not qdrant_client.collection_exists(collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
    load_people(
        Path("data/people"),
        collection_name=collection_name,
        qdrant_client=qdrant_client,
    )


def save_chunks_to_collection(
    qdrant_client: QdrantClient, collection_name: str, chunks: list[dict[str, Any]]
) -> None:
    """Save chunks to a collection in Qdrant, payload contains metadata."""
    for chunk in chunks:
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                rest.PointStruct(
                    id=str(uuid.uuid4()),
                    vector=chunk["embedding"],
                    payload=chunk,
                )
            ],
        )


def load_people(
    folder_path: Path, collection_name: str, qdrant_client: QdrantClient
) -> None:
    """Load all files in a folder and split them into chunks by page."""
    chunks = embed_folder(folder_path)

    save_chunks_to_collection(qdrant_client, collection_name, chunks)


def embed_folder(folder_path: Path) -> list[dict[str, Any]]:
    """Load all files in a folder and split them into chunks by page."""
    files = folder_path.glob("*")
    return [
        chunk
        for file in files
        if file.is_file()
        for chunk in chunk_and_embed_file(file)
    ]


def chunk_and_embed_file(file_path: Path) -> list[dict[Any, Any]]:
    """Load a pdf file and split it into chunks by page."""
    EMBEDDING_MODEL = "text-embedding-ada-002"
    openai_client = OpenAI()
    if file_path.suffix == ".pdf":
        loader = PyPDFLoader(str(file_path))
        pages = loader.load_and_split()
        return [
            {
                "content": page.page_content,
                "source": page.metadata["source"],
                "page_number": page.metadata["page"],
                "embedding": openai_client.embeddings.create(
                    input=[page.page_content], model=EMBEDDING_MODEL
                )
                .data[0]
                .embedding,
            }
            for page in pages
        ]
    if file_path.suffix == ".txt":
        with open(file_path) as file:
            content = file.read()
        return [
            {
                "content": content,
                "source": file_path.name,
                "page_number": 0,
                "embedding": openai_client.embeddings.create(
                    input=[content], model=EMBEDDING_MODEL
                )
                .data[0]
                .embedding,
            }
        ]
    else:
        raise ValueError("File type not supported.")


if __name__ == "__main__":
    main()
