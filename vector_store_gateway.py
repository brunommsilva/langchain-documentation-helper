import os
from chunk import Chunk
from uuid import uuid4

from langchain_core.embeddings import Embeddings
from langchain_ollama import OllamaEmbeddings
from pinecone import QueryResponse, ServerlessSpec
from pinecone.grpc import GRPCClientConfig, PineconeGRPC


class VectorStoreGateway:
    def __init__(self, embeddings: Embeddings | None = None):
        self.pc = PineconeGRPC(
            api_key=os.environ["PINECONE_API_KEY"],
            host=os.environ["PINECONE_HOST"],
        )
        self.embeddings = embeddings or OllamaEmbeddings(
            model="nomic-embed-text:latest"
        )

    def __get_index(self, index_name: str):
        """Get a Pinecone index with TLS disabled."""
        index_host = self.pc.describe_index(name=index_name).host
        return self.pc.Index(
            host=index_host, grpc_config=GRPCClientConfig(secure=False)
        )

    def init_index(self, index_name: str) -> None:
        """Initialize an index by deleting if exists and creating new."""
        if self.pc.has_index(index_name):
            print(f"Deleting index '{index_name}'...")
            self.pc.delete_index(index_name)

        print(f"Creating index '{index_name}'...")
        self.pc.create_index(
            name=index_name,
            vector_type="dense",
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            deletion_protection="disabled",
            tags={"environment": os.environ["PINECONE_ENVIRONMENT"]},
        )

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        embeddings = []
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            batch_embeddings = self.embeddings.embed_documents(batch)
            embeddings.extend(batch_embeddings)

        return embeddings

    def embed_query(self, query: str) -> list[float]:
        return self.embeddings.embed_query(query)

    def store_vectors(
        self, index_name: str, chunks: list[Chunk], vectors: list[list[float]]
    ) -> None:
        index = self.__get_index(index_name)
        batch_size = 100

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]
            batch_vectors = vectors[i : i + batch_size]

            batch_data = [
                {
                    "id": str(uuid4()),
                    "values": embedding,
                    "metadata": {"text": chunk.content, "source": chunk.source},
                }
                for embedding, chunk in zip(batch_vectors, batch_chunks)
            ]

            index.upsert(vectors=batch_data)
            print(
                f"Stored batch {i // batch_size + 1} ({len(batch_data)} embeddings) in index '{index_name}'"
            )

    def query(self, index_name: str, query: str, top_k: int = 5) -> list[Chunk]:
        embedded_query = self.embed_query(query)
        index = self.__get_index(index_name)
        response: QueryResponse = index.query(
            vector=embedded_query,
            include_metadata=True,
            top_k=top_k,
        )
        chunks = []
        for match in response["matches"]:
            metadata = match["metadata"]
            # print(f"Score: {score}, Text: {metadata['text']}")
            chunk = Chunk(content=metadata["text"], source=metadata["source"])
            chunks.append(chunk)

        return chunks
