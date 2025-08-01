from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from sample_hunter.config import DEFAULT_EMBEDDING_DIM

QDRANT_PORT: str = "http://localhost:6333"


if __name__ == "__main__":

    # first, start the client
    client = QdrantClient(url=QDRANT_PORT)

    # create the qdrant collection
    client.create_collection(
        collection_name="dev",
        vectors_config=VectorParams(size=)
    )
