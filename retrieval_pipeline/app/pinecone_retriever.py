from llama_index import QueryBundle, ServiceContext
from llama_index.retrievers import BaseRetriever
from llama_index.vector_stores import PineconeVectorStore, VectorStoreQuery
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import NodeWithScore
from llama_index.query_engine import RetrieverQueryEngine

class PineconeRetriever(BaseRetriever):
    """
    This class is a Retriver based on BaseRetriever for Pinecone DB

    Attributes:
        _vector_store: the vector store that we query from
        _embed_model: the model used to embed the input string
        _query_mode: the mode for querying, can be one of "default", "sparse", "hybrid"
        _similarity_top_k: the number of nodes that we retrives from the vector store

    Methods:
        _retrieve(self, query_bundle): method for getting the results from a query
    """

    def __init__(self, vector_store, embed_model, query_mode = "default", similarity_top_k = 2):
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle):
        """Retrieve."""
        query_embedding = embed_model.get_query_embedding(
            query_bundle.query_str
        )
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self._similarity_top_k,
            mode=self._query_mode,
        )
        query_result = vector_store.query(vector_store_query)

        nodes_with_scores = []
        for index, node in enumerate(query_result.nodes):
            score = None
            if query_result.similarities is not None:
                score = query_result.similarities[index]
            nodes_with_scores.append(NodeWithScore(node=node, score=score))

        return nodes_with_scores