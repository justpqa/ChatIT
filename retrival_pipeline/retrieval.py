from llama_index import QueryBundle, ServiceContext
from llama_index.retrievers import BaseRetriever
from llama_index.vector_stores import VectorStoreQuery
from llama_index.schema import NodeWithScore
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llms import LlamaCPP
from flask import Flask, jsonify

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

    def __init__(
        self,
        vector_store,
        embed_model,
        query_mode = "default",
        similarity_top_k = 2,
    ) -> None:
        """Init params."""
        self._vector_store = vector_store
        self._embed_model = embed_model
        self._query_mode = query_mode
        self._similarity_top_k = similarity_top_k
        super().__init__()

    def _retrieve(self, query_bundle: QueryBundle) -> List[NodeWithScore]:
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

def init_query_engine():
    """
    This function will return a query engine when initialize the app that allow for new requests 
    """
    
    # need api key and environment
    api_key = os.environ.get("PINECONE_API_KEY")
    environment = os.environ.get("PINECONE_ENV")

    # initialize pinecone env
    pinecone.init(api_key=api_key, environment=environment)

    # initialize index
    pinecone_index = pinecone.Index(os.environ.get("PINECONE_INDEX"))

    # initialize vector store and embed model from this
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    embed_model = HuggingFaceEmbedding(model_name=os.environ.get("DEFAULT_EMBED_MODEL"))
    retriever = PineconeRetriever(
        vector_store, embed_model, query_mode="default", similarity_top_k=2
    )

    llm = LlamaCPP(
        # You can pass in the URL to a GGML model to download it automatically
        model_url=os.environ.get("LLAMA_CPP"),
        # optionally, you can set the path to a pre-downloaded model instead of model_url
        model_path=None,
        temperature=0.1,
        max_new_tokens=256,
        # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
        context_window=3900,
        # kwargs to pass to __call__()
        generate_kwargs={},
        # kwargs to pass to __init__()
        # set to at least 1 to use GPU
        model_kwargs={"n_gpu_layers": 1},
        verbose=True,
    )

    service_context = ServiceContext.from_defaults(
        llm=llm, embed_model=embed_model
    )

    query_engine = RetrieverQueryEngine.from_args(
        retriever, service_context=service_context
    )
    
    return query_engine

def create_app():
    """
    This function will return a Flask app with retrieval engine initialization 
    """
    
    # initialize the app
    app = Flask(__name__)
    
    # App configurations
    app.config["SECRET_KEY"] = os.environ.env("FLASK_SECRET_KEY")

    # Set up query engine
    query_engine = init_query_engine()
    
    # define the query method
    @app.route('/retrieve/<query_str>', methods=['GET'])
    def retrieve(query_str):
        response = query_engine.query(query_str)
        return jsonify({"response": str(response)})

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)