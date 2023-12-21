from llama_index import QueryBundle, ServiceContext
from llama_index.retrievers import BaseRetriever
from llama_index.vector_stores import PineconeVectorStore, VectorStoreQuery
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.schema import NodeWithScore
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.llms import HuggingFaceLLM
from llama_index.prompts import PromptTemplate
from custom_llm import CustomLLM
from pinecone_retriever import PineconeRetriever
import torch
from flask import Flask, jsonify
import pinecone
import os

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

    query_wrapper_prompt = PromptTemplate(
    "Write a response that appropriately completes the request.\n\n"
    "### Question:\n{query_str}\n\n### Response:"
    )
        
    llm = HuggingFaceLLM(
        context_window=1024,
        max_new_tokens=256,
        generate_kwargs={"temperature": 0.25, "do_sample": False},
        query_wrapper_prompt=query_wrapper_prompt,
        tokenizer_name="llmware/bling-1b-0.1",
        model_name="llmware/bling-1b-0.1",
        device_map="auto",
        tokenizer_kwargs={"max_length": 1024},
        # uncomment this if using CUDA to reduce memory usage
        # model_kwargs={"torch_dtype": torch.float16}
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
    app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY")

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