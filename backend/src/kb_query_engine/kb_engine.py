from llama_index import Document, VectorStoreIndex, QueryBundle
from llama_index.vector_stores import PineconeVectorStore, VectorStoreQuery
from llama_index.schema import TextNode, NodeWithScore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.retrievers import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.service_context import ServiceContext
import pinecone
from google.cloud import storage
import re
import os
from dotenv import load_dotenv
from tqdm import tqdm
import time

class KBIngestRetrieve:
    def __init__(
        self, 
        pinecone_api_key: str, 
        pinecone_env: str, 
        my_index: str, 
        model_name: str = "BAAI/bge-base-en-v1.5", 
        embed_dim: int = 768,
        query_mode: str = "default",
        top_k: int = 5
    ) -> None:
        
        # need api key and environment
        self.api_key = pinecone_api_key
        self.environment = pinecone_env
        
        # initialize index
        self.pinecone_index = pinecone.Index(my_index)
        
        # initialize vector store and embed model from this
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)
        
        # initialize variable for retrieve
        self.query_mode = query_mode
        self.top_k = top_k 
    
    def retrieve(self, my_query_str: str) -> str:
        query_embedding = self.embed_model.get_query_embedding(my_query_str)
        vector_store_query = VectorStoreQuery(
            query_embedding=query_embedding,
            similarity_top_k=self.top_k,
            mode=self.query_mode,
        )
        query_result = self.vector_store.query(vector_store_query)

        ans = ""
        for node in query_result.nodes:
            ans += node.text
            ans += "\n"

        return ans