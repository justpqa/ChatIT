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
        # need env variable
        pinecone.init(api_key=pinecone_api_key, environment=pinecone_env)
        self.api_key = pinecone_api_key
        self.environment = pinecone_env
        
        # create index if not in and initialize index
        if my_index not in pinecone.list_indexes():
            pinecone.create_index(my_index, dimension = embed_dim, metric="euclidean", pod_type="p1")
        self.pinecone_index = pinecone.Index(my_index)
        # variable to check if the vector store is empty
        self.has_vector = self.pinecone_index.describe_index_stats().total_vector_count > 0
        
        # initialize vector store and embed model from this
        self.vector_store = PineconeVectorStore(pinecone_index=self.pinecone_index)
        self.embed_model = HuggingFaceEmbedding(model_name=model_name)
        
        # initialize variable for retrieve
        self.query_mode = query_mode
        self.top_k = top_k 
    
    def ingest_from_gcp_bucket(self, bucket, folder, sen_splitter) -> None:
        # ingestion from a bucket, need a function for a directory later
        nodes = []
        for inx, b in enumerate(tqdm(list(bucket.list_blobs(prefix = folder)))):
            # extract the text and process it
            t = bucket.blob(b.name).download_as_text()
            #t = re.sub(r'[^a-zA-Z0-9 \\.]', ' ', t)
            #t = re.sub(r'\s+', ' ', t)
            
            # split the text
            curr_text_chunks = sen_splitter.split_text(t)
            
            # add these chunks to nodes with embedding
            #for chunk in curr_text_chunks:
            for chunk in curr_text_chunks:
                node = TextNode(text=chunk)
                src_doc_inx = inx
                src_doc = t
                node.embedding = self.embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
                nodes.append(node)
            
        # add the list of nodes to vector store
        self.vector_store.add(nodes)   
        self.has_vector = True
    
    def retrieve(self, my_query_str: str) -> str:
        # check if index is empty
        if self.has_vector:
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
        else:    
            print("The index is empty right now")
            return ""
    

if __name__ == "__main__":
    # code for testing the ingestion and retrieving process
    load_dotenv()
    # initialize env variable 
    api_key = os.environ["PINECONE_API_KEY"]
    environment = os.environ["PINECONE_ENV"]
    my_index = os.environ["PINECONE_INDEX"]
    
    # need to initialize the bucket
    storage_client = storage.Client()
    bucket = storage_client.bucket(os.environ["bucket"])
    
    # initialize the class
    kbir = KBIngestRetrieve(api_key, environment, my_index)
    
    # add from bucket
    kbir.ingest_from_gcp_bucket(bucket)
    
    # need to timeout to make sure everything is loaded
    time.sleep(30)
    
    temp = kbir.retrieve("How to connect to eduroam?")
    print(temp)