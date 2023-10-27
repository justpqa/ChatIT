from flask import Flask, jsonify
from llama_index import Document, VectorStoreIndex, QueryBundle
from llama_index.vector_stores import PineconeVectorStore, VectorStoreQuery
from llama_index.schema import TextNode, NodeWithScore
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.retrievers import BaseRetriever
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.indices.service_context import ServiceContext
from llama_index.text_splitter import SentenceSplitter
import pinecone
from google.cloud import storage
import re
import os
from dotenv import load_dotenv
from kb_query_engine.kb_engine import KBIngestRetrieve 
from kb_query_engine.scraping_kb import scraping_IT
import time
load_dotenv()

# initialize the app
app = Flask(__name__)

# initialize env variable
admin = os.environ["admin"]
api_key = os.environ["PINECONE_API_KEY"]
environment = os.environ["PINECONE_ENV"]
my_index = os.environ["PINECONE_INDEX"]
embed_dim = int(os.environ["DEFAULT_EMBED_DIM"])
storage_client = storage.Client()
bucket = storage_client.bucket(os.environ["bucket"])
sen_splitter = SentenceSplitter(separator = ".", paragraph_separator = "/n/n/n")

# initialize pinecone env
pinecone.init(api_key=api_key, environment=environment)

# initialize the kb class
kbir = KBIngestRetrieve(api_key, environment, my_index)

# define the query method
@app.route('/retrieve/<query_str>', methods=['GET'])
def retrieve(query_str):
    if kbir.has_vector:
        ans = kbir.retrieve(query_str)
        return jsonify({"answer": ans})
    else:
        return jsonify({"Error": "The vector store index is currently empty."})

# define the ingest method in case there is upload to knowledge base
@app.route('/ingest/<token>', methods=["GET", "POST"])
def ingest(token):
    # need to check if it is right token
    if token == admin:
        # reingest and add to a new folder
        ingest_folder = scraping_IT()
        
        # delete current index
        if my_index in pinecone.list_indexes():
            pinecone.delete_index(my_index)
            time.sleep(15)
        
        # reinitialize index and create new vector store for ingest
        pinecone.create_index(my_index, dimension = embed_dim, metric="euclidean", pod_type="p1")
        kbir.pinecone_index = pinecone.Index(my_index)
        kbir.has_vector = False
        kbir.vector_store = PineconeVectorStore(pinecone_index=kbir.pinecone_index)
        kbir.ingest_from_gcs(bucket, ingest_folder, sen_splitter)
        
        return jsonify({"Response": "Success in ingesting the data"})
    else:
        return jsonify({"Response": "Error: You do not have permission to perform this action"})

if __name__ == '__main__':
    app.run(debug=True)