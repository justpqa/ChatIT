import os
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from google.cloud import storage
from llama_index.vector_stores import PineconeVectorStore, VectorStoreQuery
from llama_index.schema import TextNode
from llama_index.embeddings import HuggingFaceEmbedding
from llama_index.text_splitter import SentenceSplitter
import pinecone
from google.cloud import storage
import re
import requests
from bs4 import BeautifulSoup
from datetime import datetime
import time
from dotenv import load_dotenv
load_dotenv("/opt/airflow/.env")

api_key = os.environ.get("PINECONE_API_KEY")
environment = os.environ.get("PINECONE_ENV")
my_index = os.environ.get("PINECONE_INDEX")
pinecone.init(api_key=api_key, environment=environment)
storage_client = storage.Client()
bucket = storage_client.bucket(os.environ.get("bucket"))
folder_name = "IT " + str(datetime.now())
embed_model = HuggingFaceEmbedding(model_name=os.environ.get("DEFAULT_EMBED_MODEL"))
embed_dim = int(os.environ.get("DEFAULT_EMBED_DIM"))
storage_client = storage.Client()
bucket = storage_client.bucket(os.environ.get("bucket"))
sen_splitter = SentenceSplitter(separator = ".", paragraph_separator = "/n/n/n")
    
class Scraping:
    def __init__(self, url, mode, source):
        self.url = url
        self.mode = mode # mode: [get_cat, get_article, get_text]
        self.source = source
        req = requests.get(self.url)
        self.soup = BeautifulSoup(req.text, "html.parser")
    
    def scraping(self):
        if self.mode == "get_cat":
            # getting all categories with different links in each of them
            category = []
            for link in self.soup.find_all('a'):
                temp = link.get('href')
                if "CategoryID" in temp:
                    category.append(self.source + temp)
            return category
        elif self.mode == "get_article":
            row = self.soup.findAll('div', attrs={"class":'row'})[0]
            data = row.findAll('div', attrs={"class":'gutter-bottom'})
            article = []
            for div in data:
                link = div.findAll('a')
                for a in link:
                    article.append(self.source + a["href"])
            return article
        else:
            main_content = self.soup.findAll('div', attrs={"id": "divMainContent"})[0]
            main_content = main_content.findAll('div', attrs={"id": "ctl00_ctl00_cpContent_cpContent_divBody"})[0]
            res = main_content.text
            return res

def IT_website_to_GCS(folder_name):
    
    mainURL = os.environ.get("mainurl")
    source = os.environ.get("headurl")
    bucket_name = os.environ.get("bucket")
    
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # scraping main web for smaller topics
    main = Scraping(mainURL, "get_cat", source)
    cat_lst = main.scraping()
    
    # get all article in each topic
    article_lst = []
    for i in range(len(cat_lst)):
        cat = Scraping(cat_lst[i], "get_article", source)
        article_lst.extend(cat.scraping())
    
    # scrape articles
    for i in range(len(article_lst)):
        temp = Scraping(article_lst[i], "get_text", source)
        content = temp.scraping()
        blob = bucket.blob(f"{folder_name}/article_{i}.txt")
        blob.upload_from_string(content)

def create_Pinecone_Index(my_index, embed_dim):
    if my_index in pinecone.list_indexes():
        pinecone.delete_index(my_index)
        time.sleep(30)
    pinecone.create_index(my_index, dimension = embed_dim, metric="euclidean", pod_type="p1")
    time.sleep(15)
        
def GCS_to_Pinecone(my_index, embed_model, bucket, folder, sen_splitter):
    pinecone_index = pinecone.Index(my_index)
    vector_store = PineconeVectorStore(pinecone_index=pinecone_index)
    
    nodes = []
    for inx, b in enumerate(list(bucket.list_blobs(prefix = folder))):
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
            node.embedding = embed_model.get_text_embedding(node.get_content(metadata_mode="all"))
            nodes.append(node)
        
    # add the list of nodes to vector store
    vector_store.add(nodes) 
    
# define the default arguments for dags
default_args = {
    'owner': 'airflow',
    'depends_on_past': True,    
    'start_date': datetime(2023, 6, 19),
    'end_date': datetime(2100, 12, 31),
    'email': ['airflow@airflow.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2
}
# define the dags
with DAG(
    dag_id="IT_to_Pinecone_dag",
    schedule_interval="0 2 * * *",
    default_args=default_args,
    catchup=False,
    max_active_runs=1
) as dag:
    
    IT_website_to_GCS_task = PythonOperator(
        task_id = "IT_website_to_GCS",
        python_callable = IT_website_to_GCS,
        op_kwargs = {
            "folder_name": folder_name
        },
        dag = dag
    )

    create_Pinecone_Index_task = PythonOperator(
        task_id = "create_Pinecone_Index",
        python_callable = create_Pinecone_Index,
        op_kwargs = {
            "my_index": my_index,
            "embed_dim": embed_dim
        },
        dag = dag
    )
    
    GCS_to_Pinecone_task = PythonOperator(
        task_id = "GCS_to_Pinecone",
        python_callable = GCS_to_Pinecone,
        op_kwargs = {
            "my_index": my_index,
            "embed_model": embed_model,
            "bucket": bucket,
            "folder": folder_name,
            "sen_splitter": sen_splitter
        },
        dag = dag
    )
    
IT_website_to_GCS_task >> create_Pinecone_Index_task >> GCS_to_Pinecone_task