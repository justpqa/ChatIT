# ChatITAPI
A simple API that allow St. Olaf student and faculty to ask IT related question, UI interface will be updated soon. This repository is currently consist of a data pipeline for scraping from all subpages in the St. Olaf IT website and ingested them into an Google Cloud Storage bucket, which will be ingested later into a Pinecone Vector Index, and a simple Flask API that utlized Llama-index for querying the result from the vector database to answer various questions from users.

## 🔧 Built With

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)
![Apache Airflow](https://img.shields.io/badge/Apache%20Airflow-017CEE?style=for-the-badge&logo=Apache%20Airflow&logoColor=white)
![Google Cloud](https://img.shields.io/badge/GoogleCloud-%234285F4.svg?style=for-the-badge&logo=google-cloud&logoColor=white)
![Docker](https://img.shields.io/badge/docker-%230db7ed.svg?style=for-the-badge&logo=docker&logoColor=white)

## :triangular_flag_on_post: Roadmap

- [x] Create GCP account + configure VM instance
- [x] Set up Python, Docker, Airflow GCP config on VM 
- [x] Create pipeline for scraping the data to GCS Bucket
- [x] Use Jupyter notebook and Llama-Index to examine text parsing and how to design the RAG 
- [x] Create pipeline for ingesting the data into a Vector Store using Pinecone API and Llama-index and create a retrieval pipeline
