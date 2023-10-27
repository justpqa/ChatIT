import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from google.cloud import storage
import os
from tqdm import tqdm
from datetime import datetime
load_dotenv()

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
            
def scraping_IT():
    folder_name = "IT " + str(datetime.now())
    
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
    for i in tqdm(range(len(cat_lst))):
        cat = Scraping(cat_lst[i], "get_article", source)
        article_lst.extend(cat.scraping())
    
    # scrape articles
    for i in tqdm(range(len(article_lst))):
        temp = Scraping(article_lst[i], "get_text", source)
        content = temp.scraping()
        blob = bucket.blob(f"{folder_name}/article_{i}.txt")
        blob.upload_from_string(content)
    
    return folder_name

if __name__ == "__main__":
    scraping_IT()        