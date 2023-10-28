import requests
import pytest
from dotenv import load_dotenv 
load_dotenv()

def test_query():
    question = "How to connect to eduroam"
    question = "%20".join(question.split(" "))
    url = os.environ["query_url"] + question
    response = requests.get(url)
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert len(data["answer"]) > 0