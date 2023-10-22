import requests 
    
url = "http://localhost:5000/ingest/phan4"
response = requests.get(url)
if response.status_code == 200:
    # Request was successful
    data = response.json()  # Assuming the response contains JSON data
    print(data)
else:
    # Request failed
    print(f"Request failed with status code {response.status_code}")   

# testing the code for calling the api
url = "http://localhost:5000/retrieve/eduroam"
response = requests.get(url)
if response.status_code == 200:
    # Request was successful
    data = response.json()  # Assuming the response contains JSON data
    print(data)
else:
    # Request failed
    print(f"Request failed with status code {response.status_code}")
