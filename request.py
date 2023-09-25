 #Create record in Generate Content Table
import requests
import uuid

def test():

    API_URL = "https://hook.eu1.make.com/927ubkqsww2puh5uwhg6ong2fppw0t59"
    unique_id = str(uuid.uuid4())

    data = {
            "batch_id": unique_id
            }

    requests.get(API_URL, json=data)

test()