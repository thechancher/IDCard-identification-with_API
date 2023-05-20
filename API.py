import requests


class API:
    
    key: str
    
    def __init__(self):
        self.key = "5ZFzEDSpSh2mS2gKqTfm"
        self.url = "https://us-central1-identificacion-automatizada-v1.cloudfunctions.net/app/data"

    def getData(self, id):
        # identification
        params = {"id": id, "key": self.key}

        # GET petition
        response = requests.get(self.url, params=params)

        # check if ok
        if response.status_code == 200:
            # get response in JSON
            json = response.json()
            return json
        else:
            print("Error code:", response.status_code)
            return None
