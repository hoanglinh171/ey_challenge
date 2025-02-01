import yaml
import requests
import os
import json

SAVE_DIR = "data_pipeline/data/raw/"
os.makedirs(SAVE_DIR, exist_ok=True)
LIMIT = 100000  

def query_arcgis(config):
    base_url = config['base_url']
    endpoint = config['endpoint']
    params = config['params']
    name_lst = config['name']

    for name in name_lst:
        if os.path.exists(f"{SAVE_DIR}{name}.json"):
            print(f"Data file already exists")
        else: 
            url = base_url + name + endpoint
            response = requests.get(url, params=params)

            if response.status_code == 200:
                data = response.json()
                with open(f"{SAVE_DIR}{name}.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=4)
                print(f"Data saved as {SAVE_DIR}{name}.json")
            else:
                print(f"Error {response.status_code}: {response.text}")

def fetch_nyc_data(config):
    base_url = config['base_url']
    datasets = config['datasets']

    for i in range(len(datasets)):
        endpoint = datasets[i]['endpoint']
        name = datasets[i]['name']
        url_endpoint = base_url + endpoint
        offset = 0
        all_data = []

        if os.path.exists(f"{SAVE_DIR}{name}.json"):
            print(f"Data file already exists")
        else:
            while True:
                url = f"{url_endpoint}?$limit={LIMIT}&$offset={offset}"
                response = requests.get(url)

                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text}")
                    break

                data = response.json()
                if not data:
                    print("All data fetched!")
                    break

                all_data.extend(data)
                print(f"Retrieved {len(all_data)} rows (Offset: {offset})")
                offset += LIMIT
            
            with open(f"{SAVE_DIR}{name}.json", "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=4)

if __name__ == "__main__":
    with open("data_pipeline/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    arcgis_config = config['arcgis_query_config']
    soda_config = config['soda_api_config']
    query_arcgis(arcgis_config)
    fetch_nyc_data(soda_config)