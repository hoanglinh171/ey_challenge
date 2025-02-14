# Google Earth Engine
import ee
import geemap

import yaml
import requests
import os
import json


SAVE_DIR = "data_pipeline/data/raw/"
os.makedirs(SAVE_DIR, exist_ok=True)
API_LIMIT = 100000
QUERY_LIMIT = 4000

# Geographic Information from ArcGIS API
def query_arcgis(config, max_records_per_query=QUERY_LIMIT):
    base_url = config['base_url']
    endpoint = config['endpoint']
    name_lst = config['name']

    for name in name_lst:
        if os.path.exists(f"{SAVE_DIR}{name}.geojson"):
            print(f"Data file already exists")
        else: 
            url = base_url + name + endpoint
            # First, get all ObjectIDs
            params = {
                'where': '1=1',
                'returnIdsOnly': 'true',
                'f': 'json'
            }
            
            response = requests.get(url, params=params)

            if response.status_code != 200:
                print(f"Error {response.status_code}: {response.text}")
                break

            object_ids = response.json()['objectIds']
            
            if not object_ids:
                raise ValueError("No objects found in the service")
            
            # Sort ObjectIDs to ensure consistent pagination
            object_ids.sort()

            # Initialize GeoJSON structure
            geojson = {
                "type": "FeatureCollection",
                "features": []
            }
            
            # Process in chunks based on max_records_per_query
            for i in range(0, len(object_ids), max_records_per_query):
                chunk = object_ids[i:i + max_records_per_query]
                
                # Create where clause for current chunk
                where_clause = f"OBJECTID >= {chunk[0]} AND OBJECTID <= {chunk[-1]}"
                
                # Query parameters for GeoJSON format
                params = {
                    'where': where_clause,
                    'outFields': '*',  # Get all fields
                    'returnGeometry': 'true',
                    'f': 'geojson'    # Request GeoJSON format directly
                }
                
                # Make request
                response = requests.get(url, params=params)
                data = response.json()
                
                if 'features' in data:
                    geojson['features'].extend(data['features'])
                
                # Print progress
                print(f"Loaded {len(geojson['features'])} features out of {len(object_ids)} total")
            
            # Copy any additional properties from the last response
            for key in data.keys():
                if key != 'features':
                    geojson[key] = data[key]

            with open(f"{SAVE_DIR}{name}.geojson", 'w', encoding="utf-8") as f:
                json.dump(geojson, f, indent=4)

# NYC building and street information from SODA API
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
                url = f"{url_endpoint}?$limit={API_LIMIT}&$offset={offset}"
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
                offset += API_LIMIT
            
            with open(f"{SAVE_DIR}{name}.json", "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=4)

# Google Earth Engine 
def auth_and_init(config):
    proj = config['project']
    ee.Authenticate()
    ee.Initialize(project=proj)
    
def load_ggee_data(config, collections):
    coords = config['coords']
    time_window = config['time_window']
    
    gge_coords = ee.Geometry.Rectangle(coords=coords)
    gge_time_window = ee.DateRange(time_window.split('/')[0], time_window.split('/')[1])

    data_mean_value_dict = {}
    data_std_value_dict = {}
    for key, value in collections.items():
        collection_name = value['name']
        bands = value['bands']

        dataset_mean_value = ee.ImageCollection(collection_name) \
            .filterDate(gge_time_window) \
            .filterBounds(gge_coords) \
            .select(bands) \
            .mean()
        
        dataset_std_value =  ee.ImageCollection(collection_name) \
            .filterDate(gge_time_window) \
            .filterBounds(gge_coords) \
            .select(bands) \
            .std()
        
        data_mean_value_dict[key] = dataset_mean_value
        data_std_value_dict[key] = dataset_std_value
    

if __name__ == "__main__":
    with open("data_pipeline/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    arcgis_config = config['arcgis_query_config']
    soda_config = config['soda_api_config']
    query_arcgis(arcgis_config)
    fetch_nyc_data(soda_config)