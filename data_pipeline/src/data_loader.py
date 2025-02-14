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


# Google Earth Engine authentication
def auth_and_init(config):
    proj = config['project']
    ee.Authenticate()
    ee.Initialize(project=proj)


# Load AOD data from Google Earth Engine API
def load_AOD_data (config):
    coords = ee.Geometry.Rectangle(config['coords'])

    june_01_10 = ee.DateRange("2021-06-01", "2021-06-11")
    june_11_20 = ee.DateRange('2021-06-11', '2021-06-21')
    june_21_30 = ee.DateRange('2021-06-21', '2021-07-01')

    july_01_10 = ee.DateRange("2021-07-01", "2021-07-11")
    july_11_20 = ee.DateRange('2021-07-11', '2021-07-21')
    july_21_30 = ee.DateRange('2021-07-21', '2021-08-01')

    august_01_10 = ee.DateRange("2021-08-01", "2021-08-11")
    august_11_20 = ee.DateRange('2021-08-11', '2021-08-21')
    august_21_30 = ee.DateRange('2021-08-21', '2021-09-01')

    time_window = [
        june_01_10, june_11_20, june_21_30, 
        july_01_10, july_11_20, july_21_30, 
        august_01_10, august_11_20, august_21_30
    ]

    time_window_name = [
        'june_01_10', 'june_11_20', 'june_21_30', 
        'july_01_10', 'july_11_20', 'july_21_30', 
        'august_01_10', 'august_11_20', 'august_21_30'
    ]
    
    aod_collection = config['collections']['aq_collections']['aod_collection']
    
    for i, time in enumerate(time_window):
        for band in aod_collection['bands']:
            dataset = (
                ee.ImageCollection(aod_collection['name']) 
                .filterDate(time) 
                .filterBounds(coords) 
                # .median() \
                .select(band) 
            )
            # print(dataset.size().getInfo())
            data = dataset.sort('DATE_ACQUIRED').toBands()
            print(data.bandNames().size().getInfo())
            
            SAVE_DIR = "../data/tiff/air_quality/AOD/"

            output_file = f"{band}_({time_window_name[i]}).tif"
            geemap.ee_export_image(
                data, 
                filename=SAVE_DIR+output_file, 
                scale=1000, 
                region=coords, 
                file_per_band=False
            )
    

# Load AQ factors data from Google Earth Engine API
def load_AQ_factors_data(config):
    coords = ee.Geometry.Rectangle(config['coords'])
    
    # Air quality factors
    aq_factors = ['CO', 'HCHO', 'NO2', 'O3', 'SO2']

    # Air quality collection defined in config
    aq_collection = config['collections']['aq_collections']
    
    june_01_july_15 = ee.DateRange('2021-06-01', '2021-07-16')
    july_16_august_30 = ee.DateRange('2021-07-16', '2021-09-01')
    
    time_window = [june_01_july_15, july_16_august_30]
    time_window_name = ['june_01_july_15', 'july_16_august_30']

    for i, factor in enumerate(aq_factors):
        factor_collection = list(aq_collection.keys())[i]
        collection_name = aq_collection[factor_collection]['name']
        collection_bands = aq_collection[factor_collection]['bands']
        
        for band in collection_bands:
            
            for i, time in enumerate(time_window):
                dataset = ee.ImageCollection(collection_name) \
                    .filterDate(time) \
                    .filterBounds(coords) \
                    .select(band)
                    
                data = dataset.sort('DATE_ACQUIRED').toBands()
                print(data.bandNames().size().getInfo())
                
                SAVE_DIR = '../data/tiff/air_quality/AQ/'
                output_file = f"{factor}_{band}_({time_window_name[i]}).tif"
                geemap.ee_export_image(
                    data, 
                    filename=SAVE_DIR+output_file, 
                    scale=1000, 
                    region=coords, 
                    file_per_band=False
                )


if __name__ == "__main__":
    with open("data_pipeline/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    arcgis_config = config['arcgis_query_config']
    soda_config = config['soda_api_config']
    gge_engine_config = config['gge_engine_config']
    
    # query_arcgis(arcgis_config)
    # fetch_nyc_data(soda_config)
    auth_and_init(gge_engine_config)
    # load_AOD_data(gge_engine_config)
    load_AQ_factors_data(gge_engine_config)