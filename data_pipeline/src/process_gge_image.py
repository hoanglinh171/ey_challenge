# Google Earth Engine
import ee
import geemap

import yaml
import os
from tqdm import tqdm


SAVE_DIR_AOD = "data_pipeline/data/tiff/air_quality/AOD/"
SAVE_DIR_AQ = "data_pipeline/data/tiff/air_quality/AQ/"
os.makedirs(SAVE_DIR_AOD, exist_ok=True)
os.makedirs(SAVE_DIR_AQ, exist_ok=True)


# Google Earth Engine authentication
def auth_and_init(config):
    proj = config['project']
    ee.Authenticate(force=True)
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
    
    for i, time in tqdm(enumerate(time_window), desc="Downloading aod"):
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

            output_file = f"{band}_({time_window_name[i]}).tif"
            geemap.ee_export_image(
                data, 
                filename=SAVE_DIR_AOD+output_file, 
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

    for i, factor in tqdm(enumerate(aq_factors), desc="Downloading aq"):
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

                output_file = f"{factor}_{band}_({time_window_name[i]}).tif"
                geemap.ee_export_image(
                    data, 
                    filename=SAVE_DIR_AQ+output_file, 
                    scale=1000, 
                    region=coords, 
                    file_per_band=False
                )


if __name__ == "__main__":
    with open("data_pipeline/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    gge_engine_config = config['gge_engine_config']
    
    auth_and_init(gge_engine_config)
    load_AOD_data(gge_engine_config)
    load_AQ_factors_data(gge_engine_config)