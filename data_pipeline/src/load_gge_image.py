# Google Earth Engine
import ee
import geemap

import yaml
import os
from tqdm import tqdm

import numpy as np


SAVE_DIR_AOD = "data_pipeline/data/raw/air_quality/AOD/"
SAVE_DIR_AQ = "data_pipeline/data/raw/air_quality/AQ/"
os.makedirs(SAVE_DIR_AOD, exist_ok=True)
os.makedirs(SAVE_DIR_AQ, exist_ok=True)


# Google Earth Engine authentication
def auth_and_init(config):
    proj = config['project']
    ee.Authenticate()
    ee.Initialize(project=proj)


# Load AOD data from Google Earth Engine API
def load_AOD_data (config):
    # Fill values definition
    fill_values = {
        'Optical_Depth_047': -28672,
        'Optical_Depth_055': -28672,
        'AOD_Uncertainty': -28762,
        'AOD_QA': 0
    }
    
    # Coords and time window
    coords = config['coords']
    time_window = config['time_window']
    
    # crs
    crs = config['crs']
    
    # Tranform into gge engine type
    gge_engine_coords = ee.Geometry.Rectangle(coords=coords, proj=crs)
    gge_engine_time_window = ee.DateRange(time_window.split('/')[0], time_window.split('/')[1])
    
    # AOD collection
    aod_collection = config['collections']['aq_collections']['aod_collection']
    collection_name = aod_collection['name']
    collection_bands = aod_collection['bands']
    
    # Data retrieval
    dataset = ee.ImageCollection(collection_name) \
        .filterBounds(gge_engine_coords) \
        .filterDate(gge_engine_time_window) \
        .select(collection_bands) \
    
    # Number of images
    print(dataset.size().getInfo())
    
    # Mask NoData pixels 
    def mask_fill_values(img):
        for band in collection_bands:
            # print(f'Band name: {band}')
            img = img.updateMask(img.select(band).neq(fill_values[band]))
        return img
    
    masked_dataset = dataset.map(mask_fill_values)
    
    # Check if the image has valid (non-noData) pixels
    def has_valid_pixels(img):
        valid_pixel_count = img.select(collection_bands) \
        .reduceRegion(
            reducer=ee.Reducer.count(),
            geometry=gge_engine_coords,
            scale=1000,
            bestEffort=True
        ).values().get(0)
        return img.set("valid_pixel_count", valid_pixel_count)

    dataset_with_counts = masked_dataset.map(has_valid_pixels)

    # Filter out images with no valid pixels (valid_pixel_count == 0)
    valid_dataset = dataset_with_counts.filter(ee.Filter.gt("valid_pixel_count", 0))
    
    # Valid Images
    print(valid_dataset.size().getInfo())
    
    image_list = valid_dataset.toList(dataset.size())
    num_images = image_list.size().getInfo()

    
    for i in tqdm(range(num_images), desc='Export Images'):
        image = ee.Image(image_list.get(i))
        
        # Define the target CRS (EPSG:4326) and resolution
        target_crs = crs
        scale = 1000 
        
        # to EPSG:4326
        image_reprojected = image.reproject(crs=target_crs, scale=scale)
        
        # Timestamp
        timestamp = ee.Date(image.get("system:time_start")).format("YYYYMMdd_HHmmss").getInfo()
        # File name 
        filename = f"AOD_{timestamp}.tif"
    
        # Export Image
        geemap.ee_export_image(
            ee_object=image_reprojected,
            filename=os.path.join(SAVE_DIR_AOD, filename),
            region=gge_engine_coords,
            file_per_band=False,
        )
    

# Load AQ factors data from Google Earth Engine API
def load_AQ_factors_data(config):
    # Coords and time window
    coords = config['coords']
    time_window = config['time_window']
    
    # crs
    crs = config['crs']
    
    # Tranform into gge engine type
    gge_engine_coords = ee.Geometry.Rectangle(coords=coords)
    gge_engine_time_window = ee.DateRange(time_window.split('/')[0], time_window.split('/')[1])
    
    aq_factors = ['co', 'hcho', 'no2', 'o3', 'so2']
    aq_collection = config['collections']['aq_collections']
    
    for i, factor in tqdm(enumerate(aq_factors)):
        
        # Each factor collection
        factor_collection = aq_collection[f'{factor}_collection']
        collection_name = factor_collection['name']
        bands = factor_collection['bands']
        
        # Data retrieval
        dataset = ee.ImageCollection(collection_name) \
            .filterBounds(gge_engine_coords) \
            .filterDate(gge_engine_time_window) \
            .select(bands)
        
        # Number of images
        print(f'\n{dataset.size().getInfo()}')
        
        # Check if the image has valid (non-noData) pixels
        def has_valid_pixels(img):
            valid_pixel_count = img.select(bands) \
            .reduceRegion(
                reducer=ee.Reducer.count(),
                geometry=gge_engine_coords,
                scale=1000,
                bestEffort=True
            ).values().get(0)
            return img.set("valid_pixel_count", valid_pixel_count)

        dataset_with_counts = dataset.map(has_valid_pixels)

        # Filter out images with no valid pixels (valid_pixel_count == 0)
        valid_dataset = dataset_with_counts.filter(ee.Filter.gt("valid_pixel_count", 0))
        
        # Valid Images
        print(f'\n{valid_dataset.size().getInfo()}')
        
        image_list = valid_dataset.toList(dataset.size())
        num_images = image_list.size().getInfo()
        
        for i in tqdm(range(num_images), desc=f'Export {factor} images'):
            image = ee.Image(image_list.get(i))
        
            # Define the target CRS (EPSG:4326) and resolution
            target_crs = crs
            scale = 1000 
            
            # to EPSG:4326
            image_reprojected = image.reproject(crs=target_crs, scale=scale)
            
            # Timestamp
            timestamp = ee.Date(image.get("system:time_start")).format("YYYYMMdd_HHmmss").getInfo()
            # File name 
            filename = f"{factor}_{timestamp}.tif"
        
            # Export Image
            geemap.ee_export_image(
                ee_object=image_reprojected,
                filename=os.path.join(SAVE_DIR_AQ, filename),
                region=gge_engine_coords,
                file_per_band=False,
            )

if __name__ == "__main__":
    with open("data_pipeline/config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    gge_engine_config = config['gge_engine_config']
    
    auth_and_init(gge_engine_config)
    load_AOD_data(gge_engine_config)
    load_AQ_factors_data(gge_engine_config)