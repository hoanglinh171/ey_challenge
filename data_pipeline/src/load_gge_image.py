# Google Earth Engine
import ee
import geemap

import rasterio

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


def scale_aod(read_folder, save_folder):
    for i, filename in tqdm(enumerate(os.listdir(read_folder))):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            
            with rasterio.open(file_path) as dst:
                aod_47 = dst.read(1)
                aod_55 = dst.read(2)
                aod_uncertainty = dst.read(3)
                aod_qa = dst.read(4)
                transform = dst.transform
                crs = dst.crs
                
            # convert binary
            aod_qa_bits = np.bitwise_and(np.right_shift(aod_qa, 8), 0b1111)

            # masking array
            mask = (aod_47 == -28672)

            # convert type
            aod_47 = aod_47.astype('float')
            aod_55 = aod_55.astype('float')
            aod_uncertainty = aod_uncertainty.astype('float')
            aod_qa_bits = aod_qa_bits.astype('float')

            # replace None
            aod_47[mask] = np.nan
            aod_55[mask] = np.nan
            aod_uncertainty[mask] = np.nan
            aod_qa_bits[mask] = np.nan

            # scale
            aod_47 = aod_47 * 0.001
            aod_55 = aod_55 * 0.001
            aod_uncertainty = aod_uncertainty * 0.0001

            # save data
            savefile = os.path.join(save_folder, filename)
            with rasterio.open(savefile, 'w', driver='GTiff', count=4, crs=crs, transform=transform,
                    dtype=aod_47.dtype,
                    height=aod_47.shape[0], width=aod_47.shape[1]) as dst:
                dst.write(aod_47, 1)
                dst.write(aod_55, 2)
                dst.write(aod_uncertainty, 3)
                dst.write(aod_qa_bits, 4)



def nanmean_filter(matrix):
    nan_mask = np.isnan(matrix)
    filled_matrix = np.copy(matrix)

    # Compute mean using convolution (excluding NaNs)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if nan_mask[i, j]:  # If NaN, replace with local mean
                neighbors = []
                
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # Up, Down, Left, Right
                    ni, nj = i + di, j + dj
                    if 0 <= ni < matrix.shape[0] and 0 <= nj < matrix.shape[1]:
                        if not np.isnan(matrix[ni, nj]):
                            neighbors.append(matrix[ni, nj])

                if neighbors:
                    filled_matrix[i, j] = np.nanmean(neighbors)  # Replace with neighbor mean

    return filled_matrix


def calculate_median_aod(read_folder, savefile):
    aod47_lst = []
    aod55_lst = []

    for i, filename in tqdm(enumerate(os.listdir(read_folder))):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            
            with rasterio.open(file_path) as dst:
                aod_47 = dst.read(1)
                aod_55 = dst.read(2)
                transform = dst.transform
                crs = dst.crs
            aod47_lst.append(aod_47)
            aod55_lst.append(aod_55)

    aod47_median = np.nanmedian(np.array(aod47_lst), axis=0)
    aod55_median = np.nanmedian(np.array(aod55_lst), axis=0)

    filled_aod55 = nanmean_filter(aod55_median)
    filled_aod47 = nanmean_filter(aod47_median) 

    with rasterio.open(savefile, 'w', driver='GTiff', count=2, crs=crs, transform=transform,
            dtype=filled_aod47.dtype,
            height=filled_aod47.shape[0], width=filled_aod47.shape[1]) as dst:
        dst.write(filled_aod47, 1)
        dst.write(filled_aod55, 2)
        dst.set_band_description(1, 'aod47')
        dst.set_band_description(2, 'aod55')


def set_band_name_aq(tiff_file, name):
    with rasterio.open(tiff_file) as dst:
        band = dst.read(1)
        meta = dst.meta
    
    with rasterio.open(tiff_file, "w", **meta) as src:
        src.write(band, 1)
        src.set_band_description(1, name)


if __name__ == "__main__":
    # with open("data_pipeline/config.yaml", "r") as file:
    #     config = yaml.safe_load(file)
    
    # gge_engine_config = config['gge_engine_config']
    
    # auth_and_init(gge_engine_config)
    # load_AOD_data(gge_engine_config)
    # load_AQ_factors_data(gge_engine_config)

    # read_folder = "data_pipeline/data/raw/air_quality/AOD/"
    # save_folder = "data_pipeline/data/tiff/air_quality/AOD/"
    # scale_aod(read_folder, save_folder)

    # read_folder = "data_pipeline/data/tiff/air_quality/AOD/"
    # savefile = "data_pipeline/data/tiff/aod_median.tiff"
    # calculate_median_aod(read_folder, savefile)

    tiff_file = "data_pipeline/data/tiff/co_20210724_165223.tif"
    set_band_name_aq(tiff_file, "co_july24")

    tiff_file = "data_pipeline/data/tiff/no2_20210724_165223.tif"
    set_band_name_aq(tiff_file, "no2_july24")

    tiff_file = "data_pipeline/data/tiff/hcho_20210724_165223.tif"
    set_band_name_aq(tiff_file, "hcho_july24")

    tiff_file = "data_pipeline/data/tiff/o3_20210724_165223.tif"
    set_band_name_aq(tiff_file, "o3_july24")