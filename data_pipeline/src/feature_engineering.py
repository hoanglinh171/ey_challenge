import numpy as np
import ijson
import yaml
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, box
import os
import rasterio
from tqdm import tqdm
from rasterio.mask import mask

READ_DIR = "data_pipeline/data/tiff/1x1/"
SAVE_DIR = "data_pipeline/data/tiff/"
os.makedirs(SAVE_DIR, exist_ok=True)

with open("data_pipeline/config.yaml", "r") as file:
    config = yaml.safe_load(file)
COORDS = config['coords']
CRS = config['gge_engine_config']['crs']


def calculate_indices(tiff, savefile, source, resolution):
    scale = resolution / 111320.0
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    bands = {'red': 1, 'blue': 2, 'green': 3, 'nir': 4, 'swir16': 5, 'swir22': 6}
    if source == "sentinel_2":
        bands['red'] = 4
        bands['blue'] = 2
        bands['green'] = 3
        bands['nir'] = 8
        bands['swir16'] = 11
        bands['swir22'] = 12

    with rasterio.open(tiff) as dst:
        red = dst.read(bands['red'])
        blue = dst.read(bands['blue'])
        green = dst.read(bands['green'])
        nir08 = dst.read(bands['nir'])
        swir16 = dst.read(bands['swir16'])
        swir22 = dst.read(bands['swir22'])

    ndvi = (nir08 - red) / (nir08 + red)
    evi = 2.5 * (nir08 - red) / (nir08 + 6 * red - 7.5 * blue + 1 + 1e-10)
    savi = (nir08 - red) * 1.5 / (nir08 + red + 0.5 + 1e-10)
    gndvi = (nir08 - green) / (nir08 + green + 1e-10)
    arvi = (nir08 - (red - (blue - red))) / (nir08 + (red - (blue - red)) + 1e-10)

    term = (2 * nir08 + 1) ** 2 - 8 * (nir08 - red)
    term = np.maximum(term, 0)  # Ensure no negative values inside sqrt
    msavi = (2 * nir08 + 1 - np.sqrt(term)) / 2

    ndwi = (green - nir08) / (green + nir08)
    mndwi = (green - swir16) / (green + swir16 + 1e-10)
    awei_nsh = 4 * (green - swir16) - (0.25 * nir08 + 2.75 * swir22)
    awei_sh = green + 2.5 * nir08 - 1.5 * (swir16 + swir22) - 0.25 * blue
    ndsi = (green - swir16) / (green + swir16 + 1e-10)
    nbr = (nir08 - swir22) / (nir08 + swir22 + 1e-10)
    si = (swir16 - blue) / (swir16 + blue + 1e-10)
    ndbi = (swir16 - nir08) / (swir16 + nir08)
    ui = (swir16 - red) / (swir16 + red + 1e-10)
    ibi = (ndbi - (savi + mndwi)) / (ndbi + (savi + mndwi) + 1e-10)
    albedo = (0.356 * blue + 0.130 * red + 0.373 * nir08 + 0.085 * swir16 + 0.072 * swir22 - 0.018) / 1.016

    bands = [ndvi, evi, savi, gndvi, arvi, msavi, ndwi, awei_nsh, awei_sh, ndsi, nbr, si, ndbi, ui, ibi, albedo]
    band_names = ['ndvi', 'evi', 'savi', 'gndvi', 'arvi', 'msavi', 'ndwi', 'awei_nsh', 'awei_sh', 'ndsi', 'nbr', 'si', 'ndbi', 'ui', 'ibi', 'albedo']
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=CRS,
                       dtype=ndvi.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{source}_{band_names[i]}')


def building_street_features(building_tiff, street_tiff, savefile, resolution=30):
    with rasterio.open(building_tiff) as dst:
        building_height = dst.read(1)
        building_area = dst.read(4)
        meta = dst.meta

    with rasterio.open(street_tiff) as dst:
        street_width = dst.read(1)

    street_width = np.where(street_width == 0, np.nan, street_width)
    var = building_height / street_width
    var = np.where(np.isin(var, [np.inf, -np.inf, np.nan]), -1, var)

    # Calculate the building's area per pixel area (30 x 30 m^2)
    building_area_per_pixel = building_area / (resolution**2 * 10.764)

    bands = [var, building_area_per_pixel]
    band_names = ['var', 'building_area_per_pixel']
    with rasterio.open(savefile, 'w', **meta) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def smoothing_filter(raster_arr, operation, size=1):  # 3x3: size = 1, 5x5: size = 2
    m, n = raster_arr.shape
    smoothed_raster = np.zeros((m, n))

    for i in tqdm(range(m)):
        for j in range(n):
            neighbors = raster_arr[max(i-size, 0):min(i+1+size, m), max(j-size, 0):min(j+1+size, n)]

            if np.isnan(neighbors).all(): 
                smoothed_raster[i, j] = np.nan
            elif operation == "mean":
                smoothed_raster[i, j] = np.nanmean(neighbors)
            elif operation == "std_dev":
                smoothed_raster[i, j] = np.nanstd(neighbors)
            elif operation == "median":
                smoothed_raster[i, j] = np.nanmedian(neighbors)
            else:
                print("Only except mean, median, and std_dev operation.")

    return smoothed_raster


def smooth_tiff_file(tiff_file, savefile, operation, size=1):
    # read file, get number of layers, name of layer
    raster_arr = []
    with rasterio.open(tiff_file) as dst:
        num_bands = dst.count
        band_names = dst.descriptions
        meta = dst.meta
        for i in range(num_bands):
            band = dst.read(i + 1)
            raster_arr.append(band)

    smoothed_raster_arr = []
    for i, raster in tqdm(enumerate(raster_arr), desc="Iterating bands"):
        smoothed_raster = smoothing_filter(raster, operation, size=size)
        smoothed_raster_arr.append(smoothed_raster)

    # save_tiff
    with rasterio.open(savefile, 'w', **meta) as dst:
        for i, band in enumerate(smoothed_raster_arr):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_{operation}_{size*2 + 1}x{size*2 + 1}')


if __name__ == "__main__":
    # tiff = READ_DIR + "sentinel_2.tiff"
    # savefile = SAVE_DIR + "1x1/sentinel_indices.tiff"
    # calculate_indices(tiff=tiff, source="sentinel_2", savefile=savefile, resolution=10)

    # building_tiff = os.path.join(READ_DIR + "building_res30.tiff")
    # street_tiff = os.path.join(READ_DIR + "street_res30.tiff")
    # savefile = os.path.join(SAVE_DIR, "building_street_res30.tiff")
    # building_street_features(building_tiff, street_tiff, savefile, resolution=30)

    size = 1
    for i, filename in tqdm(enumerate(os.listdir(READ_DIR)), desc="3x3"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(READ_DIR, filename)

            operation = 'mean'
            savefile = os.path.join(SAVE_DIR, f"3x3/{size*2+1}x{size*2+1}_{operation}_{filename}")
            if os.path.exists(savefile):
                print("File exists!")
            else:
                smooth_tiff_file(file_path, savefile, operation, size=size)

            operation = 'std_dev'
            savefile = os.path.join(SAVE_DIR, f"3x3/{size*2+1}x{size*2+1}_{operation}_{filename}")
            if os.path.exists(savefile):
                print("File exists!")
            else:
                smooth_tiff_file(file_path, savefile, operation, size=size)

    
    size = 2
    for i, filename in tqdm(enumerate(os.listdir(READ_DIR)), desc="5x5"):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(READ_DIR, filename)

            operation = 'mean'
            savefile = os.path.join(SAVE_DIR, f"5x5/{size*2+1}x{size*2+1}_{operation}_{filename}")
            if os.path.exists(savefile):
                print("File exists!")
            else:
                smooth_tiff_file(file_path, savefile, operation, size=size)

            operation = 'std_dev'
            savefile = os.path.join(SAVE_DIR, f"5x5/{size*2+1}x{size*2+1}_{operation}_{filename}")
            if os.path.exists(savefile):
                print("File exists!")
            else:
                smooth_tiff_file(file_path, savefile, operation, size=size)
