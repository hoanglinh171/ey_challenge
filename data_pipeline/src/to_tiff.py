import os
import yaml
import numpy as np
import geopandas as gpd
import pandas as pd
from rtree import index
from tqdm import tqdm
import rasterio
from pyproj import Transformer
from shapely.geometry import box, Polygon
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder


READ_DIR = "data_pipeline/data/features_extracted/"
SAVE_DIR = "data_pipeline/data/tiff/"
os.makedirs(SAVE_DIR, exist_ok=True)

with open("data_pipeline/config.yaml", "r") as file:
    config = yaml.safe_load(file)
COORDS = config['coords']


## Values to calculate for raster pixels

def sum_area_values(clipped_df, geometry_col, grid_cell):
    # transform to feet
    transformer = Transformer.from_crs("EPSG:4326", "EPSG:2263", always_xy=True)
    clipped_df_feet = clipped_df.to_crs("EPSG:2263")
    grid_cell_feet = Polygon([transformer.transform(x, y) for x, y in grid_cell.exterior.coords])

    total_area = 0
    for _, row in clipped_df_feet.iterrows():
        intersect_area = row[geometry_col].intersection(grid_cell_feet)  # Clip the building to the grid cell
        total_area += intersect_area.area
    
    return total_area
    

def mean_values(clipped_df, value_col):
    total = 0
    count = 0 
    for _, row in clipped_df.iterrows():
        total += row[value_col]
        count += 1

    if count == 0:
        raster_value = 0
    else:
        raster_value = total / count
    
    return raster_value


def std_dev_values(clipped_df, value_col):
    all_values = []
    count = 0 
    for _, row in clipped_df.iterrows():
        all_values.append(row[value_col])
        count += 1

    if count == 0:
        raster_value = -1
    elif count == 1:
        raster_value = 0
    else:
        raster_value = np.std(all_values)
    
    return raster_value


def mode_cat_values(clipped_df, value_col):
    all_values = []
    count = 0 
    for _, row in clipped_df.iterrows():
        all_values.append(row[value_col])
        count += 1

    if count == 0:
        raster_value = -1
    elif count == 1:
        raster_value = all_values[0]
    else:
        raster_value = mode(all_values)[0]
    
    return raster_value


def count_values(clipped_df):
    count = 0 
    for _, row in clipped_df.iterrows():
        count += 1

    if count == 0:
        raster_value = 0
    else:
        raster_value = 1
    
    return raster_value


## Wrapper functions

def calculate_raster_value(operation, clipped_df, geometry_col, value_col, grid_cell):
    if operation == "mean":
        raster_value = mean_values(clipped_df, value_col)
    elif operation == "sum_area":
        raster_value = sum_area_values(clipped_df, geometry_col, grid_cell)
    elif operation == "std_dev":
        raster_value = std_dev_values(clipped_df, value_col)
    elif operation == "mode":
        raster_value = mode_cat_values(clipped_df, value_col)
    elif operation == "count":
        raster_value = count_values(clipped_df)
    else:
        print("Operation must be mean, sum, std_dev, or mode")
        return
    
    return raster_value


def rasterize(df, geometry_col, value_col, operations,
              gt, height, width):
    """
    resolution: meters per pixel
    operations: array of calculation for raster value. Elements can be the following: mean, sum, mode, std_dev
    """

    # Build the Rtree index for faster spatial queries
    idx = index.Index()
    for i, geometry in enumerate(df[geometry_col]):
        idx.insert(i, geometry.bounds)  # Insert bounding boxes into the index

    rasters = []
    for k in range(len(operations)):
        rasters.append(np.zeros((height, width), dtype=np.float32))

    for i in tqdm(range(width)):
        for j in range(height):
            # Get the bounds of the current raster cell (grid square)
            x_min, y_max = gt * (i, j)
            x_max, y_min = gt * (i + 1, j + 1)
            grid_cell = box(x_min, y_min, x_max, y_max)

            # Clip building footprints by the current grid cell
            possible_matches = list(idx.intersection(grid_cell.bounds))  # Find potential matches
            intersect_idx = [idx for idx in possible_matches if df.iloc[idx][geometry_col].intersects(grid_cell)]
            clipped_df = df.iloc[intersect_idx]
            for k in range(len(operations)):
                operation = operations[k]
                rasters[k][j, i] = calculate_raster_value(operation, clipped_df, geometry_col, value_col, grid_cell)
    
    rasters.append("_")

    return tuple(rasters)


###############################################################################################################
## Extract feature for raster and create tiff files

def building_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    mean_height, std_dev_height, _ = rasterize(df, geometry_col, 'heightroof', ['mean', 'std_dev'],
                                            gt, height, width)
    
    mean_year, _ = rasterize(df, geometry_col, 'cnstrct_yr', ['mean'],
                                            gt, height, width)
    
    building_area, _ = rasterize(df, geometry_col, 'geometry', ['sum_area'],
                                            gt, height, width)
    
    # Save raster
    with rasterio.open(savefile, 'w', driver='GTiff', count=4, crs=df.crs,
                       dtype=mean_height.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        dst.write(mean_height, 1)
        dst.write(std_dev_height, 2)
        dst.write(mean_year, 3)
        dst.write(building_area, 4)


def street_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Encode categories
    encoder = LabelEncoder()
    df['TrafDir'] = encoder.fit_transform(df['TrafDir'])
    print(list(encoder.classes_))
    df['direction'] = encoder.fit_transform(df['direction'])
    print(list(encoder.classes_))

    # Fill missing values
    df["street_width_avg"] = df.groupby("RW_TYPE")["street_width_avg"].transform(lambda x: x.fillna(x.median()))
    df["Number_Total_Lanes"] = df.groupby("RW_TYPE")["Number_Total_Lanes"].transform(lambda x: x.fillna(x.median()))

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    mean_width, _ = rasterize(df, geometry_col, 'street_width_avg', ['mean'],
                                            gt, height, width)
    
    traffic_dir, _ = rasterize(df, geometry_col, 'TrafDir', ['mode'],
                                            gt, height, width)
    
    mean_lanes, _ = rasterize(df, geometry_col, 'Number_Total_Lanes', ['mean'],
                                            gt, height, width)
    
    orientation, _ = rasterize(df, geometry_col, 'direction', ['mode'],
                                            gt, height, width)    

    
    # Save raster
    with rasterio.open(savefile, 'w', driver='GTiff', count=4, crs=df.crs,
                       dtype=mean_width.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        dst.write(mean_width, 1)
        dst.write(traffic_dir, 2)
        dst.write(mean_lanes, 3)
        dst.write(orientation, 4)



def zoning_nyco_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Encoding
    # ['C1-2', 'C1-3', 'C1-4', 'C1-5', 'C2-1', 'C2-2', 'C2-3', 'C2-4', 'C2-5']
    # ['C1', 'C2']

    encoder = LabelEncoder()
    df['overlay_level2'] = encoder.fit_transform(df['overlay_level2'])
    print(list(encoder.classes_))
    df['overlay_level1'] = encoder.fit_transform(df['overlay_level1'])
    print(list(encoder.classes_))

    # Create raster
    overlay_level2, _ = rasterize(df, geometry_col, 'overlay_level2', ['mode'],
                                            gt, height, width)
    
    overlay_level1, _ = rasterize(df, geometry_col, 'overlay_level1', ['mode'],
                                            gt, height, width)
    
    # Save raster
    with rasterio.open(savefile, 'w', driver='GTiff', count=2, crs=df.crs,
                       dtype=overlay_level1.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        dst.write(overlay_level1, 1)
        dst.write(overlay_level2, 2)


def zoning_nysp_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Encoding
    # ['125th', 'C', 'CL', 'EC-2', 'EC-3', 'EHC', 'ETC', 'GC', 'HP', 'HRP', 'HRW', 'HY', 'IN', 'J', 'L', 'LIC', 'MMU', 'MP', 'MX-1', 'MX-13', 'MX-14', 'MX-15', 'MX-17', 'MX-18', 'MX-23', 'MX-24', 'MX-7', 'MX-9', 'MiD', 'NA-2', 'PC', 'PI', 'SRI', 'TA', 'U', 'WCh']
    # ['125th', 'C', 'CL', 'EC', 'EHC', 'ETC', 'GC', 'HP', 'HRP', 'HRW', 'HY', 'IN', 'J', 'L', 'LIC', 'MMU', 'MP', 'MX', 'MiD', 'NA', 'PC', 'PI', 'SRI', 'TA', 'U', 'WCh']
    encoder = LabelEncoder()
    df['sd_level2'] = encoder.fit_transform(df['sd_level2'])
    print(list(encoder.classes_))
    df['sd_level1'] = encoder.fit_transform(df['sd_level1'])
    print(list(encoder.classes_))

    # Create raster
    sd_level2, _ = rasterize(df, geometry_col, 'sd_level2', ['mode'],
                                            gt, height, width)
    
    sd_level1, _ = rasterize(df, geometry_col, 'sd_level1', ['mode'],
                                            gt, height, width)
    
    # Save raster
    with rasterio.open(savefile, 'w', driver='GTiff', count=2, crs=df.crs,
                       dtype=sd_level1.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        dst.write(sd_level1, 1)
        dst.write(sd_level2, 2)


def zoning_nyzd_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)
    
    # Encoding
    # ['C1-7', 'C1-7A', 'C1-8', 'C1-8A', 'C1-8X', 'C1-9', 'C2-7', 'C2-7A', 'C2-8', 'C2-8A', 'C3', 'C4-1', 'C4-2', 'C4-2A', 'C4-2F', 'C4-3', 'C4-4', 'C4-4A', 'C4-4D', 'C4-5', 'C4-5D', 'C4-5X', 'C4-6', 'C4-6A', 'C4-7', 'C5-1', 'C5-1A', 'C5-2', 'C5-2.5', 'C5-3', 'C5-P', 'C6-1', 'C6-2', 'C6-2A', 'C6-3', 'C6-3D', 'C6-3X', 'C6-4', 'C6-4.5', 'C6-4M', 'C6-4X', 'C6-5', 'C6-5.5', 'C6-6', 'C6-6.5', 'C6-7', 'C6-7T', 'C8-1', 'C8-2', 'C8-3', 'C8-4', 'M1-1', 'M1-1/R7-2', 'M1-1A/R7-3', 'M1-2', 'M1-2/R5B', 'M1-2/R5D', 'M1-2/R6A', 'M1-2/R7-2', 'M1-3', 'M1-3/R7X', 'M1-3/R8', 'M1-4', 'M1-4/R6A', 'M1-4/R7-3', 'M1-4/R7A', 'M1-4/R7D', 'M1-4/R7X', 'M1-4/R8A', 'M1-4/R9', 'M1-4/R9A', 'M1-5', 'M1-5/R10', 'M1-5/R6A', 'M1-5/R7-2', 'M1-5/R7-3', 'M1-5/R8A', 'M1-5/R9', 'M1-5/R9-1', 'M1-6', 'M1-6/R10', 'M1-6/R9', 'M2-1', 'M2-2', 'M2-3', 'M2-4', 'M3-1', 'M3-2', 'PARK', 'R1-2', 'R10', 'R10A', 'R10H', 'R2', 'R2A', 'R3-1', 'R3-2', 'R3A', 'R3X', 'R4', 'R4-1', 'R4A', 'R4B', 'R5', 'R5A', 'R5B', 'R5D', 'R6', 'R6-1', 'R6A', 'R6B', 'R7-1', 'R7-2', 'R7-3', 'R7A', 'R7B', 'R7D', 'R7X', 'R8', 'R8A', 'R8B', 'R8X', 'R9', 'R9A', 'R9X']
    # ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C8', 'M1', 'M2', 'M3', 'PARK', 'R1', 'R10', 'R10A', 'R10H', 'R2', 'R2A', 'R3', 'R3A', 'R3X', 'R4', 'R4A', 'R4B', 'R5', 'R5A', 'R5B', 'R5D', 'R6', 'R6A', 'R6B', 'R7', 'R7A', 'R7B', 'R7D', 'R7X', 'R8', 'R8A', 'R8B', 'R8X', 'R9', 'R9A', 'R9X']
    # ['C', 'M', 'P', 'R']
    encoder = LabelEncoder()
    df['zonedist_level3'] = encoder.fit_transform(df['zonedist_level3'])
    print(list(encoder.classes_))
    df['zonedist_level2'] = encoder.fit_transform(df['zonedist_level2'])
    print(list(encoder.classes_))
    df['zonedist_level1'] = encoder.fit_transform(df['zonedist_level1'])
    print(list(encoder.classes_))

    # Create raster
    zonedist_level3, _ = rasterize(df, geometry_col, 'zonedist_level3', ['mode'],
                                            gt, height, width)
    
    zonedist_level2, _ = rasterize(df, geometry_col, 'zonedist_level2', ['mode'],
                                            gt, height, width)
    
    zonedist_level1, _ = rasterize(df, geometry_col, 'zonedist_level1', ['mode'],
                                            gt, height, width)
    
    
    # Create raster for existence of R, M, C, P
    residence_df = df[df['zonedist_level1'] == 3]
    has_residence, _ = rasterize(residence_df, geometry_col, 'zonedist_level1', ['count'],
                                 gt, height, width)
    
    manufacture_df = df[df['zonedist_level1'] == 1]
    has_manufacture, _ = rasterize(manufacture_df, geometry_col, 'zonedist_level1', ['count'],
                                 gt, height, width)
    
    commercial_df = df[df['zonedist_level1'] == 0]
    has_commercial, _ = rasterize(commercial_df, geometry_col, 'zonedist_level1', ['count'],
                                 gt, height, width)
    
    park_df = df[df['zonedist_level1'] == 2]
    has_park, _ = rasterize(park_df, geometry_col, 'zonedist_level1', ['count'],
                                 gt, height, width)


    # Save raster
    with rasterio.open(savefile, 'w', driver='GTiff', count=7, crs=df.crs,
                       dtype=zonedist_level1.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        dst.write(zonedist_level1, 1)
        dst.write(zonedist_level2, 2)
        dst.write(zonedist_level3, 3)
        dst.write(has_residence, 4)
        dst.write(has_manufacture, 5)
        dst.write(has_commercial, 6)
        dst.write(has_park, 7)


if __name__ == "__main__":
    resolution = 30
    # readfile = READ_DIR + "building.geojson"
    # savefile = SAVE_DIR + f"building_res{resolution}.tiff"
    # building_tiff(readfile, savefile, resolution)

    # readfile = READ_DIR + "street.geojson"
    # savefile = SAVE_DIR + f"street_res{resolution}.tiff"
    # street_tiff(readfile, savefile, resolution)

    # readfile = READ_DIR + "nyco.geojson"
    # savefile = SAVE_DIR + f"nyco_res{resolution}.tiff"
    # zoning_nyco_tiff(readfile, savefile, resolution)

    # readfile = READ_DIR + "nysp.geojson"
    # savefile = SAVE_DIR + f"nysp_res{resolution}.tiff"
    # zoning_nysp_tiff(readfile, savefile, resolution)

    # readfile = READ_DIR + "nyzd.geojson"
    # savefile = SAVE_DIR + f"nyzd_res{resolution}.tiff"
    # zoning_nyzd_tiff(readfile, savefile, resolution)

    resolution = 100
    readfile = READ_DIR + "nyzd.geojson"
    savefile = SAVE_DIR + f"nyzd_res{resolution}.tiff"
    zoning_nyzd_tiff(readfile, savefile, resolution)

    resolution = 200
    readfile = READ_DIR + "nyzd.geojson"
    savefile = SAVE_DIR + f"nyzd_res{resolution}.tiff"
    zoning_nyzd_tiff(readfile, savefile, resolution)

    # resolution = 500
    # readfile = READ_DIR + "nyzd.geojson"
    # savefile = SAVE_DIR + f"nyzd_res{resolution}.tiff"
    # zoning_nyzd_tiff(readfile, savefile, resolution)

    # resolution = 1000
    # readfile = READ_DIR + "nyzd.geojson"
    # savefile = SAVE_DIR + f"nyzd_res{resolution}.tiff"
    # zoning_nyzd_tiff(readfile, savefile, resolution)