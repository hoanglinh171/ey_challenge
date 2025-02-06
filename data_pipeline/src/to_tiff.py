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
    else:
        raster_value = mode(all_values)[0]
    
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


if __name__ == "__main__":
    resolution = 30
    readfile = READ_DIR + "building.geojson"
    savefile = SAVE_DIR + f"building_res{resolution}.tiff"
    building_tiff(readfile, savefile, resolution)

    readfile = READ_DIR + "street.geojson"
    savefile = SAVE_DIR + f"street_res{resolution}.tiff"
    street_tiff(readfile, savefile, resolution)