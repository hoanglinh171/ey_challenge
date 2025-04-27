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
from rasterio.enums import Resampling
from scipy.ndimage import uniform_filter
from scipy.stats import entropy


READ_DIR = "data_pipeline/data/features_extracted/"
SAVE_DIR = "data_pipeline/data/tiff/1x1/"
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
        # if row[value_col] is not None:
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
        raster_value = count
    
    return raster_value


def entropy_values(clipped_df, value_col, bin_width=3):
    bins = np.arange(0, 180 + bin_width, bin_width)
    labels = [f"{int(b)}-{int(b+bin_width)}" for b in bins[:-1]]
    value_bin = pd.cut(clipped_df[value_col], bins=bins, labels=labels, include_lowest=True, right=False)

    # 2. Count frequencies per bin
    freq = value_bin.value_counts().sort_index()

    # 3. Convert to probabilities
    probs = freq / freq.sum()

    return entropy(probs, base=np.e)


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
    elif operation == "entropy":
        raster_value = entropy_values(clipped_df, value_col)
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
    
    # building_area, _ = rasterize(df, geometry_col, 'geometry', ['sum_area'],
                                            # gt, height, width)
    
    # Save raster
    bands = [mean_height, std_dev_height, mean_year, 
            #  building_area
             ]
    band_names = ['building_height', 'building_height_std', 'building_year', 
                #   'building_area'
                  ]
    with rasterio.open(savefile, 'w', driver='GTiff', count=4, crs=df.crs,
                       dtype=mean_height.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def street_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Convert type
    df['RW_TYPE'] = df['RW_TYPE'].astype('int')

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
    
    street_type, _ = rasterize(df, geometry_col, 'RW_TYPE', ['mode'],
                               gt, height, width)
    
    df_highway = df[df['RW_TYPE'] == 2]
    high_way, _ = rasterize(df_highway, geometry_col, 'RW_TYPE', ['count'],
                               gt, height, width)
    high_way[high_way >= 1] = 1

    # Save raster
    bands = [mean_width, traffic_dir, mean_lanes, orientation, street_type, high_way]
    band_names = ['street_width', 'street_traffic', 'street_lanes', 'street_orientation', 'street_type', 'high_way']
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=df.crs,
                       dtype=mean_width.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')



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
    bands = [overlay_level2, overlay_level1]
    band_names = ['overlay_level2', 'overlay_level1']
    with rasterio.open(savefile, 'w', driver='GTiff', count=2, crs=df.crs,
                       dtype=overlay_level1.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


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
    bands = [sd_level1, sd_level2]
    band_names = ['sd_level1', 'sd_level2']
    with rasterio.open(savefile, 'w', driver='GTiff', count=2, crs=df.crs,
                       dtype=sd_level1.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')



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
    bands = [zonedist_level1, zonedist_level2, zonedist_level3, has_residence, has_manufacture, has_commercial, has_park]
    band_names = ['zonedist_level1', 'zonedist_level2', 'zonedist_level3', 'has_residence', 'has_manufacture', 'has_commercial', 'has_park']
    with rasterio.open(savefile, 'w', driver='GTiff', count=7, crs=df.crs,
                       dtype=zonedist_level1.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


## Resampling for available tiff file
def resample_canopy_height(readfile, savefile, resolution):
    resolution_degree = resolution / 111035

    with rasterio.open(readfile) as src:
        # Calculate new width and height
        data = src.read(1)
        band_name = src.descriptions

        # Convert to binary (1 for values > 0, 0 otherwise)
        binary_data = (data > 0).astype(np.uint8)

        block_size = int(resolution_degree/ src.res[0])  # Compute scale factor
        print(block_size)
        
        # Use a uniform filter to count nonzero pixels in each block
        count_resampled = uniform_filter(binary_data, size=block_size, mode="constant") * (block_size ** 2)
        count_resampled = count_resampled[::block_size, ::block_size]  # Downsample

        # Define output metadata
        transform = src.transform * src.transform.scale(
            (src.width / count_resampled.shape[1]), (src.height / count_resampled.shape[0])
        )
        
        new_meta = src.meta.copy()
        new_meta.update({
            "transform": transform,
            "width": count_resampled.shape[1],
            "height": count_resampled.shape[0],
            "count": 2
        })

        # Perform resampling
        with rasterio.open(savefile, "w", **new_meta) as dst:
            dst.write(src.read(1, out_shape=(count_resampled.shape[0], count_resampled.shape[1]), resampling=Resampling.average), 1)
            dst.write(count_resampled, 2)
            dst.set_band_description(1, f'{band_name[0]}_res{resolution}')
            dst.set_band_description(2, f'{band_name[0]}_count_res{resolution}')


def osm_edge_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    mean_length, segment_count, _ = rasterize(df, geometry_col, 'length', ['mean', 'count'],
                                              gt, height, width)
    
    total_length = mean_length * segment_count
    
    degree_orientation, entropy_orientation, _ = rasterize(df, geometry_col, 'orientation', ['mode', 'entropy'],
                                            gt, height, width)
    
    mean_circuity, _ = rasterize(df, geometry_col, 'circuity', ['mean'],
                                            gt, height, width)
    
    # Save raster
    bands = [mean_length, segment_count, total_length, degree_orientation, entropy_orientation, mean_circuity]
    band_names = ['segment_avg_length', 'segment_count', 'segment_total_length', 'degree_orientation', 'entropy_orientation', 'mean_circuity']
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=df.crs,
                       dtype=mean_length.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def osm_node_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    connnectivity_index, node_density, entropy_degree, _ = rasterize(df, geometry_col, 'degree', ['mean', 'count', 'entropy'],
                                                                                  gt, height, width)
    
    closeness_100, _ = rasterize(df, geometry_col, 'closeness_100', ['mean'],
                                 gt, height, width)
    
    closeness_500, _ = rasterize(df, geometry_col, 'closeness_500', ['mean'],
                                            gt, height, width)
    
    closeness_1000, _ = rasterize(df, geometry_col, 'closeness_1000', ['mean'],
                                            gt, height, width)
    
    df_intersection = df[df['degree'] >= 3]
    # print(df_intersection.shape)

    intersection_count, _ = rasterize(df_intersection, geometry_col, 'degree', ['count'], 
                                   gt, height, width)
    
    # Save raster
    bands = [connnectivity_index, node_density, entropy_degree, closeness_100, closeness_500, closeness_1000, intersection_count]
    band_names = ['connectivity', 'node_density', 'entropy_degree', 'closeness_100', 'closeness_500', 'closeness_1000', 'intersection']
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=df.crs,
                       dtype=closeness_100.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def elevation_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    mean_elevation, _ = rasterize(df, geometry_col, 'elevation', ['mean'],
                                            gt, height, width)

    mean_elevation[mean_elevation == 0] = np.nan
    m, n = mean_elevation.shape
    for i in range(m):
        for j in range(n):
            if np.isnan(mean_elevation[i, j]):
                mean_elevation[i, j] = np.nanmean([mean_elevation[i, max(j-1, 0)],
                                               mean_elevation[i, min(j+1, n-1)],
                                               mean_elevation[max(i-1, 0), j],
                                               mean_elevation[min(i+1, m-1), j]
                                               ]
                )

            if np.isnan(mean_elevation[i, j]):
                mean_elevation[i, j] = 0


    # Save raster
    bands = [mean_elevation]
    band_names = ['surface_elevation']
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=df.crs,
                       dtype=mean_elevation.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')

    
def hydro_tiff(readfile, savefile, resolution):

    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    water_type, _ = rasterize(df, geometry_col, 'feat_code', ['mode'],
                                            gt, height, width)
    
    water_area, _ = rasterize(df, geometry_col, 'geometry', ['sum_area'],
                                            gt, height, width)
    
    water_area = water_area / ((resolution * 3.28084) ** 2)
    
    # Save raster
    bands = [water_type, water_area]
    band_names = ['water_type', 'water_area']
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=df.crs,
                       dtype=water_area.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def railroad_structure_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    df_vent = df[df['feat_code'] == 2470]
    ventilation_grate , _ = rasterize(df_vent, geometry_col, 'geometry', ['count'],
                                            gt, height, width)

    df_subway = df[df['feat_code'].isin([2160, 2140])]
    subway, _ = rasterize(df_subway, geometry_col, 'geometry', ['count'],
                                            gt, height, width) 
    
    # Save raster
    bands = [ventilation_grate, subway]
    band_names = ['ventilation', 'subway_station']
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=df.crs,
                       dtype=subway.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def cooling_tower_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    cooling_tower, _ = rasterize(df, geometry_col, 'geometry', ['count'],
                                            gt, height, width) 
    
    # Save raster
    bands = [cooling_tower]
    band_names = ['cooling_tower']
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=df.crs,
                       dtype=cooling_tower.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def green_infra_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    green_infra, _ = rasterize(df, geometry_col, 'geometry', ['count'],
                                            gt, height, width) 
    
    # Save raster
    bands = [green_infra]
    band_names = ['green_infra']
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=df.crs,
                       dtype=green_infra.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def tree_point_tiff(readfile, savefile, resolution):
    # Load data
    df = gpd.read_file(readfile)
    geometry_col = 'geometry'

    # Encode categories
    encoder = LabelEncoder()
    df['tpcondition'] = encoder.fit_transform(df['tpcondition'])
    print(list(encoder.classes_))
    df['tpstructure'] = encoder.fit_transform(df['tpstructure'])
    print(list(encoder.classes_))

    # Fill missing values
    df["stumpdiameter"] = df.groupby(["tpstructure", "genusspecies"])["stumpdiameter"].transform(lambda x: x.median())
    df["tpcondition"] = df["tpcondition"].fillna('Unknown')
    df["tpstructure"] = df["tpstructure"].fillna('Unknown')

    # Create raster bounds
    scale = resolution / 111320.0 # degrees per pixel for crs=4326
    width = int(np.round((COORDS[2] - COORDS[0]) / scale) + 1)
    height = int(np.round((COORDS[3] - COORDS[1]) / scale) + 1)
    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)

    # Create raster
    stump = df.dropna(subset="stumpdiameter")
    mean_stump, _ = rasterize(stump, geometry_col, 'stumpdiameter', ['mean'],
                                            gt, height, width)
    
    dbh = df.dropna(subset='dbh')
    mean_dbh, _ = rasterize(dbh, geometry_col, 'dbh', ['mean'],
                                            gt, height, width)
    
    condition, _ = rasterize(df, geometry_col, 'tpcondition', ['mode'],
                                            gt, height, width)
    
    structure, _ = rasterize(df, geometry_col, 'tpstructure', ['mode'],
                                            gt, height, width)    
    
    tree_count, _ = rasterize(df, geometry_col, 'geometry', ['count'],
                               gt, height, width)

    # Save raster
    bands = [mean_stump, mean_dbh, condition, structure, tree_count]
    band_names = ['tree_stump', 'tree_dbh', 'tree_condition', 'tree_structure', 'tree_count']
    with rasterio.open(savefile, 'w', driver='GTiff', count=len(bands), crs=df.crs,
                       dtype=mean_stump.dtype,
                       height=height, width=width,
                       transform=gt) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


if __name__ == "__main__":
    # resolution = 10
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

    # resolution = 500
    # readfile = READ_DIR + "street.geojson"
    # savefile = SAVE_DIR + f"street_res{resolution}.tiff"
    # street_tiff(readfile, savefile, resolution)

    # resolution = 100
    # readfile = READ_DIR + "street.geojson"
    # savefile = SAVE_DIR + f"street_res{resolution}.tiff"
    # street_tiff(readfile, savefile, resolution)

    # resolution = 30
    # readfile = READ_DIR + "street.geojson"
    # savefile = SAVE_DIR + f"street_res{resolution}.tiff"
    # street_tiff(readfile, savefile, resolution)

    # readfile = READ_DIR + "nyzd.geojson"
    # savefile = SAVE_DIR + f"nyzd_res{resolution}.tiff"
    # zoning_nyzd_tiff(readfile, savefile, resolution)

    # resolution = 200
    # readfile = READ_DIR + "nyzd.geojson"
    # savefile = SAVE_DIR + f"nyzd_res{resolution}.tiff"
    # zoning_nyzd_tiff(readfile, savefile, resolution)

    # resolution = 500
    # readfile = READ_DIR + "nyzd.geojson"
    # savefile = SAVE_DIR + f"nyzd_res{resolution}.tiff"
    # zoning_nyzd_tiff(readfile, savefile, resolution)

    # resolution = 1000
    # readfile = READ_DIR + "nyzd.geojson"
    # savefile = SAVE_DIR + f"nyzd_res{resolution}.tiff"
    # zoning_nyzd_tiff(readfile, savefile, resolution)

    # resolution = 30
    # readfile = SAVE_DIR + f"1x1/canopy_height_res1.tif"
    # savefile = SAVE_DIR + f"1x1/canopy_height_res{resolution}.tif"
    # resample_canopy_height(readfile, savefile, resolution)

    # resolution = 10
    # readfile = SAVE_DIR + f"1x1/canopy_height_res1.tif"
    # savefile = SAVE_DIR + f"1x1/canopy_height_res{resolution}.tif"
    # resample_canopy_height(readfile, savefile, resolution)

    # resolution = 5
    # readfile = SAVE_DIR + f"1x1/canopy_height_res1.tif"
    # savefile = SAVE_DIR + f"1x1/canopy_height_res{resolution}.tif"
    # resample_canopy_height(readfile, savefile, resolution)

    # resolution = 1000
    # readfile = READ_DIR + "edges.geojson"
    # savefile = SAVE_DIR + f"edges_res{resolution}.tiff"
    # osm_edge_tiff(readfile, savefile, resolution)

    # readfile = READ_DIR + "nodes.geojson"
    # savefile = SAVE_DIR + f"nodes_res{resolution}.tiff"
    # osm_node_tiff(readfile, savefile, resolution)

    # resolution = 500
    # readfile = READ_DIR + "edges.geojson"
    # savefile = SAVE_DIR + f"edges_res{resolution}.tiff"
    # osm_edge_tiff(readfile, savefile, resolution)

    # readfile = READ_DIR + "nodes.geojson"
    # savefile = SAVE_DIR + f"nodes_res{resolution}.tiff"
    # osm_node_tiff(readfile, savefile, resolution)

    # resolution = 100
    # readfile = READ_DIR + "edges.geojson"
    # savefile = SAVE_DIR + f"edges_res{resolution}.tiff"
    # osm_edge_tiff(readfile, savefile, resolution)

    # readfile = READ_DIR + "nodes.geojson"
    # savefile = SAVE_DIR + f"nodes_res{resolution}.tiff"
    # osm_node_tiff(readfile, savefile, resolution)

    # resolution = 30
    # readfile = READ_DIR + "edges.geojson"
    # savefile = SAVE_DIR + f"edges_res{resolution}.tiff"
    # osm_edge_tiff(readfile, savefile, resolution)

    # readfile = READ_DIR + "nodes.geojson"
    # savefile = SAVE_DIR + f"nodes_res{resolution}.tiff"
    # osm_node_tiff(readfile, savefile, resolution)

    READ_DIR = "data_pipeline/data/preprocessed/"

    # resolution = 100
    # readfile = READ_DIR + "surface_elevation.geojson"
    # savefile = SAVE_DIR + f"surface_elevation_res{resolution}.tiff"
    # elevation_tiff(readfile, savefile, resolution)

    # resolution = 30
    # readfile = READ_DIR + "surface_elevation.geojson"
    # savefile = SAVE_DIR + f"surface_elevation_res{resolution}.tiff"
    # elevation_tiff(readfile, savefile, resolution)

    # resolution = 10
    # readfile = READ_DIR + "surface_elevation.geojson"
    # savefile = SAVE_DIR + f"surface_elevation_res{resolution}.tiff"
    # elevation_tiff(readfile, savefile, resolution)

    # resolution = 1000
    # readfile = READ_DIR + "hydrography.geojson"
    # savefile = SAVE_DIR + f"hydrography_res{resolution}.tiff"
    # hydro_tiff(readfile, savefile, resolution)

    # resolution = 500
    # readfile = READ_DIR + "hydrography.geojson"
    # savefile = SAVE_DIR + f"hydrography_res{resolution}.tiff"
    # hydro_tiff(readfile, savefile, resolution)

    # resolution = 100
    # readfile = READ_DIR + "hydrography.geojson"
    # savefile = SAVE_DIR + f"hydrography_res{resolution}.tiff"
    # hydro_tiff(readfile, savefile, resolution)

    resolution = 10
    readfile = READ_DIR + "hydrography.geojson"
    savefile = SAVE_DIR + f"hydrography_res{resolution}.tiff"
    hydro_tiff(readfile, savefile, resolution)

    # resolution = 1000
    # readfile = READ_DIR + "railroad_structure.geojson"
    # savefile = SAVE_DIR + f"railroad_structure_res{resolution}.tiff"
    # railroad_structure_tiff(readfile, savefile, resolution)

    # resolution = 500
    # readfile = READ_DIR + "railroad_structure.geojson"
    # savefile = SAVE_DIR + f"railroad_structure_res{resolution}.tiff"
    # railroad_structure_tiff(readfile, savefile, resolution)

    # resolution = 100
    # readfile = READ_DIR + "railroad_structure.geojson"
    # savefile = SAVE_DIR + f"railroad_structure_res{resolution}.tiff"
    # railroad_structure_tiff(readfile, savefile, resolution)

    # resolution = 30
    # readfile = READ_DIR + "railroad_structure.geojson"
    # savefile = SAVE_DIR + f"railroad_structure_res{resolution}.tiff"
    # railroad_structure_tiff(readfile, savefile, resolution)

    # resolution = 10
    # readfile = READ_DIR + "railroad_structure.geojson"
    # savefile = SAVE_DIR + f"railroad_structure_res{resolution}.tiff"
    # railroad_structure_tiff(readfile, savefile, resolution)

    # resolution = 1000
    # readfile = READ_DIR + "cooling_tower.geojson"
    # savefile = SAVE_DIR + f"cooling_tower_res{resolution}.tiff"
    # cooling_tower_tiff(readfile, savefile, resolution)

    # resolution = 500
    # readfile = READ_DIR + "cooling_tower.geojson"
    # savefile = SAVE_DIR + f"cooling_tower_res{resolution}.tiff"
    # cooling_tower_tiff(readfile, savefile, resolution)

    # resolution = 100
    # readfile = READ_DIR + "cooling_tower.geojson"
    # savefile = SAVE_DIR + f"cooling_tower_res{resolution}.tiff"
    # cooling_tower_tiff(readfile, savefile, resolution)

    # resolution = 30
    # readfile = READ_DIR + "cooling_tower.geojson"
    # savefile = SAVE_DIR + f"cooling_tower_res{resolution}.tiff"
    # cooling_tower_tiff(readfile, savefile, resolution)

    # resolution = 10
    # readfile = READ_DIR + "cooling_tower.geojson"
    # savefile = SAVE_DIR + f"cooling_tower_res{resolution}.tiff"
    # cooling_tower_tiff(readfile, savefile, resolution)

    # resolution = 1000
    # readfile = READ_DIR + "green_infrastructure.geojson"
    # savefile = SAVE_DIR + f"green_infrastructure_res{resolution}.tiff"
    # green_infra_tiff(readfile, savefile, resolution)

    # resolution = 500
    # readfile = READ_DIR + "green_infrastructure.geojson"
    # savefile = SAVE_DIR + f"green_infrastructure_res{resolution}.tiff"
    # green_infra_tiff(readfile, savefile, resolution)

    # resolution = 100
    # readfile = READ_DIR + "green_infrastructure.geojson"
    # savefile = SAVE_DIR + f"green_infrastructure_res{resolution}.tiff"
    # green_infra_tiff(readfile, savefile, resolution)

    # resolution = 30
    # readfile = READ_DIR + "green_infrastructure.geojson"
    # savefile = SAVE_DIR + f"green_infrastructure_res{resolution}.tiff"
    # green_infra_tiff(readfile, savefile, resolution)

    # resolution = 10
    # readfile = READ_DIR + "green_infrastructure.geojson"
    # savefile = SAVE_DIR + f"green_infrastructure_res{resolution}.tiff"
    # green_infra_tiff(readfile, savefile, resolution)

    # resolution = 1000
    # readfile = READ_DIR + "tree_points.geojson"
    # savefile = SAVE_DIR + f"tree_points_res{resolution}.tiff"
    # tree_point_tiff(readfile, savefile, resolution)

    # resolution = 500
    # readfile = READ_DIR + "tree_points.geojson"
    # savefile = SAVE_DIR + f"tree_points_res{resolution}.tiff"
    # tree_point_tiff(readfile, savefile, resolution)

    # resolution = 100
    # readfile = READ_DIR + "tree_points.geojson"
    # savefile = SAVE_DIR + f"tree_points_res{resolution}.tiff"
    # tree_point_tiff(readfile, savefile, resolution)

    # resolution = 30
    # readfile = READ_DIR + "tree_points.geojson"
    # savefile = SAVE_DIR + f"tree_points_res{resolution}.tiff"
    # tree_point_tiff(readfile, savefile, resolution)

    # resolution = 10
    # readfile = READ_DIR + "tree_points.geojson"
    # savefile = SAVE_DIR + f"tree_points_res{resolution}.tiff"
    # tree_point_tiff(readfile, savefile, resolution)