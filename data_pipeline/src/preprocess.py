import json
import ijson
import yaml
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, box
import os
import rasterio
from rasterio.mask import mask

READ_DIR = "data_pipeline/data/raw/"
SAVE_DIR = "data_pipeline/data/preprocessed/"
os.makedirs(SAVE_DIR, exist_ok=True)

with open("data_pipeline/config.yaml", "r") as file:
    config = yaml.safe_load(file)
COORDS = config['coords']

def filter_building(readfile, savefile):
    # Read data from json
    features = []
    with open(readfile, "r", encoding="utf-8") as f:
        for feature in ijson.items(f, "item"): 
            features.append(feature)
    print("Load data sucessfully!")

    df = gpd.GeoDataFrame(features)

    # Drop missing value
    df = df.dropna(subset=['the_geom', 'bin', 'cnstrct_yr', 'heightroof'], how='any')

    # Convert data type
    df['lstmoddate'] = pd.to_datetime(df['lstmoddate'])
    df['bin'] = df['bin'].astype('int')
    df['cnstrct_yr'] = df['cnstrct_yr'].astype('int')
    df['heightroof'] = df['heightroof'].astype('float')
    df['feat_code'] = df['feat_code'].astype('int')

    # Filter condition
    built_before_2021 = df['cnstrct_yr'] <= 2021
    in_mahanttan_bronx = (df['bin'] // 10**6).isin([1, 2])
    higher_12_feet = df['heightroof'] >= 12
    is_building = df['feat_code'].isin([1006, 2100])
    constructed_before_date = (df['lstmoddate'] < '2021-07-24') & (df['lststatype'].isin(['Constructed']))
    df = df[built_before_2021 & in_mahanttan_bronx & higher_12_feet & is_building & constructed_before_date]

    # Filter the areas
    df['the_geom'] = df['the_geom'].apply(lambda x: shape(x) if x is not None else x)
    df = df.set_geometry('the_geom', crs="EPSG:4326")
    df = df.cx[COORDS[0]:COORDS[2], COORDS[1]:COORDS[3]]

    # Calculate ground area
    df = df.to_crs(epsg=2263)
    df['shape_area'] = df['the_geom'].area
    larger_400_feet = df['shape_area'] >= 400
    df = df[larger_400_feet]

    # Save data
    df = df.to_crs(epsg=4326)
    df.to_file(SAVE_DIR + savefile, driver="GeoJSON")
    print(f"Data is saved at {SAVE_DIR + savefile}.")


def filter_street(readfile, savefile):
    # Read data from geojson
    df = gpd.read_file(readfile)
    print("Load data successfully!")

    # Filter condition
    df['RW_TYPE'] = df['RW_TYPE'].str.strip()

    is_street = ~df['FeatureTyp'].isin(['2', '5', '7', '9', 'F'])
    not_imaginary = ~df['SegmentTyp'].isin(['G', 'F'])
    canyon_type = ~df['RW_TYPE'].isin(['4', '12', '14'])
    constructed = df['Status'] == '2'

    df = df[is_street & not_imaginary & canyon_type & constructed]

    # Filter feature
    feature_to_keep = ['OBJECTID', 'Join_ID', 'StreetCode', 'Street', 
                       'TrafDir', 'StreetWidth_Min', 'StreetWidth_Max', 'RW_TYPE', 'POSTED_SPEED',
                       'Number_Travel_Lanes', 'Number_Park_Lanes', 'Number_Total_Lanes',
                       'FeatureTyp', 'SegmentTyp', 'BikeLane', 'BIKE_TRAFDIR',
                       'XFrom', 'YFrom', 'XTo', 'YTo', 'ArcCenterX', 'ArcCenterY',
                       'NodeIDFrom', 'NodeIDTo', 'NodeLevelF', 'NodeLevelT',
                       'TRUCK_ROUTE_TYPE', 'Shape__Length', 'geometry']
    
    df = df[feature_to_keep]

    # Filter area
    df = df.set_geometry('geometry', crs="EPSG:4326")
    df = df.cx[COORDS[0]:COORDS[2], COORDS[1]:COORDS[3]]

    # Save data
    df.to_file(SAVE_DIR + savefile, driver="GeoJSON")
    print(f"Data is saved at {SAVE_DIR + savefile}.")


def filter_zoning(readfile_lst, savefile_lst):
    # Read data from geojson
    for i, readfile in enumerate(readfile_lst):
        df = gpd.read_file(READ_DIR + readfile)
        print("Load data successfully!")

        # Filter condition
        df = df.set_geometry('geometry', crs="EPSG:4326")
        df = df.cx[COORDS[0]:COORDS[2], COORDS[1]:COORDS[3]]

        # Save data
        df.to_file(SAVE_DIR + savefile_lst[i], driver="GeoJSON")
        print(f"Data is saved at {SAVE_DIR + savefile_lst[i]}.")


def filter_population_tiff(pop_file, savefile):
    # Create a Polygon (bounding box) using Shapely
    bbox_geom = box(COORDS[0], COORDS[1], COORDS[2], COORDS[3])

    # Load the shapely geometry into GeoDataFrame for masking
    gdf = gpd.GeoDataFrame({"geometry": [bbox_geom]}, crs="EPSG:4326")

    # Open the TIFF file
    with rasterio.open(pop_file) as src:
        gdf = gdf.to_crs(src.crs)
        out_image, out_transform = mask(src, gdf.geometry, crop=True)
        out_image = out_image.squeeze()

        # Save the clipped image to a new file
        with rasterio.open(savefile, 'w', savefile, 'w', driver='GTiff', count=1, crs=gdf.crs,
                    dtype=out_image.dtype,
                    height=out_image.shape[0], width=out_image.shape[1]) as dst:
            dst.write(out_image)
            dst.transform = out_transform


def filter_traffic(readfile, savefile):
    # # Read data from json
    # features = []
    # with open(readfile, "r", encoding="utf-8") as f:
    #     for feature in ijson.items(f, "item"): 
    #         features.append(feature)
    # print("Load data sucessfully!")

    # df = gpd.GeoDataFrame(features)

    # # Drop missing value
    # df = df.dropna(subset=['the_geom', 'bin', 'cnstrct_yr', 'heightroof'], how='any')

    # # Convert data type
    # df['lstmoddate'] = pd.to_datetime(df['lstmoddate'])
    # df['bin'] = df['bin'].astype('int')
    # df['cnstrct_yr'] = df['cnstrct_yr'].astype('int')
    # df['heightroof'] = df['heightroof'].astype('float')
    # df['feat_code'] = df['feat_code'].astype('int')

    # # Filter condition
    # built_before_2021 = df['cnstrct_yr'] <= 2021
    # in_mahanttan_bronx = (df['bin'] // 10**6).isin([1, 2])
    # higher_12_feet = df['heightroof'] >= 12
    # is_building = df['feat_code'].isin([1006, 2100])
    # constructed_before_date = (df['lstmoddate'] < '2021-07-24') & (df['lststatype'].isin(['Constructed']))
    # df = df[built_before_2021 & in_mahanttan_bronx & higher_12_feet & is_building & constructed_before_date]

    # # Filter the areas
    # df['the_geom'] = df['the_geom'].apply(lambda x: shape(x) if x is not None else x)
    # df = df.set_geometry('the_geom', crs="EPSG:4326")
    # df = df.cx[COORDS[0]:COORDS[2], COORDS[1]:COORDS[3]]

    # # Calculate ground area
    # df = df.to_crs(epsg=2263)
    # df['shape_area'] = df['the_geom'].area
    # larger_400_feet = df['shape_area'] >= 400
    # df = df[larger_400_feet]

    # # Save data
    # df = df.to_crs(epsg=4326)
    # df.to_file(SAVE_DIR + savefile, driver="GeoJSON")
    # print(f"Data is saved at {SAVE_DIR + savefile}.")

    pass


if __name__ == "__main__":
    # readfiles = ['building.json', 'LION.geojson']
    # savefiles = ['building.geojson', 'street.geojson']

    # for i in range(len(readfiles)):
    #     if os.path.exists(SAVE_DIR + savefiles[i]):
    #         print("Data file already exists")
    #     else: 
    #         readfile = READ_DIR + readfiles[i]
    #         if 'building' in readfile:
    #             filter_building(readfile, savefiles[i])
    #         elif 'LION' in readfile:
    #             filter_street(readfile, savefiles[i])

    # readfile_lst = ['nyco.geojson', 'nysp.geojson', 'nyzd.geojson']
    # savefile_lst = ['nyco.geojson', 'nysp.geojson', 'nyzd.geojson']

    # filter_zoning(readfile_lst, savefile_lst)

    pop_file = "GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R5_C11/GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0_R5_C11.tif"
    filter_population_tiff(READ_DIR + pop_file, "data_pipeline/data/tiff/population_res100.tiff")