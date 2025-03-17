import ijson
import yaml
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape, box
import os
import rasterio
from rasterio.mask import mask
import rioxarray as rxr
from pyproj import Transformer
from affine import Affine


READ_DIR = "data_pipeline/data/raw/"
SAVE_DIR = "data_pipeline/data/preprocessed/"
os.makedirs(SAVE_DIR, exist_ok=True)

with open("data_pipeline/config.yaml", "r") as file:
    config = yaml.safe_load(file)
COORDS = config['coords']
CRS = config['satellite_config']['params']['crs']


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
    feature_to_keep = ['OBJECTID', 'SegmentID', 'Join_ID', 'StreetCode', 'Street', 
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
        print(out_image.shape)


        # Save the clipped image to a new file
        with rasterio.open(savefile, 'w', driver='GTiff', count=1, crs=gdf.crs,
                    dtype=out_image.dtype,
                    height=out_image.shape[0], width=out_image.shape[1],
                    transform=out_transform) as dst:
            dst.write(out_image, 1)
            dst.set_band_description(1, f'population_res1000')


def crop_and_reshape(readfile, savefile):
    with rasterio.open(readfile) as src:
        source_crs = src.crs  # Get the source CRS of the TIFF file
        
    # Create a transformer to convert coordinates from target CRS to source CRS
    transformer = Transformer.from_crs(CRS, source_crs, always_xy=True)
    
    # Transform the bounding box coordinates
    xmin_source, ymin_source = transformer.transform(COORDS[0], COORDS[1])
    xmax_source, ymax_source = transformer.transform(COORDS[2], COORDS[3])
    
    # Define the bounding box in the source CRS
    bbox_source = (xmin_source, ymin_source, xmax_source, ymax_source)
    

    # Step 1: Open the TIFF file using rioxarray
    rds = rxr.open_rasterio(readfile)
    rds_cropped = rds.rio.clip_box(*bbox_source)
    rds_reprojected = rds_cropped.rio.reproject(CRS)

    # Step 4: Save the cropped and reprojected raster to a new file
    rds_reprojected.rio.to_raster(savefile)

    with rasterio.open(savefile, 'r+') as dst:
        dst.set_band_description(1, 'canopy_heigth')


def filter_elevation(readfile, savefile):
    # Read data from json
    df = gpd.read_file(readfile)
    print("Load data successfully!")

    # Drop missing value
    df = df.dropna(subset=['geometry'], how='any')

    # Convert data type
    df['elevation'] = df['elevation'].astype('float')

    # Filter the areas
    df['geometry'] = df['geometry'].apply(lambda x: shape(x) if x is not None else x)
    df = df.set_geometry('geometry', crs="EPSG:4326")
    df = df.cx[COORDS[0]:COORDS[2], COORDS[1]:COORDS[3]]

    # Save data
    df = df.to_crs(epsg=4326)
    df.to_file(SAVE_DIR + savefile, driver="GeoJSON")
    print(f"Data is saved at {SAVE_DIR + savefile}.")


def filter_hydro(readfile, savefile):
    # Read data from json
    df = gpd.read_file(readfile)
    print("Load data successfully!")

    # Drop missing value
    df = df.dropna(subset=['geometry'], how='any')

    # Convert data type
    df['shape_leng'] = df['shape_leng'].astype('float')
    df['shape_area'] = df['shape_area'].astype('float')
    df['feat_code'] = df['feat_code'].astype('float')

    # Filter the areas
    df['geometry'] = df['geometry'].apply(lambda x: shape(x) if x is not None else x)
    df = df.set_geometry('geometry', crs="EPSG:4326")
    df = df.cx[COORDS[0]:COORDS[2], COORDS[1]:COORDS[3]]

    # Filter type
    df = df[df['feat_code'].isin([2600, 2610, 2620, 2630, 2660])]

    # Save data
    df = df.to_crs(epsg=4326)
    df.to_file(SAVE_DIR + savefile, driver="GeoJSON")
    print(f"Data is saved at {SAVE_DIR + savefile}.")


if __name__ == "__main__":
    # readfiles = ['building.json', 'LION.geojson']
    # savefiles = ['building.geojson', 'street.geojson']

    # readfiles = ['LION.geojson']
    # savefiles = ['street.geojson']

    # for i in range(len(readfiles)):
        # if os.path.exists(SAVE_DIR + savefiles[i]):
        #     print("Data file already exists")
        # else: 
        # readfile = READ_DIR + readfiles[i]
        # if 'building' in readfile:
        #     filter_building(readfile, savefiles[i])
        # elif 'LION' in readfile:
        #     filter_street(readfile, savefiles[i])

    # readfile_lst = ['nyco.geojson', 'nysp.geojson', 'nyzd.geojson']
    # savefile_lst = ['nyco.geojson', 'nysp.geojson', 'nyzd.geojson']

    # filter_zoning(readfile_lst, savefile_lst)

    # pop_file = "GHS_POP_E2020_GLOBE_R2023A_4326_30ss_V1_0_R5_C11.tif"
    # filter_population_tiff(READ_DIR + pop_file, "data_pipeline/data/tiff/population_res1000.tiff")

    # readfile = READ_DIR + "032010110.tif"
    # savefile = "data_pipeline/data/tiff/1x1/canopy_height_res1.tif"
    # crop_and_reshape(readfile, savefile)

    # readfile = READ_DIR + "surface_elevation.geojson"
    # savefile = "surface_elevation.geojson"
    # filter_elevation(readfile, savefile)

    readfile = READ_DIR + "hydrography.geojson"
    savefile = "hydrography.geojson"
    filter_hydro(readfile, savefile)

