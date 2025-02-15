import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
import os

READ_DIR = "data_pipeline/data/preprocessed/"
SAVE_DIR = "data_pipeline/data/features_extracted/"
os.makedirs(name=SAVE_DIR, exist_ok=True)

def extract_feature_building(readfile, savefile):
    df = gpd.read_file(readfile)

    # Convert data type
    df['bin'] = df['bin'].astype('int')
    df['cnstrct_yr'] = df['cnstrct_yr'].astype('int')
    df['heightroof'] = df['heightroof'].astype('float')
    df['shape_area'] = df['shape_area'].astype('float')

    # Extract feature
    df = df[['bin', 'cnstrct_yr', 'heightroof', 'shape_area', 'geometry']]

    # Save data
    df.to_file(SAVE_DIR + savefile, driver="GeoJSON")
    print(f"Data is saved at {SAVE_DIR + savefile}.")


def convert_to_float(feature):
    feature = feature.str.strip()
    feature = np.where(feature == '', None, feature)

    return feature.astype('float')


def feet_to_degree(df, lon, lat):  # lon, lat:
    transformer = Transformer.from_crs("EPSG:2263", "EPSG:4326", always_xy=True)
    spatial_degree = df.apply(lambda row: pd.Series(transformer.transform(row[lon], row[lat])), axis=1)

    return spatial_degree


def street_direction(xfrom, yfrom, xto, yto):
    dx = xto - xfrom  # Change in longitude
    dy = yto - yfrom  # Change in latitude
    
    # Convert angle to degrees
    angle = np.degrees(np.arctan2(dy, dx))  
    
    # Adjust to fit compass direction
    angle = np.where(angle < 0, angle+360, angle)
    
    # Determine direction based on angle
    condlist = [
        ((22.5 <= angle) & (angle < 67.5)) | ((157.5 <= angle) & (angle < 202.5)),  # NE-SW
        ((67.5 <= angle) & (angle < 112.5)) | ((247.5 <= angle) & (angle < 292.5)), # N-S
        ((112.5 <= angle) & (angle < 157.5)) | ((292.5 <= angle) & (angle < 337.5)) # NW-SE
    ]
    choicelist = ["NE-SW", "N-S", "NW-SE"]
    # choicelist = ['NW-SE', 'N-S', 'NE-SW']

    return np.select(condlist, choicelist, "E-W")


def extract_feature_street(readfile, savefile):
    df = gpd.read_file(readfile)

    # Convert string to float
    df['RW_TYPE'] = convert_to_float(df['RW_TYPE']).astype('int')
    df['Number_Travel_Lanes'] = convert_to_float(df['Number_Travel_Lanes'])
    df['Number_Park_Lanes'] = convert_to_float(df['Number_Park_Lanes'])
    df['Number_Total_Lanes'] = convert_to_float(df['Number_Total_Lanes'])
    df['POSTED_SPEED'] = convert_to_float(df['POSTED_SPEED'])
    df['BikeLane'] = convert_to_float(df['BikeLane'])   
    df['BIKE_TRAFDIR'] = np.where(df['BIKE_TRAFDIR'].str.strip() == '', None, df['BIKE_TRAFDIR'].str.strip())
    df['TRUCK_ROUTE_TYPE'] = convert_to_float(df['TRUCK_ROUTE_TYPE'])

    # Calculate average street width
    df['street_width_avg'] = (df['StreetWidth_Min'] + df['StreetWidth_Max']) / 2

    # Calculate street orientation
    df[['XFrom', 'YFrom']] = feet_to_degree(df, 'XFrom', 'YFrom')
    df[['XTo', 'YTo']] = feet_to_degree(df, 'XTo', 'YTo')
    df['direction'] = street_direction(df['XFrom'], df['YFrom'], df['XTo'], df['YTo'])

    # Extract Feature
    df = df[['OBJECTID', 'Join_ID', 'StreetCode', 'Street', 'TrafDir',
            'StreetWidth_Min', 'StreetWidth_Max', 'street_width_avg', 'RW_TYPE', 'POSTED_SPEED',
            'Number_Travel_Lanes', 'Number_Park_Lanes', 'Number_Total_Lanes',
            'FeatureTyp', 'SegmentTyp', 'BikeLane', 'BIKE_TRAFDIR', 'TRUCK_ROUTE_TYPE',
            'Shape__Length', 'direction', 'geometry']]

    # Save data
    df.to_file(SAVE_DIR + savefile, driver="GeoJSON")
    print(f"Data is saved at {SAVE_DIR + savefile}.")


def extract_feature_zoning_nyzd(readfile, savefile):
    df = gpd.read_file(readfile)

    # Extract from zone district
    df = df.rename(columns={'ZONEDIST': 'zonedist_level3'})
    df['zonedist_level2'] = df['zonedist_level3'].str.split("-").str[0]
    df['zonedist_level1'] = df['zonedist_level3'].str[0]

    # Save data
    df.to_file(SAVE_DIR + savefile, driver="GeoJSON")
    print(f"Data is saved at {SAVE_DIR + savefile}.")


def extract_feature_zoning_nyco(readfile, savefile):
    df = gpd.read_file(readfile)

    # Extract from zone district
    df = df.rename(columns={'OVERLAY': 'overlay_level2'})
    df['overlay_level1'] = df['overlay_level2'].str.split("-").str[0]

    # Save data
    df.to_file(SAVE_DIR + savefile, driver="GeoJSON")
    print(f"Data is saved at {SAVE_DIR + savefile}.")


def extract_feature_zoning_nysp(readfile, savefile):
    df = gpd.read_file(readfile)

    # Extract from zone district
    df = df.rename(columns={'SDLBL': 'sd_level2'})
    df['sd_level1'] = df['sd_level2'].str.split("-").str[0]

    # Save data
    df.to_file(SAVE_DIR + savefile, driver="GeoJSON")
    print(f"Data is saved at {SAVE_DIR + savefile}.")


if __name__ == "__main__":
    # readfile = READ_DIR + "building.geojson"
    # savefile = "building.geojson"
    # extract_feature_building(readfile, savefile)

    # readfile = READ_DIR + "street.geojson"
    # savefile = "street.geojson"
    # extract_feature_street(readfile, savefile)

    readfile = READ_DIR + "nyco.geojson"
    savefile = "nyco.geojson"
    extract_feature_zoning_nyco(readfile, savefile)

    readfile = READ_DIR + "nyzd.geojson"
    savefile = "nyzd.geojson"
    extract_feature_zoning_nyzd(readfile, savefile)

    readfile = READ_DIR + "nysp.geojson"
    savefile = "nysp.geojson"
    extract_feature_zoning_nysp(readfile, savefile)