import json
import yaml
import geopandas as gpd
import pandas as pd
from shapely.geometry import shape
import os


with open("data_pipeline/config.yaml", "r") as file:
        config = yaml.safe_load(file)
COORDS = config['coords']

SAVE_DIR = "data_pipeline/data/preprocessed/"
os.makedirs(SAVE_DIR, exist_ok=True)

def filter_building(readfile, savefile):
    with open(readfile, "r") as f:
        df = gpd.GeoDataFrame(json.load(f))

    # Drop missing value
    df = df.dropna(subset=['the_geom', 'bin', 'cnstrct_yr', 'heightroof'], how='any')
    
    # Filter the areas
    df['the_geom'] = df['the_geom'].map(lambda x: shape(x) if x is not None else x)
    df = df.set_geometry('the_geom', crs="EPSG:4326")
    df = df.cx[COORDS[1]:COORDS[3], COORDS[0]:COORDS[2]]

    # Convert data type
    df = df.to_crs(epsg=2263)
    df['lstmoddate'] = pd.to_datetime(df['lstmoddate'])
    df['bin'] = df['bin'].astype('int')
    df['cnstrct_yr'] = df['cnstrct_yr'].astype('int')
    df['heightroof'] = df['heightroof'].astype('float')

    # Calculate ground area
    df['shape_area'] = df['the_geom'].area

    # Filter condition
    built_before_2021 = df['cnstrct_yr'] <= 2021
    in_mahanttan_bronx = (df['bin'] // 10**6).isin([1, 2])
    higher_12_feet = df['heightroof'] >= 12
    larger_400_feet = df['shape_area'] >= 400
    is_building = df['feat_code'].isin([1006, 2100])
    constructed_before_date = (df['lstmoddate'] < '2021-07-24') & (df['lststatype'].isin(['Constructed']))

    df = df[built_before_2021 & in_mahanttan_bronx & higher_12_feet & larger_400_feet & is_building & constructed_before_date]
    df.to_json(SAVE_DIR + savefile, orient="records", indent=4)


if __name__ == "__main__":
     readfile = "data_pipeline/data/raw/building.json"
     filter_building(readfile, "building.json")

