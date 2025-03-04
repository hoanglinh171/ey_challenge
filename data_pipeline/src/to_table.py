import os
import yaml
import numpy as np
import geopandas as gpd
import pandas as pd
from tqdm import tqdm
import rasterio
import rioxarray as rxr

CSV_PATH = "Training_data_uhi_index_2025-02-18.csv"
READ_DIR = "data_pipeline/data/tiff/"
SAVE_DIR = "data_pipeline/data/tabular_data/"
os.makedirs(SAVE_DIR, exist_ok=True)

with open("data_pipeline/config.yaml", "r") as file:
    config = yaml.safe_load(file)
COORDS = config['coords']


def map_satellite_data(tiff_path, csv_path, save_path):
    
    # Load the GeoTIFF data
    data = rxr.open_rasterio(tiff_path)
    layer_num = 0
    with rasterio.open(tiff_path) as dts:
        layer_num = dts.count
        layer_names = dts.descriptions

    # Read the Excel file using pandas
    df = pd.read_csv(csv_path)
    latitudes = df['Latitude'].values
    longitudes = df['Longitude'].values   

    df = pd.DataFrame()
    df['lat'] = latitudes
    df['long'] = longitudes

    for i in tqdm(range(layer_num), desc="Go through layer"):
        values = []
        # Iterate over the latitudes and longitudes, and extract the corresponding band values
        for lat, lon in tqdm(zip(latitudes, longitudes), total=len(latitudes), desc="Mapping values"):
        # Assuming the correct dimensions are 'y' and 'x' (replace these with actual names from data.coords)
            cell_value = data.sel(x=lon, y=lat,  band=i+1, method="nearest").values
            values.append(cell_value)
        # Add column of feature
        df[layer_names[i]] = values

    df.to_csv(save_path, index=False)
    print("File saved!")
    
    return df


if __name__ == "__main__":
    CSV_PATH = "Training_data_uhi_index_2025-02-18.csv"

    read_folder = os.path.join(READ_DIR, "1x1")
    save_folder = os.path.join(SAVE_DIR, "train/1x1")
    for i, filename in tqdm(enumerate(os.listdir(read_folder)), desc="1x1"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            csv_path = os.path.splitext(filename)[0] + ".csv"
            save_path = os.path.join(save_folder, csv_path)
            if os.path.exists(save_path):
                print("File exists!")
            else:
                map_satellite_data(tiff_path=file_path, csv_path=CSV_PATH, save_path=save_path)

    read_folder = os.path.join(READ_DIR, "3x3")
    save_folder = os.path.join(SAVE_DIR, "train/3x3")
    for i, filename in tqdm(enumerate(os.listdir(read_folder)), desc="3x3"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            csv_path = os.path.splitext(filename)[0] + ".csv"
            save_path = os.path.join(save_folder, csv_path)
            if os.path.exists(save_path):
                print("File exists!")
            else:
                map_satellite_data(tiff_path=file_path, csv_path=CSV_PATH, save_path=save_path)

    read_folder = os.path.join(READ_DIR, "5x5")
    save_folder = os.path.join(SAVE_DIR, "train/5x5")
    for i, filename in tqdm(enumerate(os.listdir(read_folder)), desc="5x5"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            csv_path = os.path.splitext(filename)[0] + ".csv"
            save_path = os.path.join(save_folder, csv_path)
            if os.path.exists(save_path):
                print("File exists!")
            else:
                map_satellite_data(tiff_path=file_path, csv_path=CSV_PATH, save_path=save_path)


    read_folder = os.path.join(READ_DIR, "9x9")
    save_folder = os.path.join(SAVE_DIR, "train/9x9")
    for i, filename in tqdm(enumerate(os.listdir(read_folder)), desc="9x9"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            csv_path = os.path.splitext(filename)[0] + ".csv"
            save_path = os.path.join(save_folder, csv_path)
            if os.path.exists(save_path):
                print("File exists!")
            else:
                map_satellite_data(tiff_path=file_path, csv_path=CSV_PATH, save_path=save_path)
    

    

    CSV_PATH = "Submission_template_UHI2025-v2.csv"

    read_folder = os.path.join(READ_DIR, "1x1")
    save_folder = os.path.join(SAVE_DIR, "submission/1x1")
    for i, filename in tqdm(enumerate(os.listdir(read_folder)), desc="1x1"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            csv_path = os.path.splitext(filename)[0] + ".csv"
            save_path = os.path.join(save_folder, csv_path)
            if os.path.exists(save_path):
                print("File exists!")
            else:
                map_satellite_data(tiff_path=file_path, csv_path=CSV_PATH, save_path=save_path)

    read_folder = os.path.join(READ_DIR, "3x3")
    save_folder = os.path.join(SAVE_DIR, "submission/3x3")
    for i, filename in tqdm(enumerate(os.listdir(read_folder)), desc="3x3"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            csv_path = os.path.splitext(filename)[0] + ".csv"
            save_path = os.path.join(save_folder, csv_path)
            if os.path.exists(save_path):
                print("File exists!")
            else:
                map_satellite_data(tiff_path=file_path, csv_path=CSV_PATH, save_path=save_path)

    read_folder = os.path.join(READ_DIR, "5x5")
    save_folder = os.path.join(SAVE_DIR, "submission/5x5")
    for i, filename in tqdm(enumerate(os.listdir(read_folder)), desc="5x5"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            csv_path = os.path.splitext(filename)[0] + ".csv"
            save_path = os.path.join(save_folder, csv_path)
            if os.path.exists(save_path):
                print("File exists!")
            else:
                map_satellite_data(tiff_path=file_path, csv_path=CSV_PATH, save_path=save_path)


    read_folder = os.path.join(READ_DIR, "9x9")
    save_folder = os.path.join(SAVE_DIR, "submission/9x9")
    for i, filename in tqdm(enumerate(os.listdir(read_folder)), desc="9x9"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            file_path = os.path.join(read_folder, filename)
            csv_path = os.path.splitext(filename)[0] + ".csv"
            save_path = os.path.join(save_folder, csv_path)
            if os.path.exists(save_path):
                print("File exists!")
            else:
                map_satellite_data(tiff_path=file_path, csv_path=CSV_PATH, save_path=save_path)
    