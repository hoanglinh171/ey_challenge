import numpy as np
from collections import deque
import yaml
import os
import rasterio
from tqdm import tqdm
from skimage.feature import graycomatrix, graycoprops


READ_DIR = "data_pipeline/data/tiff/1x1/"
SAVE_DIR = "data_pipeline/data/tiff/"
os.makedirs(SAVE_DIR, exist_ok=True)

with open("data_pipeline/config.yaml", "r") as file:
    config = yaml.safe_load(file)
COORDS = config['coords']
CRS = config['gge_engine_config']['crs']


def check_range(array, min, max):
    array[array < min] = min
    array[array > max] = max

    return array

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

    ndvi = check_range((nir08 - red) / (nir08 + red), -1, 1)
    evi = check_range(2.5 * (nir08 - red) / (nir08 + 6 * red - 7.5 * blue + 1 + 1e-10), -1, 1)
    savi = check_range((nir08 - red) * 1.5 / (nir08 + red + 0.5 + 1e-10), -1.5, 1.5)
    gndvi = check_range((nir08 - green) / (nir08 + green + 1e-10), -1, 1)
    arvi = check_range((nir08 - (red - (blue - red))) / (nir08 + (red - (blue - red)) + 1e-10), -1, 1)

    term = (2 * nir08 + 1) ** 2 - 8 * (nir08 - red)
    term = np.maximum(term, 0)  # Ensure no negative values inside sqrt
    msavi = check_range((2 * nir08 + 1 - np.sqrt(term)) / 2, 0, 1)

    ndwi = check_range((green - nir08) / (green + nir08), -1, 1)
    mndwi = check_range((green - swir16) / (green + swir16 + 1e-10), -1, 1)
    awei_nsh = 4 * (green - swir16) - (0.25 * nir08 + 2.75 * swir22)
    awei_sh = green + 2.5 * nir08 - 1.5 * (swir16 + swir22) - 0.25 * blue
    ndsi = check_range((green - swir16) / (green + swir16 + 1e-10), -1, 1)
    nbr = check_range((nir08 - swir22) / (nir08 + swir22 + 1e-10), -1, 1)
    si = check_range((swir16 - blue) / (swir16 + blue + 1e-10), -1, 1)
    ndbi = check_range((swir16 - nir08) / (swir16 + nir08 + 1e-10), -1, 1)
    ui = check_range((swir16 - red) / (swir16 + red + 1e-10), -1, 1)
    ibi = check_range((ndbi - (savi + mndwi)) / (ndbi + (savi + mndwi) + 1e-10), -1, 1)
    albedo = check_range((0.356 * blue + 0.130 * red + 0.373 * nir08 + 0.085 * swir16 + 0.072 * swir22 - 0.018) / 1.016, 0, 1)

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
    meta['count'] = 2
    with rasterio.open(savefile, 'w', **meta) as dst:
        for i, band in enumerate(bands):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_res{resolution}')


def extract_glcm(tiff, savefile, distance=1, size=1):
    ops = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    glcm_arr = []
    glcm_name = []

    with rasterio.open(tiff) as dst:
        meta = dst.meta
        desc = dst.descriptions

        for k in range(dst.count):
            raster_arr = dst.read(k+1)

            raster_scaled = ((raster_arr - raster_arr.min()) / (raster_arr.max() - raster_arr.min() + 1) * 16).astype(np.uint8)
            m, n = raster_scaled.shape

            for op in ops:
                smoothed_raster = np.zeros((m, n))
                for i in tqdm(range(m)):
                    for j in range(n):
                        neighbors = raster_scaled[max(i-size, 0):min(i+1+size, m), max(j-size, 0):min(j+1+size, n)]
                        glcm = graycomatrix(neighbors, distances=[1], angles=[np.pi/4], levels=16, symmetric=True, normed=True)
                        smoothed_raster[i, j] = graycoprops(glcm, op)[0, 0]

                glcm_arr.append(smoothed_raster)
                glcm_name.append(f'{desc[k]}_{op}_{size*2 + 1}x{size*2 + 1}')


    meta['count'] = len(glcm_arr)
    with rasterio.open(savefile, 'w', **meta) as dst:
        for i, band in enumerate(glcm_arr):
            dst.write(band, i+1)
            dst.set_band_description(i+1, glcm_name[i])


def calculate_distance(raster_arr):
    rows, cols = raster_arr.shape
    dist = np.full((rows, cols), np.inf)
    queue = deque()
    
    # Initialize queue with all '1' positions and set their distance to 0
    ones = np.argwhere(raster_arr > 0)
    for r, c in ones:
        dist[r, c] = 0
        queue.append((r, c))
    
    directions = [
        (-1, 0, 1), (1, 0, 1), (0, -1, 1), (0, 1, 1),  # Up, Down, Left, Right (distance = 1)
        (-1, -1, 2**0.5), (-1, 1, 2**0.5), (1, -1, 2**0.5), (1, 1, 2**0.5)  # Diagonal moves (distance = sqrt(2))
    ]
    
    # BFS to compute shortest distances
    while queue:
        r, c = queue.popleft()
        for dr, dc, cost in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                new_dist = dist[r, c] + cost
                if new_dist < dist[nr, nc]:
                    dist[nr, nc] = new_dist
                    queue.append((nr, nc))
    
    return dist


def zoning_distance(tiff_file, savefile):
    # calculate the distance of a pixel to the closest distance
    raster_arr = []
    with rasterio.open(tiff_file) as dst:
        band_names = dst.descriptions[3:]
        meta = dst.meta
        for i in range(3, 7, 1):
            band = dst.read(i + 1)
            raster_arr.append(band)

    distance_raster_arr = []
    for i, raster in tqdm(enumerate(raster_arr), desc="Iterating bands"):
        distance_raster = calculate_distance(raster)
        distance_raster_arr.append(distance_raster)

    # save_tiff
    meta['count'] = 4
    with rasterio.open(savefile, 'w', **meta) as dst:
        for i, band in enumerate(distance_raster_arr):
            dst.write(band, i+1)
            dst.set_band_description(i+1, f'{band_names[i]}_distance')


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
    # tiff = READ_DIR + "landsat_8.tiff"
    # savefile = SAVE_DIR + "1x1/landsat_indices.tiff"
    # calculate_indices(tiff=tiff, source="landsat_8", savefile=savefile, resolution=30)

    # building_tiff = os.path.join(READ_DIR + "building_res100.tiff")
    # street_tiff = os.path.join(READ_DIR + "street_res100.tiff")
    # savefile = os.path.join(SAVE_DIR, "1x1/building_street_res100.tiff")
    # building_street_features(building_tiff, street_tiff, savefile, resolution=100)

    # zoning_tiff = os.path.join(READ_DIR + "nyzd_res30.tiff")
    # savefile = os.path.join(SAVE_DIR, "1x1/zoning_res30_distance.tiff")
    # zoning_distance(zoning_tiff, savefile)

    glcm_lst = ['landsat', 'building', 'street', 'canopy', 'aod', 'population', 'co', 'hcho', 'no2', 'o3']
    size = 1
    for i, filename in tqdm(enumerate(os.listdir(READ_DIR)), desc="3x3"):
        print(filename)
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            if (filename.startswith("surface")):
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

                if any(filename.startswith(prefix) for prefix in glcm_lst):
                    savefile = os.path.join(SAVE_DIR, f"3x3/{size*2+1}x{size*2+1}_glcm_{filename}")
                    if os.path.exists(savefile):
                        print("File exists!")
                    else:
                        extract_glcm(file_path, savefile, distance=1, size=size)

    
    size = 2
    for i, filename in tqdm(enumerate(os.listdir(READ_DIR)), desc="5x5"):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            if (filename.startswith("surface")):
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

                if any(filename.startswith(prefix) for prefix in glcm_lst):
                    savefile = os.path.join(SAVE_DIR, f"5x5/{size*2+1}x{size*2+1}_glcm_{filename}")
                    if os.path.exists(savefile):
                        print("File exists!")
                    else:
                        extract_glcm(file_path, savefile, distance=1, size=size)


    size = 4
    for i, filename in tqdm(enumerate(os.listdir(READ_DIR)), desc="9x9"):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            if (filename.startswith("surface")):
                file_path = os.path.join(READ_DIR, filename)
                print(filename)

                operation = 'mean'
                savefile = os.path.join(SAVE_DIR, f"9x9/{size*2+1}x{size*2+1}_{operation}_{filename}")
                if os.path.exists(savefile):
                    print("File exists!")
                else:
                    smooth_tiff_file(file_path, savefile, operation, size=size)

                operation = 'std_dev'
                savefile = os.path.join(SAVE_DIR, f"9x9/{size*2+1}x{size*2+1}_{operation}_{filename}")
                if os.path.exists(savefile):
                    print("File exists!")
                else:
                    smooth_tiff_file(file_path, savefile, operation, size=size)

                if any(filename.startswith(prefix) for prefix in glcm_lst):
                    savefile = os.path.join(SAVE_DIR, f"9x9/{size*2+1}x{size*2+1}_glcm_{filename}")
                    if os.path.exists(savefile):
                        print("File exists!")
                    else:
                        extract_glcm(file_path, savefile, distance=2, size=size)


    size = 7
    for i, filename in tqdm(enumerate(os.listdir(READ_DIR)), desc="15x15"):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            print(filename)

            if (filename.startswith("surface")):
                file_path = os.path.join(READ_DIR, filename)

                operation = 'mean'
                savefile = os.path.join(SAVE_DIR, f"15x15/{size*2+1}x{size*2+1}_{operation}_{filename}")
                if os.path.exists(savefile):
                    print("File exists!")
                else:
                    smooth_tiff_file(file_path, savefile, operation, size=size)

                operation = 'std_dev'
                savefile = os.path.join(SAVE_DIR, f"15x15/{size*2+1}x{size*2+1}_{operation}_{filename}")
                if os.path.exists(savefile):
                    print("File exists!")
                else:
                    smooth_tiff_file(file_path, savefile, operation, size=size)

                if any(filename.startswith(prefix) for prefix in glcm_lst):
                    savefile = os.path.join(SAVE_DIR, f"15x15/{size*2+1}x{size*2+1}_glcm_{filename}")
                    if os.path.exists(savefile):
                        print("File exists!")
                    else:
                        extract_glcm(file_path, savefile, distance=3, size=size)
            
    
    size = 12
    for i, filename in tqdm(enumerate(os.listdir(READ_DIR)), desc="25x25"):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            print(filename)
            if (filename.startswith("surface")):
                # if (filename.startswith("landsat") or filename.startswith("canopy") or filename.startswith("building") or filename.startswith("street")):
                file_path = os.path.join(READ_DIR, filename)

                operation = 'mean'
                savefile = os.path.join(SAVE_DIR, f"25x25/{size*2+1}x{size*2+1}_{operation}_{filename}")
                if os.path.exists(savefile):
                    print("File exists!")
                else:
                    smooth_tiff_file(file_path, savefile, operation, size=size)

                operation = 'std_dev'
                savefile = os.path.join(SAVE_DIR, f"25x25/{size*2+1}x{size*2+1}_{operation}_{filename}")
                if os.path.exists(savefile):
                    print("File exists!")
                else:
                    smooth_tiff_file(file_path, savefile, operation, size=size)

                if any(filename.startswith(prefix) for prefix in glcm_lst):
                    savefile = os.path.join(SAVE_DIR, f"25x25/{size*2+1}x{size*2+1}_glcm_{filename}")
                    if os.path.exists(savefile):
                        print("File exists!")
                    else:
                        extract_glcm(file_path, savefile, distance=5, size=size)


    size = 25
    for i, filename in tqdm(enumerate(os.listdir(READ_DIR)), desc="51x51"):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            print(filename)
            if (filename.startswith("surface")):
                # (filename.startswith("landsat") or filename.startswith("canopy") or filename.startswith("building") or filename.startswith("street")):
                file_path = os.path.join(READ_DIR, filename)

                operation = 'mean'
                savefile = os.path.join(SAVE_DIR, f"51x51/{size*2+1}x{size*2+1}_{operation}_{filename}")
                if os.path.exists(savefile):
                    print("File exists!")
                else:
                    smooth_tiff_file(file_path, savefile, operation, size=size)

                operation = 'std_dev'
                savefile = os.path.join(SAVE_DIR, f"51x51/{size*2+1}x{size*2+1}_{operation}_{filename}")
                if os.path.exists(savefile):
                    print("File exists!")
                else:
                    smooth_tiff_file(file_path, savefile, operation, size=size)
                
                if any(filename.startswith(prefix) for prefix in glcm_lst):
                    savefile = os.path.join(SAVE_DIR, f"51x51/{size*2+1}x{size*2+1}_glcm_{filename}")
                    if os.path.exists(savefile):
                        print("File exists!")
                    else:
                        extract_glcm(file_path, savefile, distance=11, size=size)


    size = 50
    for i, filename in tqdm(enumerate(os.listdir(READ_DIR)), desc="101x101"):
        if filename.endswith(".tif") or filename.endswith(".tiff"):
            print(filename)
            if (filename.startswith("surface")):
                #  or filename.startswith("canopy") or filename.startswith("building") or filename.startswith("street")):
                file_path = os.path.join(READ_DIR, filename)

                operation = 'mean'
                savefile = os.path.join(SAVE_DIR, f"101x101/{size*2+1}x{size*2+1}_{operation}_{filename}")
                if os.path.exists(savefile):
                    print("File exists!")
                else:
                    smooth_tiff_file(file_path, savefile, operation, size=size)

                operation = 'std_dev'
                savefile = os.path.join(SAVE_DIR, f"101x101/{size*2+1}x{size*2+1}_{operation}_{filename}")
                if os.path.exists(savefile):
                    print("File exists!")
                else:
                    smooth_tiff_file(file_path, savefile, operation, size=size)
                
                if any(filename.startswith(prefix) for prefix in glcm_lst):
                    savefile = os.path.join(SAVE_DIR, f"101x101/{size*2+1}x{size*2+1}_glcm_{filename}")
                    if os.path.exists(savefile):
                        print("File exists!")
                    else:
                        extract_glcm(file_path, savefile, distance=21, size=size)