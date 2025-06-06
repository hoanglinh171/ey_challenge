import yaml
import rasterio
import pystac_client
import planetary_computer 
import rioxarray
from odc.stac import stac_load

# lon low, lat low, lon upp, lat upp
COORDS =  (-74.01, 40.75, -73.86, 40.88)
TIME_WINDOW = "2021-06-01/2021-09-01"

def load_image(config, source):
    api_url = config['url']
    cloud_cover =  config['params']['cloud_cover']
    crs =  config['params']['crs']
    chunks =  config['params']['chunks']
    data_type =  config['params']['dtype']
    satellite_lst = config['satellites']

    if source == 'landsat_8':
        elem = 0
        query = {"eo:cloud_cover": {"lt": cloud_cover},"platform": {"in": satellite_lst[elem]['platform']}}
    elif source == 'sentinel_2': 
        elem = 1
        query = {"eo:cloud_cover": {"lt": cloud_cover}}

    collections = satellite_lst[elem]['collection']
    resolution = satellite_lst[elem]['resolution']
    bands = satellite_lst[elem]['bands']
    image_choice = satellite_lst[elem]['image_choice']

    stac = pystac_client.Client.open(api_url)
    search = stac.search(
        bbox=COORDS, 
        datetime=TIME_WINDOW,
        collections=collections,
        query=query
    )

    items = list(search.get_items())
    data = stac_load(
        items,
        bands=bands,
        crs=crs, # Latitude-Longitude
        resolution=resolution, # Degrees
        chunks=chunks,
        dtype=data_type,
        patch_url=planetary_computer.sign,
        bbox=COORDS
    )

    return data.isel(time=image_choice)


def process_landsat(config, data):
    landsat_params = config['satellites'][0]
    bands = landsat_params['bands']
    lst_bands = bands[-1]
    bands_no_lst = bands[:-1]

    scale1 = 0.0000275 
    offset1 = -0.2 
    data[bands_no_lst] = data[bands_no_lst].astype(float) * scale1 + offset1

    scale2 = 0.00341802 
    offset2 = 149.0 
    kelvin_celsius = 273.15 # convert from Kelvin to Celsius
    data[lst_bands] = data[lst_bands].astype(float) * scale2 + offset2 - kelvin_celsius

    return data


def save_raster(data, filename, config, source):
    if source == 'landsat_8':
        params = config['satellites'][0]
    elif source == 'sentinel_2': 
        params = config['satellites'][1]
    
    bands = params['bands']
    height = data.dims["latitude"]
    width = data.dims["longitude"]
        
    # print(height, width)

    gt = rasterio.transform.from_bounds(COORDS[0], COORDS[1], COORDS[2], COORDS[3], width, height)
    data.rio.write_crs("epsg:4326", inplace=True)
    data.rio.write_transform(transform=gt, inplace=True)

    n_layers = len(data)
    layers = list(data.data_vars)
    with rasterio.open(filename,'w', driver='GTiff', width=width, height=height,
                   crs='epsg:4326', transform=gt, count=n_layers, compress='lzw', dtype='float64') as dst:
        for i in range(n_layers):
            dst.write(data[layers[i]], i+1)
            dst.set_band_description(i+1, f'{bands[i]}')
        dst.close()


if __name__ == "__main__":
    with open("data_pipeline/config.yaml", "r") as file:
        config = yaml.safe_load(file)

    config = config['satellite_config']

    data_landsat = load_image(config, "landsat_8")
    data_landsat = process_landsat(config, data_landsat)
    print(len(data_landsat))
    save_raster(data_landsat, "data_pipeline/data/tiff/landsat_8.tiff", config, "landsat_8")

    data_sentinel = load_image(config, "sentinel_2")
    save_raster(data_sentinel, "data_pipeline/data/tiff/sentinel_2.tiff", config, "sentinel_2")