import boto3
from botocore import UNSIGNED
from botocore.config import Config
import os

SAVE_DIR = "data_pipeline/data/raw/"


def load_canopy_height(savefile):
    s3_client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
    bucket = "dataforgood-fb-data"
    s3chmpath='forests/v1/alsgedi_global_v6_float/chm'
    s3metapath='forests/v1/alsgedi_global_v6_float/metadata'
    tile = "032010110"

    s3file=f"{s3chmpath}/{tile}.tif"
    localfile=f"{savefile}/{os.path.basename(s3file)}"
    if not os.path.exists(localfile):
        s3_client.download_file(bucket, s3file, localfile)

    #download metadata
    jsonfile=f"{s3metapath}/{tile}.geojson"
    localjsonfile=f"{savefile}/{os.path.basename(jsonfile)}"
    if not os.path.exists(localjsonfile):
        s3_client.download_file(bucket, jsonfile, localjsonfile)


if __name__ == "__main__":
    load_canopy_height(SAVE_DIR)