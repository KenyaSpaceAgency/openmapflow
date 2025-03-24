# type: ignore
import json
import os
import warnings
from datetime import date, timedelta
from typing import List, Union

import pandas as pd
from azure.storage.blob import BlobServiceClient
from dotenv import load_dotenv

from openmapflow.bands import DAYS_PER_TIMESTEP, DYNAMIC_BANDS
from openmapflow.bbox import BBox
from openmapflow.constants import END, LAT, LON, START
from openmapflow.utils import tqdm
from openmapflow.config import BucketNames

# Load environment variables
load_dotenv()

# Retrieve Azure credentials from environment
AZURE_STORAGE_ACCOUNT_NAME = os.getenv('AZURE_STORAGE_ACCOUNT_NAME')
AZURE_STORAGE_ACCOUNT_KEY = os.getenv('AZURE_STORAGE_ACCOUNT_KEY')
AZURE_CONTAINER_NAME = os.getenv('AZURE_CONTAINER_NAME', 'openmapflow-labeled-eo')

if not all([AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY, AZURE_CONTAINER_NAME]):
    raise ValueError("Missing Azure Storage credentials. Please set environment variables.")

# Construct connection string securely
connection_string = f"DefaultEndpointsProtocol=https;AccountName={AZURE_STORAGE_ACCOUNT_NAME};AccountKey={AZURE_STORAGE_ACCOUNT_KEY};EndpointSuffix=core.windows.net"

try:
    import ee
    from openmapflow.ee_boundingbox import EEBoundingBox
    from openmapflow.eo.era5 import get_single_image as get_single_era5_image
    from openmapflow.eo.sentinel1 import get_image_collection as get_s1_image_collection
    from openmapflow.eo.sentinel1 import get_single_image as get_single_s1_image
    from openmapflow.eo.sentinel2 import get_single_image as get_single_s2_image
    from openmapflow.eo.srtm import get_single_image as get_single_srtm_image

    DYNAMIC_IMAGE_FUNCTIONS = [get_single_s2_image, get_single_era5_image]
    STATIC_IMAGE_FUNCTIONS = [get_single_srtm_image]
except ImportError:
    warnings.warn("Earth Engine API is required. Install with `pip install earthengine-api`.")

# Initialize Azure Blob Service Client
try:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
except Exception as e:
    print(f"Error connecting to Azure Blob Storage: {e}")
    raise


def get_azure_blob_list() -> List[str]:
    """
    Lists available TIF files in the local `openmapflow/satdata` directory.
    """
    satdata_dir = os.path.abspath(os.path.join(os.getcwd(), "openmapflow/satdata"))

    if not os.path.isdir(satdata_dir):
        raise ValueError(f"Local directory not found: {satdata_dir}")
    
    return [os.path.join(satdata_dir, f) for f in os.listdir(satdata_dir) if f.endswith(".tif")]


def create_ee_image(polygon: "ee.Geometry.Polygon", start_date: date, end_date: date):
    """
    Creates an Earth Engine image stack for the given time range and polygon.
    """
    image_collection_list = []
    cur_date, cur_end_date = start_date, start_date + timedelta(days=DAYS_PER_TIMESTEP)

    vv_imcol, vh_imcol = get_s1_image_collection(polygon, start_date - timedelta(days=31), end_date + timedelta(days=31))

    while cur_end_date <= end_date:
        image_list = [
            get_single_s1_image(polygon, cur_date, cur_end_date, vv_imcol, vh_imcol)
        ] + [func(polygon, cur_date, cur_end_date) for func in DYNAMIC_IMAGE_FUNCTIONS]

        image_collection_list.append(ee.Image.cat(image_list))
        cur_date += timedelta(days=DAYS_PER_TIMESTEP)
        cur_end_date += timedelta(days=DAYS_PER_TIMESTEP)

    img = ee.ImageCollection(image_collection_list).reduce(ee.Reducer.mean())
    for static_func in STATIC_IMAGE_FUNCTIONS:
        img = img.addBands(static_func(polygon))
    
    return img


class EarthEngineExporter:
    """
    Exports satellite data from Earth Engine to Azure Blob Storage.
    """

    def __init__(self, check_azure: bool = False):
        ee.Initialize()
        self.check_azure = check_azure
        self.azure_blob_list = get_azure_blob_list() if check_azure else []

    def _export_for_polygon(
        self, polygon: "ee.Geometry.Polygon", polygon_id: str, start_date: date, end_date: date
    ) -> bool:
        filename = f"tifs/{polygon_id}.tif"
        if filename in self.azure_blob_list:
            return True

        img = create_ee_image(polygon, start_date, end_date)
        try:
            task = ee.batch.Export.image.toCloudStorage(
                bucket=AZURE_CONTAINER_NAME,
                fileNamePrefix=filename,
                image=img.clip(polygon),
                description=filename.replace("/", "-"),
                scale=10,
                region=polygon,
                maxPixels=1e13,
            )
            task.start()
        except ee.ee_exception.EEException as e:
            print(f"Task not started! Got exception {e}")
        return True

    def export_for_bbox(self, bbox: BBox, bbox_name: str, start_date: date, end_date: date):
        ee_bbox = EEBoundingBox.from_bounding_box(bbox)
        return self._export_for_polygon(ee_bbox.to_ee_polygon(), bbox_name, start_date, end_date)

    def export_for_labels(self, labels: pd.DataFrame):
        assert all(col in labels for col in [START, END, LAT, LON])
        labels[START] = pd.to_datetime(labels[START]).dt.date
        labels[END] = pd.to_datetime(labels[END]).dt.date
        
        for _, row in tqdm(labels.iterrows(), total=len(labels), desc="Exporting"):
            ee_bbox = EEBoundingBox.from_centre(row[LAT], row[LON], surrounding_metres=80)
            self._export_for_polygon(ee_bbox.to_ee_polygon(), ee_bbox.get_identifier(row[START], row[END]), row[START], row[END])
