# type: ignore
import json
import os
import warnings
from datetime import date, timedelta
from typing import Dict, List, Optional, Union

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
    warnings.warn("ee_exporter requires earthengine-api, `pip install earthengine-api`")

# Initialize Azure Blob Service Client
try:
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(AZURE_CONTAINER_NAME)
except Exception as e:
    print(f"Error connecting to Azure Blob Storage: {e}")
    raise


def get_azure_blob_list(container_name: str, region=None) -> List[str]:
    """
    Lists local file paths from a directory (simulating Azure Blob Storage).
    """
    satdata_dir_name = "openmapflow/satdata"
    local_dir = os.path.join(os.getcwd(), satdata_dir_name)
    local_dir = os.path.abspath(local_dir)

    print(f"Current working directory: {os.getcwd()}")
    print(f"Constructed satdata directory path: {local_dir}")

    if not os.path.isdir(local_dir):
        raise ValueError(f"Local directory not found: {local_dir}")

    file_paths = []
    for filename in os.listdir(local_dir):
        if filename.endswith(".tif"):
            file_path = os.path.join(local_dir, filename)
            file_paths.append(file_path)
    return file_paths


def make_combine_bands_function(bands):
    def combine_bands(current, previous):
        previous = ee.Image(previous)
        current = current.select(bands)
        return ee.Algorithms.If(
            ee.Algorithms.IsEqual(previous, None),
            current,
            previous.addBands(ee.Image(current)),
        )

    return combine_bands


def ee_safe_str(s: str):
    """Earth Engine descriptions only allow certain characters"""
    return s.replace(".", "-").replace("=", "-").replace("/", "-")[:100]


def create_ee_image(polygon: "ee.Geometry.Polygon", start_date: date, end_date: date):
    image_collection_list: List[ee.Image] = []
    cur_date = start_date
    cur_end_date = cur_date + timedelta(days=DAYS_PER_TIMESTEP)

    vv_imcol, vh_imcol = get_s1_image_collection(
        polygon, start_date - timedelta(days=31), end_date + timedelta(days=31)
    )

    while cur_end_date <= end_date:
        image_list: List[ee.Image] = []

        image_list.append(
            get_single_s1_image(
                region=polygon,
                start_date=cur_date,
                end_date=cur_end_date,
                vv_imcol=vv_imcol,
                vh_imcol=vh_imcol,
            )
        )
        for image_function in DYNAMIC_IMAGE_FUNCTIONS:
            image_list.append(image_function(region=polygon, start_date=cur_date, end_date=cur_end_date))

        image_collection_list.append(ee.Image.cat(image_list))
        cur_date += timedelta(days=DAYS_PER_TIMESTEP)
        cur_end_date += timedelta(days=DAYS_PER_TIMESTEP)

    imcoll = ee.ImageCollection(image_collection_list)
    combine_bands_function = make_combine_bands_function(DYNAMIC_BANDS)
    img = ee.Image(imcoll.iterate(combine_bands_function))

    total_image_list: List[ee.Image] = [img]
    for static_image_function in STATIC_IMAGE_FUNCTIONS:
        total_image_list.append(static_image_function(region=polygon))

    return ee.Image.cat(total_image_list)


class EarthEngineExporter:
    """
    Export satellite data from Earth engine to Azure Blob Storage.
    """

    def __init__(self, check_ee: bool = False, check_azure: bool = False, dest_container: str = None) -> None:
        ee.Initialize()
        self.check_ee = check_ee
        self.ee_task_list = get_ee_task_list() if self.check_ee else []
        self.check_azure = check_azure
        self.azure_blob_list = get_azure_blob_list(container_name=dest_container) if self.check_azure and dest_container else []
        self.dest_container = dest_container

    def _export_for_polygon(
        self,
        polygon: "ee.Geometry.Polygon",
        polygon_identifier: Union[int, str],
        start_date: date,
        end_date: date,
        test: bool = False,
    ) -> bool:
        filename = f"tifs/{polygon_identifier}.tif"

        if filename in self.azure_blob_list:
            return True

        if not test and filename in self.ee_task_list:
            return True

        if len(self.ee_task_list) >= 3000:
            return False

        img = create_ee_image(polygon, start_date, end_date)

        if not test:
            filename = f"tifs/{filename}"

        try:
            task = ee.batch.Export.image.toCloudStorage(
                bucket=AZURE_CONTAINER_NAME,
                fileNamePrefix=filename,
                image=img.clip(polygon),
                description=ee_safe_str(filename),
                scale=10,
                region=polygon,
                maxPixels=1e13,
            )
            task.start()
            self.ee_task_list.append(filename)
        except ee.ee_exception.EEException as e:
            print(f"Task not started! Got exception {e}")

        return True

    def export_for_bbox(self, bbox: BBox, bbox_name: str, start_date: date, end_date: date):
        if start_date > end_date:
            raise ValueError(f"Start date {start_date} is after end date {end_date}")

        ee_bbox = EEBoundingBox.from_bounding_box(bounding_box=bbox, padding_metres=0)
        regions = [ee_bbox.to_ee_polygon()]
        ids = ["batch/0"]

        return_obj = {}
        for identifier, region in zip(ids, regions):
            return_obj[identifier] = self._export_for_polygon(
                polygon=region,
                polygon_identifier=f"{bbox_name}/{identifier}",
                start_date=start_date,
                end_date=end_date,
                test=True,
            )
        return return_obj

    def export_for_labels(self, labels: pd.DataFrame):
        for expected_column in [START, END, LAT, LON]:
            assert expected_column in labels

        labels[START] = pd.to_datetime(labels[START]).dt.date
        labels[END] = pd.to_datetime(labels[END]).dt.date

        print(f"Exporting {len(labels)} labels: ")
        for _, row in tqdm(labels.iterrows(), desc="Exporting", total=len(labels)):
            ee_bbox = EEBoundingBox.from_centre(
                mid_lat=row[LAT],
                mid_lon=row[LON],
                surrounding_metres=80,
            )

            self._export_for_polygon(
                polygon=ee_bbox.to_ee_polygon(),
                polygon_identifier=ee_bbox.get_identifier(row[START], row[END]),
                start_date=row[START],
                end_date=row[END],
                test=False,
            )


class EarthEngineAPI:
    """
    Fetch satellite data from Earth engine by URL.
    :param credentials: The credentials to use for the export. If not specified,
        the default credentials will be used
    """

    def __init__(self, credentials=None) -> None:
        ee.Initialize(
            credentials if credentials else get_ee_credentials(),
            opt_url="https://earthengine-highvolume.googleapis.com",
        )

    def get_ee_url(self, lat, lon, start_date, end_date):
        ee_bbox = EEBoundingBox.from_centre(
            mid_lat=lat,
            mid_lon=lon,
            surrounding_metres=80,
        ).to_ee_polygon()
        img = create_ee_image(ee_bbox, start_date, end_date)
        return img.getDownloadURL(
            {
                "region": ee_bbox,
                "scale": 10,
                "filePerBand": False,
                "format": "GEO_TIFF",
            }
        )

