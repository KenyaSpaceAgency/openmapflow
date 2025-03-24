import os
import re
from collections import defaultdict
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple

import requests

# For Azure Storage
from azure.storage.blob import BlobServiceClient
from azure.core.exceptions import ResourceNotFoundError

# For Local Storage
import shutil

from openmapflow.bbox import BBox
from openmapflow.config import BucketNames as bn
from openmapflow.utils import tqdm


# Assuming you have these configurations defined somewhere
# For Azure
AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=https;EndpointSuffix=core.windows.net;AccountName=openmapflow;AccountKey=gBh30r5wqeU2HMhfG5jTmG0Ags++3rsYe1wTotQoxNK/EVnCnBCOt7ytHQrJuBya9/qMT/63xE3k+ASth7eOBQ==;BlobEndpoint=https://openmapflow.blob.core.windows.net/;FileEndpoint=https://openmapflow.file.core.windows.net/;QueueEndpoint=https://openmapflow.queue.core.windows.net/;TableEndpoint=https://openmapflow.table.core.windows.net/"
AZURE_CONTAINER_INFERENCE_EO = "inference-eo-container"
AZURE_CONTAINER_PREDS = "preds-container"

# For Local
LOCAL_INFERENCE_EO_PATH = "local_inference_eo"
LOCAL_PREDS_PATH = "local_preds"

def get_available_models(models_url: str) -> List[str]:
    response = requests.get(models_url)
    if response.status_code == 403:
        print(
            f"""
Cloud Run access denied. Please ensure proper permissions are set.
"""
        )
        return []
    return [item["modelName"] for item in response.json()["models"]]

def get_available_bboxes(
    containers_to_check: List[str] = [AZURE_CONTAINER_INFERENCE_EO],
    use_azure: bool = True,
    use_local: bool = False,
) -> List[BBox]:
    """
    Get all available bboxes from the given containers (Azure) or directories (Local) using regex.
    Args:
        containers_to_check: List of Azure containers or local directories to check.
        use_azure: Use Azure Storage.
        use_local: Use Local Storage.
    Returns:
        List of BBoxes.
    """
    if not use_azure and not use_local:
        raise ValueError("At least one of use_azure or use_local must be True.")

    previous_matches = []
    available_bboxes = []
    bbox_regex = (
        r".*min_lat=-?\d*\.?\d*_min_lon=-?\d*\.?\d*_max_lat=-?\d*\.?\d*_max_lon=-?\d*\.?\d*_"
        + r"dates=\d{4}-\d{2}-\d{2}_\d{4}-\d{2}-\d{2}.*?\/"
    )

    if use_azure:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        for container_name in containers_to_check:
            container_client = blob_service_client.get_container_client(container_name)
            for blob in container_client.list_blobs():
                match = re.search(bbox_regex, blob.name)
                if not match:
                    continue
                p = match.group()
                if p not in previous_matches:
                    previous_matches.append(p)
                    available_bboxes.append(BBox.from_str(f"az://{container_name}/{p}"))

    if use_local:
        for local_dir in [LOCAL_INFERENCE_EO_PATH if c == AZURE_CONTAINER_INFERENCE_EO else LOCAL_PREDS_PATH for c in containers_to_check]:
            for root, _, files in os.walk(local_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    relative_path = os.path.relpath(full_path, local_dir)
                    match = re.search(bbox_regex, relative_path)
                    if not match:
                        continue
                    p = match.group()
                    if p not in previous_matches:
                        previous_matches.append(p)
                        available_bboxes.append(BBox.from_str(f"file://{os.path.join(local_dir, p)}"))

    return available_bboxes


def get_file_amount(container_name: str, prefix: str, use_azure: bool = True, use_local: bool = False) -> int:
    if use_azure:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        return len(list(blobs))
    elif use_local:
        local_dir = LOCAL_INFERENCE_EO_PATH if container_name == AZURE_CONTAINER_INFERENCE_EO else LOCAL_PREDS_PATH
        return len(glob(os.path.join(local_dir, prefix, "*")))
    else:
        raise ValueError("At least one of use_azure or use_local must be True.")

def get_file_dict_and_amount(
    container_name: str, prefix: str, use_azure: bool = True, use_local: bool = False
) -> Tuple[Dict[str, List[str]], int]:
    files_dict = defaultdict(lambda: [])
    amount = 0

    if use_azure:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        for blob in tqdm(blobs, desc=f"From {container_name}"):
            p = Path(blob.name)
            files_dict[str(p.parent)].append(p.stem.replace("pred_", ""))
            amount += 1
    elif use_local:
        local_dir = LOCAL_INFERENCE_EO_PATH if container_name == AZURE_CONTAINER_INFERENCE_EO else LOCAL_PREDS_PATH
        for root, _, files in os.walk(os.path.join(local_dir, prefix)):
            for file in files:
                p = Path(os.path.join(root, file))
                relative_path = os.path.relpath(p, os.path.join(local_dir, prefix))
                parent = Path(relative_path).parent
                files_dict[str(parent)].append(Path(file).stem.replace("pred_", ""))
                amount += 1
    else:
        raise ValueError("At least one of use_azure or use_local must be True.")

    return files_dict, amount

def print_between_lines(text: str, line: str = "-", is_tabbed: bool = False):
    tab = "\t" if is_tabbed else ""
    print(tab + (line * len(text)))
    print(tab + text)
    print(tab + (line * len(text)))

def get_status(prefix: str, use_azure: bool = True, use_local: bool = False) -> Tuple[int, int, int]:
    print_between_lines(prefix)
    #ee_task_amount = get_ee_task_amount(prefix=prefix) #Removed EE dependency
    ee_task_amount = 0 #Place holder, remove when EE dependency is added back
    tifs_amount = get_file_amount(AZURE_CONTAINER_INFERENCE_EO, prefix, use_azure, use_local) if use_azure or use_local else 0
    predictions_amount = get_file_amount(AZURE_CONTAINER_PREDS, prefix, use_azure, use_local) if use_azure or use_local else 0
    print(f"1) Obtaining input data: {ee_task_amount}")
    print(f"2) Input data available: {tifs_amount}")
    print(f"3) Predictions made: {predictions_amount}")
    return ee_task_amount, tifs_amount, predictions_amount

def find_missing_predictions(
    prefix: str, verbose: bool = False, use_azure: bool = True, use_local: bool = False
) -> Dict[str, List[str]]:
    print("Addressing missing files")
    tif_files, tif_amount = get_file_dict_and_amount(AZURE_CONTAINER_INFERENCE_EO, prefix, use_azure, use_local)
    pred_files, pred_amount = get_file_dict_and_amount(AZURE_CONTAINER_PREDS, prefix, use_azure, use_local)
    missing = {}
    for full_k in tqdm(tif_files.keys(), desc="Missing files"):
        if full_k not in pred_files:
            diffs =tif_files[full_k]
        else:
            diffs = list(set(tif_files[full_k]) - set(pred_files[full_k]))
        if len(diffs) > 0:
            missing[full_k] = diffs

    batches_with_issues = len(missing.keys())
    if verbose:
        print_between_lines(prefix)

    if batches_with_issues == 0:
        print("All files in each batch match")
        return missing

    print(
        f"{batches_with_issues}/{len(tif_files.keys())} "
        + f"batches have a total {tif_amount - pred_amount} missing predictions"
    )

    if verbose:
        for batch, files in missing.items():
            print_between_lines(
                text=f"\t{Path(batch).stem}: {len(files)}", is_tabbed=True
            )
            for f in files:
                print(f"\t{f}")

    return missing

def make_new_predictions(
    missing: Dict[str, List[str]], container_name: str = AZURE_CONTAINER_INFERENCE_EO, use_azure: bool = True, use_local: bool = False
):
    if use_azure:
        blob_service_client = BlobServiceClient.from_connection_string(AZURE_STORAGE_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(container_name)
        for batch, files in tqdm(missing.items(), desc="Going through batches"):
            for file in tqdm(files, desc="Renaming files", leave=False):
                blob_name = f"{batch}/{file}.tif"
                try:
                    blob_client = container_client.get_blob_client(blob_name)
                    blob_properties = blob_client.get_blob_properties()
                    new_blob_name = f"{batch}/{file}-retry.tif"
                    blob_client.rename_blob(new_blob_name)
                except ResourceNotFoundError:
                    print(f"Could not find: {blob_name}")
    elif use_local:
        local_dir = LOCAL_INFERENCE_EO_PATH if container_name == AZURE_CONTAINER_INFERENCE_EO else LOCAL_PREDS_PATH
        for batch, files in tqdm(missing.items(), desc="Going through batches"):
            for file in tqdm(files, desc="Renaming files", leave=False):
                original_file_path = os.path.join(local_dir, batch, f"{file}.tif")
                retry_file_path = os.path.join(local_dir, batch, f"{file}-retry.tif")
                if os.path.exists(original_file_path):
                    os.rename(original_file_path, retry_file_path)
                else:
                    print(f"Could not find: {original_file_path}")
    else:
        raise ValueError("At least one of use_azure or use_local must be True.")

def gdal_cmd(cmd_type: str, in_file: str, out_file: str, msg=None, print_cmd=False):
    """
    Runs a GDAL command: gdalbuildvrt or gdal_translate.
    """
    if cmd_type == "gdalbuildvrt":
        cmd = f"gdalbuildvrt {out_file} {in_file}"
    elif cmd_type == "gdal_translate":
        cmd = f"gdal_translate -a_srs EPSG:4326 -of GTiff {in_file} {out_file}"
    else:
        raise NotImplementedError(f"{cmd_type} not implemented.")
    if msg:
        print(msg)
    if print_cmd:
        print(cmd)
    os.system(cmd)

def build_vrt(prefix):
    """
    Builds a VRT file for each batch and then creates one VRT file for all batches.
    """
    print("Building vrt for each batch")
    for d in tqdm(glob(f"{prefix}_preds/*/*/")):
        if "batch" not in d:
            continue
        match = re.search("batch_(.*?)/", d)
        if match:
            i = int(match.group(1))
        else:
            raise ValueError(f"Cannot parse i from {d}")
        vrt_file = Path(f"{prefix}_vrts/{i}.vrt")
        if not vrt_file.exists():
            gdal_cmd(cmd_type="gdalbuildvrt", in_file=f"{d}*", out_file=str(vrt_file))

    print(f"Prefix for final VRT: {prefix}") #added print statement.
    gdal_cmd(
        cmd_type="gdalbuildvrt",
        in_file=f"{prefix}_vrts/*.vrt",
        out_file=f"{prefix}_final.vrt",
        msg="Building full vrt",
    )