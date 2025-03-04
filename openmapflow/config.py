import os
from collections.abc import Mapping
from pathlib import Path
from typing import Dict

import yaml

from openmapflow.constants import (
    CONFIG_FILE,
    DATA_DIR,
    DEFAULT_CONFIG_PATH,
    LIBRARY_DIR,
    VERSION,
)


def update_dict(d: Dict, u: Mapping) -> Dict:
    """
    Update a dictionary with another dictionary.
    Source: https://stackoverflow.com/a/3233356/8702341
    """
    for k, v in u.items():
        if isinstance(v, Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def load_custom_config(path: Path) -> Dict:
    if path.exists():
        with path.open() as f:
            return yaml.safe_load(f)
    print(f"{path.name} not found in: {path.parent}\n")
    print(f"Using folder as project name: {path.parent.name}")
    return {"project": path.parent.name}


def load_default_config(project_name: str) -> Dict:
    with DEFAULT_CONFIG_PATH.open() as f:
        content = f.read().replace("<PROJECT>", project_name)
        return yaml.safe_load(content)


cwd = Path.cwd()
PROJECT_ROOT: Path = cwd.parent if (cwd.parent / CONFIG_FILE).exists() else cwd
CUSTOM_CONFIG = load_custom_config(PROJECT_ROOT / CONFIG_FILE)
PROJECT = CUSTOM_CONFIG["project"]
DEFAULT_CONFIG = load_default_config(PROJECT)
CONFIG_YML = update_dict(DEFAULT_CONFIG, CUSTOM_CONFIG)

# Azure Blob Storage Configuration
STORAGE_ACCOUNT_CONNECTION_STRING = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "DefaultEndpointsProtocol=https;AccountName=openmapflow;AccountKey=gBh30r5wqeU2HMhfG5jTmG0Ags++3rsYe1wTotQoxNK/EVnCnBCOt7ytHQrJuBya9/qMT/63xE3k+ASth7eOBQ==;EndpointSuffix=core.windows.net")
# You can define default container names here or in your config.yml and load them similarly to GCS buckets
LABELED_EO_CONTAINER_NAME = CONFIG_YML.get("azure", {}).get("labeled_eo_container", "openmap")
INFERENCE_EO_CONTAINER_NAME = CONFIG_YML.get("azure", {}).get("inference_eo_container", "inference-eo-container")
PREDS_CONTAINER_NAME = CONFIG_YML.get("azure", {}).get("preds_container", "preds-container")
PREDS_MERGED_CONTAINER_NAME = CONFIG_YML.get("azure", {}).get("preds_merged_container", "preds-merged-container")


class DataPaths:
    RAW_LABELS = DATA_DIR + CONFIG_YML["data_paths"]["raw_labels"]
    DATASETS = DATA_DIR + CONFIG_YML["data_paths"]["datasets"]
    MODELS = DATA_DIR + CONFIG_YML["data_paths"]["models"]
    METRICS = DATA_DIR + CONFIG_YML["data_paths"]["metrics"]
    REPORT = DATA_DIR + CONFIG_YML["data_paths"]["report"]

    @classmethod
    def get(cls, key: str = "") -> str:
        if key in cls.__dict__:
            return cls.__dict__[key]
        dp_list = [
            f"{k}: {v}"
            for k, v in vars(cls).items()
            if not k.startswith("__") and k != "get"
        ]
        return "\n".join(dp_list)


class BucketNames:
    # Azure Blob Storage Container Names
    LABELED_EO = LABELED_EO_CONTAINER_NAME
    INFERENCE_EO = INFERENCE_EO_CONTAINER_NAME
    PREDS = PREDS_CONTAINER_NAME
    PREDS_MERGED = PREDS_MERGED_CONTAINER_NAME
    STORAGE_ACCOUNT_CONNECTION_STRING = STORAGE_ACCOUNT_CONNECTION_STRING # Add connection string to BucketNames for easier access


def get_model_names_as_str() -> str:
    """Get the names of all models as a string."""
    models = [Path(p).stem for p in Path(PROJECT_ROOT / DataPaths.MODELS).glob("*.pt")]
    return " ".join(models)


def deploy_env_variables(empty_check: bool = True) -> str:
    prefix = "OPENMAPFLOW"
    deploy_env_dict = {
        "PROJECT": PROJECT,
        "MODELS_DIR": DataPaths.MODELS,
        "LIBRARY_DIR": LIBRARY_DIR,
        # Azure Blob Storage related environment variables
        "AZURE_STORAGE_CONNECTION_STRING": BucketNames.STORAGE_ACCOUNT_CONNECTION_STRING,
        "AZURE_LABELED_EO_CONTAINER": BucketNames.LABELED_EO,
        "AZURE_INFERENCE_EO_CONTAINER": BucketNames.INFERENCE_EO,
        "AZURE_PREDS_CONTAINER": BucketNames.PREDS,
        "AZURE_PREDS_MERGED_CONTAINER": BucketNames.PREDS_MERGED,
        "VERSION": VERSION,
    }
    if empty_check:
        for k, v in deploy_env_dict.items():
            if v == "" or v is None:
                raise ValueError(f"{k} is not set")

    env_variables = " ".join([f"{prefix}_{k}={v}" for k, v in deploy_env_dict.items()])
    return env_variables