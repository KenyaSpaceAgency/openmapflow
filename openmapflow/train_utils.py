from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.utils import resample

from openmapflow.config import PROJECT, PROJECT_ROOT, DataPaths
from openmapflow.constants import CLASS_PROB, COUNTRY, EO_DATA, MONTHS, START
from openmapflow.utils import to_date, tqdm

def generate_model_name(val_df: pd.DataFrame, start_month: Optional[str] = None) -> str:
    """Generate a model name based on the validation data."""
    model_name = ""
    try:
        model_name += val_df[COUNTRY].iloc[0] + "_"
    except KeyError:
        pass

    model_name += PROJECT.replace("-example", "")
    model_name += f"_{to_date(val_df[START].iloc[0]).year}"
    if start_month:
        model_name += f"_{start_month}"
    return model_name

def model_path_from_name(model_name: str) -> Path:
    """Get the path to a model from its name."""
    return PROJECT_ROOT / DataPaths.MODELS / f"{model_name}.pt"

def upsample_multiclass(df: pd.DataFrame, label_col: str, upsample_ratio: float = 0.5) -> pd.DataFrame:
    """
    Upsamples minority classes in a multi-class dataset.
    """
    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()
    
    upsampled_dfs = []
    for label, count in class_counts.items():
        class_df = df[df[label_col] == label]
        if count < max_count * upsample_ratio:
            class_df = resample(
                class_df,
                replace=True,
                n_samples=int(max_count * upsample_ratio),
                random_state=42
            )
        upsampled_dfs.append(class_df)
    
    return pd.concat(upsampled_dfs).sample(frac=1, random_state=42)

def get_x_y(
    df: pd.DataFrame,
    label_col: str = CLASS_PROB,
) -> Tuple[List[np.ndarray], List[float]]:
    """Get the X and y data from a dataframe with single-value EO data."""

    def process_eo_data(x):
        try:
            eo_value = float(x)  # Convert to float
            return np.array([eo_value])  # Wrap as numpy array
        except ValueError:
            print(f"Warning: Invalid EO_DATA found. Skipping entry: {x}")
            return None

    tqdm.pandas()
    x_data = df[EO_DATA].astype(str).progress_apply(process_eo_data).dropna().to_list()
    y_data = df[label_col].dropna().to_list()

    return x_data, y_data
