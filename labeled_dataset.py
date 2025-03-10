import os
from pathlib import Path
import glob
import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm
from sklearn.utils import resample

def read_tif_file(file_path):
    """
    Read a TIF file and return its data as a flattened numpy array.
    """
    try:
        with rasterio.open(file_path) as src:
            return src.read().flatten()  # Return a flattened NumPy array
    except Exception as e:
        print(f"Error reading TIF file {file_path}: {e}")
        return None

def upsample_multiclass(df, label_col="eo_data", upsample_ratio=0.5):
    """
    Upsamples each class to a defined ratio of the majority class.

    Parameters:
        df (pd.DataFrame): Input dataframe
        label_col (str): Column containing class labels
        upsample_ratio (float): Ratio of the largest class to upsample smaller ones

    Returns:
        pd.DataFrame: Upsampled dataset
    """
    class_counts = df[label_col].value_counts()
    max_count = class_counts.max()

    upsampled_dfs = []
    for label, count in class_counts.items():
        class_df = df[df[label_col] == label]
        if count < max_count * upsample_ratio:
            class_df = resample(class_df, replace=True, n_samples=int(max_count * upsample_ratio), random_state=42)
        upsampled_dfs.append(class_df)

    return pd.concat(upsampled_dfs).sample(frac=1, random_state=42)  # Shuffle dataset

class LabeledDataset:
    """
    Base class for labeled datasets with Earth Observation (EO) data.
    """
    def __init__(self, tif_data_dir="/openmapflow/satdata"):
        self.name = self.__class__.__name__
        if self.name == "LabeledDataset":
            raise ValueError("LabeledDataset must be inherited to be used.")
        self.df_path = Path(f"datasets/{self.name}.csv")
        self.tif_data_dir = tif_data_dir

    def load_labels(self) -> pd.DataFrame:
        raise NotImplementedError("Subclasses must implement load_labels method")

    def create_dataset(self):
        if not self.df_path.exists():
            df = self.load_labels()
            df.to_csv(self.df_path, index=False)
        else:
            df = pd.read_csv(self.df_path)

        df = self._fetch_eo_data(df)

        # Upsample EO data classes
        df = upsample_multiclass(df, label_col="eo_data", upsample_ratio=0.5)

        df.to_csv(self.df_path, index=False)
        return self._generate_dataset_summary(df)

    def _fetch_eo_data(self, df: pd.DataFrame) -> pd.DataFrame:
        eo_data_col = "eo_data"
        if eo_data_col not in df.columns:
            df[eo_data_col] = None
        mask = df[eo_data_col].isnull()

        for idx, row in tqdm(df[mask].iterrows(), total=mask.sum(), desc="Reading TIF files"):
            file_path = row.get("EO_FILE")
            if not file_path:
                lat, lon = row.get("lat"), row.get("lon")
                if pd.notna(lat) and pd.notna(lon):
                    pattern = f"{self.tif_data_dir}/*_{lat}_{lon}*.tif"
                    matching_files = glob.glob(pattern)
                    if matching_files:
                        file_path = matching_files[0]
                        df.loc[idx, "EO_FILE"] = file_path
                    else:
                        continue
                else:
                    continue

            if not os.path.exists(file_path):
                alternative_path = os.path.join(self.tif_data_dir, os.path.basename(file_path))
                if os.path.exists(alternative_path):
                    file_path = alternative_path
                else:
                    continue

            tif_data = read_tif_file(file_path)
            if tif_data is not None:
                df.loc[idx, eo_data_col] = tif_data  # Store as a NumPy array
                df.loc[idx, "eo_status"] = "complete"
            else:
                df.loc[idx, "eo_status"] = "failed"

        return df

    def _generate_dataset_summary(self, df: pd.DataFrame) -> str:
        summary = f"Dataset: {self.name}\nTotal entries: {len(df)}\n"
        if "eo_data" in df.columns:
            summary += f"Entries with EO data: {df['eo_data'].notnull().sum()}\n"
        if "eo_status" in df.columns:
            summary += f"Status distribution:\n{df['eo_status'].value_counts()}\n"
        return summary
