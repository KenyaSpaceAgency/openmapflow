import os
import pandas as pd
from google.cloud import storage
from openmapflow.labeled_dataset import LabeledDataset


# Define local dataset directory
LOCAL_DATASET_DIR = "/home/mapflow/datasets"

def download_csv_from_gcs(filename):
    """Load dataset from a local file instead of GCS."""
    local_path = os.path.join("/home/mapflow/Desktop/openmapflow/datasets", filename)  # Adjust this path as needed
    if os.path.exists(local_path):
        print(f"Using local dataset: {local_path}")
        return local_path
    else:
        raise FileNotFoundError(f"Dataset {filename} not found locally.")

# Define dataset classes inheriting from LabeledDataset
class GeowikiLandcover2017(LabeledDataset):
    def load_labels(self):
        """Load dataset from a local file."""
        csv_path = "/home/mapflow/Desktop/openmapflow/datasets/GeowikiLandcover2017.csv"  
        df = pd.read_csv(csv_path)
    
        print("Columns in the dataset:", df.columns)  # Debugging step
    
        if "class_prob" not in df.columns:
            raise KeyError("The dataset does not contain a 'class_prob' column.")
    
        df = df[df["class_prob"] != 0.5].copy()
        return df   
class TogoCrop2019(LabeledDataset):
    def load_labels(self):
        """Load dataset from a local file."""
        csv_path = "/home/mapflow/Desktop/openmapflow/datasets/TogoCrop2019.csv"
        df = pd.read_csv(csv_path)

        print("Columns in the dataset:", df.columns)  # Debugging step

        if "class_prob" not in df.columns:
            raise KeyError("The dataset does not contain a 'class_prob' column.")

        df = df[df["class_prob"] != 0.5].copy()
        return df

class KenyaCrop201819(LabeledDataset):
    def load_labels(self):
        return pd.read_csv(download_csv_from_gcs("filterd_crops.csv"))

# Ensure instances of the correct class
datasets = [GeowikiLandcover2017(), TogoCrop2019(), KenyaCrop201819()]

# Debugging: Check class types before calling create_datasets
for d in datasets:
    print(f"{d.__class__.__name__} is instance of LabeledDataset: {isinstance(d, LabeledDataset)}")
