import os
import pandas as pd
from openmapflow.labeled_dataset import LabeledDataset

# Define relative dataset directory
DATASET_DIR = "datasets"  # Assuming datasets are in a 'datasets' subfolder

def load_local_csv(filename):
    """Load a CSV dataset from a local file."""
    local_path = os.path.join(DATASET_DIR, filename)
    if os.path.exists(local_path):
        print(f"Using local dataset: {local_path}")
        return pd.read_csv(local_path)
    else:
        raise FileNotFoundError(f"Dataset {filename} not found locally.")

class LabeledDatasetWithProbFilter(LabeledDataset):
    """Base class with common class_prob filtering."""
    def load_labels(self):
        df = load_local_csv(self.filename) #self.filename must be set in the child classes.
        if "class_prob" not in df.columns:
            raise KeyError(f"The dataset {self.filename} does not contain a 'class_prob' column.")
        df = df[df["class_prob"] != 0.5].copy()
        return df

class GeowikiLandcover2017(LabeledDatasetWithProbFilter):
    filename = "GeowikiLandcover2017.csv"

class TogoCrop2019(LabeledDatasetWithProbFilter):
    filename = "TogoCrop2019.csv"

class KenyaCrop201819(LabeledDataset):
    def load_labels(self):
        return load_local_csv("filterd_crops.csv")

# Ensure instances of the correct class
datasets = [GeowikiLandcover2017(), TogoCrop2019(), KenyaCrop201819()]

# Debugging: Check class types before calling create_datasets
for d in datasets:
    print(f"{d.__class__.__name__} is instance of LabeledDataset: {isinstance(d, LabeledDataset)}")