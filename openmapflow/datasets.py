import pandas as pd
<<<<<<< HEAD

from openmapflow.constants import CLASS_PROB
from openmapflow.labeled_dataset import LabeledDataset

gcloud_url = "https://storage.googleapis.com/openmapflow-public/datasets"

label_col = CLASS_PROB


class GeowikiLandcover2017(LabeledDataset):
    def load_labels(self):
        # Source: https://github.com/nasaharvest/crop-mask/blob/master/datasets.py
        df = pd.read_csv(f"{gcloud_url}/crop/GeowikiLandcover2017.csv")
        df = df[df[label_col] != 0.5].copy()
=======
from google.cloud import storage

def download_csv_from_gcs(filename):
    """Download a file from local storage or Google Cloud Storage (GCS)."""

    local_path = os.path.join("/home/mapflow/Desktop/openmapflow/datasets", filename)

    # Check if the file exists locally first
    if os.path.exists(local_path):
        print(f"Using local dataset: {local_path}")
        return local_path

    # If not found locally, attempt to download from GCS
    print(f"Downloading {filename} from Google Cloud Storage...")
    try:
        setup_gcp_credentials()  # Ensure GCP credentials are set up before accessing GCS
        client = storage.Client()
        bucket = client.bucket("your-gcs-bucket-name")  # Replace with actual bucket name
        blob = bucket.blob(f"dataset-path/{filename}")  # Adjust the path if necessary
        
        temp_path = f"/tmp/{filename}"  # Temporary download location
        blob.download_to_filename(temp_path)

        print(f"Downloaded {filename} to {temp_path}")
        return temp_path
    except Exception as e:
        print(f"Error downloading {filename} from GCS: {e}")
        raise FileNotFoundError(f"Dataset {filename} not found locally or in GCS.")

# Define dataset classes inheriting from LabeledDataset
class GeowikiLandcover2017(LabeledDataset):
    def load_labels(self):
        """Load dataset from local storage or Google Cloud Storage."""
        csv_path = download_csv_from_gcs("filtered_crops.csv")
        df = pd.read_csv(csv_path)

        # Ensure the column exists before filtering
        if "Name" not in df.columns:
            raise KeyError(f"'Name' column not found in {csv_path}. Available columns: {df.columns}")

        df = df[df["Name"] != 3000].copy()
>>>>>>> b3778400 (data merging)
        return df


class TogoCrop2019(LabeledDataset):
    def load_labels(self):
<<<<<<< HEAD
        # Source: https://github.com/nasaharvest/crop-mask/blob/master/datasets.py
        df = pd.read_csv(f"{gcloud_url}/crop/TogoCrop2019.csv")
        df = df[df[label_col] != 0.5].copy()
=======
        csv_path = download_csv_from_gcs("filtered_crops.csv")
        df = pd.read_csv(csv_path)

        if "Name" not in df.columns:
            raise KeyError(f"'Name' column not found in {csv_path}. Available columns: {df.columns}")

        df = df[df["Name"] != 1000].copy()
>>>>>>> b3778400 (data merging)
        return df


class KenyaCrop201819(LabeledDataset):
    def load_labels(self):
        # Source: https://github.com/nasaharvest/crop-mask/blob/master/datasets.py
        return pd.read_csv(f"{gcloud_url}/crop/Kenya_2018_2019.csv")


datasets = [GeowikiLandcover2017(), TogoCrop2019()]
