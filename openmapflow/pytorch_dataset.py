import rasterio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PyTorchDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['EO_FILE']
        label = row['eo_data']

        try:
            with rasterio.open(file_path) as src:
                x_data = src.read()

            x_data = np.array(x_data)

            if x_data.ndim == 4:
                x_data = np.transpose(x_data, (3, 0, 1, 2))  # time, bands, height, width
                x_data = x_data.reshape(x_data.shape[0], x_data.shape[1], x_data.shape[2] * x_data.shape[3]) # time, bands, height*width

            x_data = torch.tensor(x_data, dtype=torch.float32)
            y_data = torch.tensor(label, dtype=torch.long)

            if self.transform:
                x_data = self.transform(x_data)

            return x_data, y_data

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None, None