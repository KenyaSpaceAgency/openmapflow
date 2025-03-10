import rasterio
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class PyTorchDataset(Dataset):
    def __init__(self, csv_file, patch_size=128, transform=None):
        self.df = pd.read_csv(csv_file)
        self.patch_size = patch_size
        self.transform = transform
        
        # Create a mapping from original labels to consecutive indices
        unique_labels = sorted(self.df['eo_data'].unique())
        self.label_map = {label: idx for idx, label in enumerate(unique_labels)}
        self.num_classes = len(unique_labels)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = row['EO_FILE']
        label = row['eo_data']
        mapped_label = self.label_map[label]
        
        try:
            with rasterio.open(file_path) as src:
                x_data = src.read()
                
            x_data = np.array(x_data)
            
            # Handle 5D data (time series)
            if x_data.ndim == 4:
                # Combine time and bands dimensions for compatibility with 2D convolutions
                time_steps, bands, height, width = x_data.shape
                # Reshape to [bands*time_steps, height, width] to work with 2D CNN
                x_data = x_data.reshape(-1, height, width)
            
            # Randomly extract a patch
            _, height, width = x_data.shape
            x = np.random.randint(0, width - self.patch_size)
            y = np.random.randint(0, height - self.patch_size)
            x_data = x_data[:, y:y + self.patch_size, x:x + self.patch_size]
            
            # Convert to tensor
            x_data = torch.tensor(x_data, dtype=torch.float32)
            
            # Ensure proper shape for 2D CNN: [batch, channels, height, width]
            if len(x_data.shape) == 3:
                # We already have [channels, height, width], no need to unsqueeze
                pass
                
            y_data = torch.tensor(mapped_label, dtype=torch.long)
            
            if self.transform:
                x_data = self.transform(x_data)
                
            return x_data, y_data
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            # Return an empty tensor with correct shape instead of None
            return torch.zeros((1, self.patch_size, self.patch_size)), torch.tensor(0, dtype=torch.long)