import warnings
from argparse import ArgumentParser
import pandas as pd
import torch
import yaml
import numpy as np
from datasets import datasets, label_col
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
from tsai.models.TransformerModel import TransformerModel
from openmapflow.bands import BANDS_MAX
from openmapflow.constants import SUBSET
from openmapflow.pytorch_dataset import PyTorchDataset
from openmapflow.train_utils import (
    generate_model_name,
    model_path_from_name,
    upsample_multiclass,
)
from openmapflow.utils import tqdm
from sklearn.model_selection import train_test_split  # Added import

warnings.simplefilter("ignore", UserWarning)

# Argument Parsing
parser = ArgumentParser()
parser.add_argument("--model_name", type=str, default="")
parser.add_argument("--start_month", type=str, default="February")
parser.add_argument("--input_months", type=int, default=12)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--upsample_minority_ratio", type=float, default=0.5)
parser.add_argument("--lr", type=float, default=0.001)
parser.add_argument("--epochs", type=int, default=10)

args = parser.parse_args()
start_month = args.start_month
batch_size = args.batch_size
upsample_minority_ratio = args.upsample_minority_ratio
num_epochs = args.epochs
lr = args.lr
model_name = args.model_name
input_months = args.input_months

# Load Dataset
df = pd.read_csv('datasets/filterd_crops.csv')  # Load CSV directly
print(f"Unique classes in dataset: {df[label_col].unique()}")

# Split into train and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)  # Adjust test_size as needed

print(f"Unique classes in train_df before upsampling: {train_df[label_col].unique()}")
print(f"Shape of train_df before upsampling: {train_df.shape}")

train_df = upsample_multiclass(train_df, label_col=label_col, upsample_ratio=upsample_minority_ratio)
print(f"Unique classes in train_df after upsampling: {train_df[label_col].unique()}")
print(f"Shape of train_df after upsampling: {train_df.shape}")

train_csv_path = 'train_data.csv'
val_csv_path = 'val_data.csv'

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)

train_data = PyTorchDataset(train_csv_path)  # Use CSV paths
val_data = PyTorchDataset(val_csv_path)  # Use CSV paths

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Model Definition
num_bands, height, width = train_data[0][0].shape
num_features = num_bands * height * width
num_timesteps = 1  # If not timeseries, set to 1.

class Model(torch.nn.Module):
    def __init__(self, normalization_vals=BANDS_MAX):
        super().__init__()
        self.model = TransformerModel(c_in=num_features, c_out=len(train_df[label_col].unique()))  # Adjust c_in
        self.normalization_vals = torch.tensor(normalization_vals, dtype=torch.float32)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input.
        x = x / self.normalization_vals.to(x.device)
        x = x.unsqueeze(1)  # Add a timestep dimension.
        x = x.transpose(2, 1)
        x = self.model(x)
        return torch.softmax(x, dim=1)

# Training Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.CrossEntropyLoss()
if model_name == "":
    model_name = generate_model_name(val_df=val_df, start_month=start_month)

# Training Loop
lowest_validation_loss = None
metrics = {}
train_batches = 1 + len(train_data) // batch_size
val_batches = 1 + len(val_data) // batch_size

with tqdm(range(num_epochs), desc="Epoch") as tqdm_epoch:
    for epoch in tqdm_epoch:
        total_train_loss = 0.0
        model.train()
        for x in train_dataloader:
            inputs, labels = x[0].to(device), x[1].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * len(inputs)

        total_val_loss = 0.0
        y_true, y_pred = [], []
        model.eval()
        with torch.no_grad():
            for x in val_dataloader:
                inputs, labels = x[0].to(device), x[1].to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.long())
                total_val_loss += loss.item() * len(inputs)
                y_true += labels.tolist()
                y_pred += torch.argmax(outputs, dim=1).tolist()

        train_loss = total_train_loss / len(train_data)
        val_loss = total_val_loss / len(val_data)

        if lowest_validation_loss is None or val_loss < lowest_validation_loss:
            lowest_validation_loss = val_loss
            metrics = {
                "accuracy": accuracy_score(y_true, y_pred),
                "f1": f1_score(y_true, y_pred, average="weighted"),
                "precision": precision_score(y_true, y_pred, average="weighted"),
                "recall": recall_score(y_true, y_pred, average="weighted"),
            }
            metrics = {k: round(float(v), 4) for k, v in metrics.items()}
        tqdm_epoch.set_postfix(loss=val_loss)

        if lowest_validation_loss == val_loss:
            sm = torch.jit.script(model)
            model_path = model_path_from_name(model_name=model_name)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            sm.save(str(model_path))

print(f"MODEL_NAME={model_name}")
print(yaml.dump(metrics, allow_unicode=True, default_flow_style=False))