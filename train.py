import warnings
from argparse import ArgumentParser
import pandas as pd
import torch
import yaml
import numpy as np
from datasets import datasets, label_col
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader
import torchvision.models as models  # Import ResNet
import torch.nn as nn
from openmapflow.bands import BANDS_MAX
from openmapflow.constants import SUBSET
from openmapflow.pytorch_dataset import PyTorchDataset
from openmapflow.train_utils import (
    generate_model_name,
    model_path_from_name,
    upsample_multiclass,
)
from openmapflow.utils import tqdm
from sklearn.model_selection import train_test_split
import rasterio # Add this line

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
parser.add_argument("--patch_size", type=int, default=128)  # add patch size argument

args = parser.parse_args()
start_month = args.start_month
batch_size = args.batch_size
upsample_minority_ratio = args.upsample_minority_ratio
num_epochs = args.epochs
lr = args.lr
model_name = args.model_name
input_months = args.input_months
patch_size = args.patch_size  # get patch size from argument.

# Load Dataset
df = pd.read_csv("datasets/filterd_crops.csv")
df = df[df["EO_FILE"].notna()]  # Remove nan values.
print(f"Unique classes in dataset: {df[label_col].unique()}")

# Split into train and validation
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

print(f"Unique classes in train_df before upsampling: {train_df[label_col].unique()}")
print(f"Shape of train_df before upsampling: {train_df.shape}")

train_df = upsample_multiclass(
    train_df, label_col=label_col, upsample_ratio=upsample_minority_ratio
)
print(f"Unique classes in train_df after upsampling: {train_df[label_col].unique()}")
print(f"Shape of train_df after upsampling: {train_df.shape}")

train_csv_path = "train_data.csv"
val_csv_path = "val_data.csv"

train_df.to_csv(train_csv_path, index=False)
val_df.to_csv(val_csv_path, index=False)

train_data = PyTorchDataset(
    train_csv_path, patch_size=patch_size
)  # Use CSV paths, pass patch size.
val_data = PyTorchDataset(
    val_csv_path, patch_size=patch_size
)  # Use CSV paths, pass patch size.

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

# Model Definition (CNN)
class CNNModel(nn.Module):
    def __init__(self, num_classes, num_bands=4): # Add num_bands parameter
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)
        # Modify the first convolutional layer to accept num_bands channels
        self.resnet.conv1 = nn.Conv2d(num_bands, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet(x)

# Training Setup
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Get number of bands from first image
with rasterio.open(train_df.iloc[0]['EO_FILE']) as src:
    num_bands = src.count

# Initialize the model with the correct number of bands
model = CNNModel(num_classes=len(train_df[label_col].unique()), num_bands=num_bands).to(device)

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
            print(f"Input shape: {inputs.shape}")  # Add this line
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