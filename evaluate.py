"""
Evaluate a trained model using a test dataset.
"""

from argparse import ArgumentParser
import pandas as pd
import torch
import yaml
from datasets import label_col
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from openmapflow.config import PROJECT_ROOT, DataPaths
from openmapflow.constants import SUBSET
from openmapflow.pytorch_dataset import PyTorchDataset
from openmapflow.train_utils import model_path_from_name
from openmapflow.utils import tqdm
from sklearn.model_selection import train_test_split
import torch.nn.functional as F  # Import softmax

def evaluate_model(model_name, start_month="February", input_months=12, batch_size=64, skip_yaml=False):
    """
    Evaluates a trained model.

    Args:
        model_name (str): Name of the model to evaluate.
        start_month (str): Starting month for data processing.
        input_months (int): Number of input months to use.
        batch_size (int): Batch size for data loading.
        skip_yaml (bool): If True, skips saving metrics to YAML.
    """

    model_path = model_path_from_name(model_name=model_name)

    # Load and prepare the dataset
    df = pd.read_csv('datasets/filterd_crops.csv')
    df = df[df['EO_FILE'].notna()]
    df[label_col] = df['eo_data']
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    print(f"Test DataFrame shape: {test_df.shape}")
    print(f"Test DataFrame head: {test_df.head()}")

    # Create test CSV and PyTorch DataLoader
    test_csv_path = 'test_data.csv'
    test_df.to_csv(test_csv_path, index=False)
    test_data = PyTorchDataset(csv_file=test_csv_path)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Load and prepare the model
    model_pt = torch.jit.load(model_path)
    print(model_pt)
    model_pt.eval()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Evaluate the model
    test_batches = 1 + len(test_data) // batch_size
    y_true, y_score, y_pred = [], [], []
    with torch.no_grad():
        for x in tqdm(test_dataloader, total=test_batches, desc="Testing", leave=False):
            inputs, labels = x[0].to(device), x[1].to(device)
            outputs = model_pt(inputs)
            probabilities = F.softmax(outputs, dim=1)  # Apply softmax
            y_true += labels.tolist()
            y_score += probabilities.tolist()
            y_pred += torch.argmax(outputs, dim=1).tolist()
            print(f"probabilities[0:5]: {probabilities[0:5]}") #debugging

    print(f"Unique y_true classes: {len(set(y_true))}") #debugging
    print(f"y_score shape: {len(y_score[0]) if y_score else 'Empty'}") #debugging
    print(f"y_true[0:5]: {y_true[0:5]}") #debugging

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted"),
        "precision": precision_score(y_true, y_pred, average="weighted"),
        "recall": recall_score(y_true, y_pred, average="weighted"),
        "roc_auc": roc_auc_score(y_true, y_score, multi_class='ovr'),
    }
    metrics = {k: round(float(v), 4) for k, v in metrics.items()}

    # Prepare and save metrics
    all_metrics = {}
    if not skip_yaml and (PROJECT_ROOT / DataPaths.METRICS).exists():
        with (PROJECT_ROOT / DataPaths.METRICS).open() as f:
            all_metrics = yaml.safe_load(f)

    all_metrics[model_name] = {
        "test_metrics": metrics,
        "test_size": len(test_df),
        label_col: test_df[label_col].value_counts(normalize=True).to_dict(),
    }
    print(yaml.dump(all_metrics[model_name], allow_unicode=True, default_flow_style=False))

    if not skip_yaml:
        with open((PROJECT_ROOT / DataPaths.METRICS), "w") as f:
            yaml.dump(all_metrics, f)

if __name__ == "__main__":
    parser = ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to evaluate.")
    parser.add_argument("--start_month", type=str, default="February", help="Starting month for data processing.")
    parser.add_argument("--input_months", type=int, default=12, help="Number of input months to use.")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for data loading.")
    parser.add_argument("--skip_yaml", action="store_true", help="If set, skips saving metrics to YAML.")

    args = parser.parse_args()
    evaluate_model(
        model_name=args.model_name,
        start_month=args.start_month,
        input_months=args.input_months,
        batch_size=args.batch_size,
        skip_yaml=args.skip_yaml,
    )

    