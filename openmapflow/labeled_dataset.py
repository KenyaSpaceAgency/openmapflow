import os
from pathlib import Path
import argparse
import dataclasses
from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

def get_local_file_path(base_folder, file_name):
    """Gets the local file path for a file in a given folder."""
    file_path = Path(base_folder) / file_name
    if file_path.exists():
        return str(file_path)  # Return the path as a string
    else:
        return None  # Or raise an exception if you prefer

class LabeledDataset:
    """
    Base class for creating labeled datasets for machine learning with earth observation data.
    """
    
    def __init__(self):
        """
        Initialize the dataset with basic properties.
        """
        self.name = self.__class__.__name__
        if self.name == "LabeledDataset":
            raise ValueError("LabeledDataset must be inherited to be used.")
        
        # Path to the CSV file for this dataset
        self.df_path = Path(f"datasets/{self.name}.csv")

    def load_labels(self) -> pd.DataFrame:
        """
        Load and prepare labels for the dataset.
        
        This method MUST be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement load_labels method")

    def create_dataset(
        self, 
        use_ee_api: bool = False, 
        interactive: bool = True, 
        num_partitions: int = 4
    ) -> str:
        """
        Create or update the dataset with earth observation data.
        """
        # Load or create labels
        if not self.df_path.exists():
            df = self.load_labels()
            df = self._validate_dataframe(df)
            df.to_csv(self.df_path, index=False)
        else:
            df = pd.read_csv(self.df_path)

        # Fetch earth observation data
        df = self._fetch_eo_data(
            df, 
            use_ee_api=use_ee_api, 
            interactive=interactive, 
            num_partitions=num_partitions
        )

        # Save updated dataset
        df.to_csv(self.df_path, index=False)
        return self._generate_dataset_summary(df)

    def _validate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and standardize input DataFrame.
        """
        # Comprehensive column mapping and detection
        column_mappings = [
            {
                'status': ['eo_status', 'status', 'EO_STATUS'],
                'data': ['eo_data', 'data', 'EO_DATA'],
                'lat': ['eo_lat', 'lat', 'latitude', 'EO_LAT', 'LAT'],
                'lon': ['eo_lon', 'lon', 'longitude', 'EO_LON', 'LON'],
                'file': ['eo_file', 'file', 'EO_FILE'],
                'start': ['start_date', 'start', 'START'],
                'end': ['end_date', 'end', 'END']
            }
        ]

        # Find first matching column
        def find_column(df: pd.DataFrame, possible_names: List[str]) -> Optional[str]:
            for name in possible_names:
                if name in df.columns:
                    return name
            return None

        # Normalized column names
        normalized_columns = {}
        for mapping in column_mappings:
            for key, possible_names in mapping.items():
                col = find_column(df, possible_names)
                if col:
                    normalized_columns[key] = col
                else:
                    # Add the column if not found
                    normalized_columns[key] = possible_names[0]
                    df[normalized_columns[key]] = None

        # Rename columns to standard names
        rename_map = {
            normalized_columns.get('status', 'EO_STATUS'): 'EO_STATUS',
            normalized_columns.get('data', 'EO_DATA'): 'EO_DATA',
            normalized_columns.get('lat', 'EO_LAT'): 'EO_LAT',
            normalized_columns.get('lon', 'EO_LON'): 'EO_LON',
            normalized_columns.get('file', 'EO_FILE'): 'EO_FILE',
            normalized_columns.get('start', 'START'): 'START',
            normalized_columns.get('end', 'END'): 'END'
        }
        
        # Perform renaming
        df = df.rename(columns=rename_map)

        # Debugging: print out column names and first few rows
        print("DataFrame Columns:", list(df.columns))
        print("First few rows:\n", df.head())

        # Convert date columns to consistent format
        date_columns = ['START', 'END']
        for col in date_columns:
            if col in df.columns and df[col].notnull().any():
                try:
                    # Try multiple date parsing strategies
                    date_formats = [
                        '%Y-%m-%d',  # ISO format
                        '%d/%m/%Y',  # DD/MM/YYYY
                        '%m/%d/%Y',  # MM/DD/YYYY
                        '%Y/%m/%d'   # YYYY/MM/DD
                    ]
                    
                    parsed_successfully = False
                    for date_format in date_formats:
                        try:
                            df[col] = pd.to_datetime(
                                df[col], 
                                format=date_format,
                                errors='raise'
                            ).dt.strftime("%Y-%m-%d")
                            parsed_successfully = True
                            break
                        except:
                            continue
                    
                    if not parsed_successfully:
                        print(f"Warning: Could not parse dates in column {col}")
                        
                except Exception as e:
                    print(f"Error parsing dates in {col}: {e}")

        # Ensure required columns exist
        required_columns = ['EO_STATUS', 'EO_DATA', 'EO_LAT', 'EO_LON', 'EO_FILE', 'START', 'END']
        for col in required_columns:
            if col not in df.columns:
                df[col] = None

        # Set initial status
        df['EO_STATUS'] = df['EO_STATUS'].fillna('waiting')
        
        return df

    def _fetch_eo_data(
        self, 
        df: pd.DataFrame, 
        use_ee_api: bool = False, 
        interactive: bool = True, 
        num_partitions: int = 4
    ) -> pd.DataFrame:
        """
        Fetch earth observation data for the given DataFrame.
        """
        # Identify missing data columns
        eo_data_col = 'EO_DATA'
        
        # Debugging: print out column information
        print(f"Columns before fetch: {list(df.columns)}")
        print(f"Missing data check - column exists: {eo_data_col in df.columns}")
        
        # If the column doesn't exist, add it
        if eo_data_col not in df.columns:
            df[eo_data_col] = None
        
        # Find rows without EO data
        mask = df[eo_data_col].isnull()
        
        if mask.sum() > 0:
            print(f"Fetching EO data for {mask.sum()} rows")
            # Placeholder for actual EO data fetching
            df.loc[mask, eo_data_col] = df.loc[mask].apply(
                lambda row: np.random.rand(10).tolist(), 
                axis=1
            )
            df.loc[mask, 'EO_STATUS'] = 'complete'
        
        return df

    def _generate_dataset_summary(self, df: pd.DataFrame) -> str:
        """
        Generate a summary of the dataset.
        """
        # Debugging: ensure columns exist before accessing
        summary = f"Dataset: {self.name}\n"
        summary += f"Total entries: {len(df)}\n"
        
        # Check for column existence before summarizing
        if 'EO_DATA' in df.columns:
            summary += f"Entries with EO data: {df['EO_DATA'].notnull().sum()}\n"
        else:
            summary += "Entries with EO data: Column missing\n"
        
        if 'EO_STATUS' in df.columns:
            summary += f"Status distribution:\n{df['EO_STATUS'].value_counts()}"
        else:
            summary += "Status distribution: Column missing"
        
        return summary

def create_datasets(datasets: List[LabeledDataset]) -> None:
    """
    Create multiple datasets with configurable options.
    """
    parser = argparse.ArgumentParser(description="Create Labeled Datasets")
    parser.add_argument(
        "--ee_api", 
        action="store_true", 
        help="Use Earth Engine API for data fetching"
    )
    parser.add_argument(
        "--non-interactive",
        dest="interactive", 
        action="store_false", 
        help="Run in non-interactive mode"
    )
    parser.add_argument(
        "--npartitions", 
        type=int, 
        default=4, 
        help="Number of partitions for data processing"
    )
    
    args = parser.parse_args()
    
    # Create report
    report = "DATASET CREATION REPORT\n"
    report += "=" * 40 + "\n"
    
    # Ensure reports directory exists
    report_path = Path("reports")
    report_path.mkdir(parents=True, exist_ok=True)
    report_path = report_path / "dataset_creation_report.txt"
    
    # Process each dataset
    for dataset in datasets:
        try:
            summary = dataset.create_dataset(
                use_ee_api=args.ee_api,
                interactive=args.interactive,
                num_partitions=args.npartitions
            )
            report += f"\n{summary}\n"
            print(summary)
        except Exception as e:
            report += f"Error processing {dataset.name}: {str(e)}\n"
            print(f"Error processing {dataset.name}: {str(e)}")
            # Print full traceback for more details
            import traceback
            traceback.print_exc()
    
    # Write report
    with open(report_path, 'w') as f:
        f.write(report)