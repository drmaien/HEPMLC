# src/preprocessing/data_splitter.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Tuple

class DataSplitter:
    def __init__(self, data: pd.DataFrame, feature_cols: list, label_cols: list, random_state: int = 42):
        """Initialize with data and column specifications."""
        self.data = data
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.random_state = random_state
        
        self.X = data[feature_cols]
        self.y = data[label_cols]

    def create_splits(self, test_size: float = 0.15, val_size: float = 0.15) -> Dict:
        """
        Create train/validation/test splits.
        test_size and val_size are relative to total dataset size
        """
        # First split off test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.X, self.y, 
            test_size=test_size, 
            random_state=self.random_state
        )
        
        # Then split remaining data into train and validation
        # Calculate validation size relative to remaining data
        remaining_val_size = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=remaining_val_size, 
            random_state=self.random_state
        )
        
        splits = {
            'train': (X_train, y_train),
            'val': (X_val, y_val),
            'test': (X_test, y_test)
        }
        
        # Print split sizes
        print(f"Training set: {len(X_train)} samples ({len(X_train)/len(self.X):.1%})")
        print(f"Validation set: {len(X_val)} samples ({len(X_val)/len(self.X):.1%})")
        print(f"Test set: {len(X_test)} samples ({len(X_test)/len(self.X):.1%})")
        
        return splits

    def save_splits(self, splits: Dict, output_dir: str):
        """Save the splits to files."""
        os.makedirs(output_dir, exist_ok=True)
        
        for split_name, (X, y) in splits.items():
            # Save features and labels together
            split_data = pd.concat([X, y], axis=1)
            filepath = os.path.join(output_dir, f"{split_name}_set.tsv")
            split_data.to_csv(filepath, sep='\t', index=False)
            print(f"Saved {split_name} set to {filepath}")
