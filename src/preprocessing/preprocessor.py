import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer, StandardScaler
from typing import Dict, Tuple, Optional
import joblib

class FeaturePreprocessor:
    def __init__(self, apply_yj: bool = True, apply_scaler: bool = True):
        """Initialize preprocessor with options."""
        self.apply_yj = apply_yj
        self.apply_scaler = apply_scaler
        self.yj_transformer = PowerTransformer(method='yeo-johnson') if apply_yj else None
        self.scaler = StandardScaler() if apply_scaler else None
        
    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit transformers and transform data."""
        X_processed = X.copy()
        
        if self.apply_yj:
            X_processed = self.yj_transformer.fit_transform(X_processed)
            
        if self.apply_scaler:
            X_processed = self.scaler.fit_transform(X_processed)
            
        return X_processed
    
    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform data using fitted transformers."""
        X_processed = X.copy()
        
        if self.apply_yj:
            if self.yj_transformer is None:
                raise ValueError("Yeo-Johnson transformer not fitted!")
            X_processed = self.yj_transformer.transform(X_processed)
            
        if self.apply_scaler:
            if self.scaler is None:
                raise ValueError("Scaler not fitted!")
            X_processed = self.scaler.transform(X_processed)
            
        return X_processed
    
    def save_transformers(self, output_dir: str):
        """Save fitted transformers."""
        os.makedirs(output_dir, exist_ok=True)
        
        if self.apply_yj and self.yj_transformer is not None:
            joblib.dump(self.yj_transformer, 
                       os.path.join(output_dir, 'yj_transformer.joblib'))
            
        if self.apply_scaler and self.scaler is not None:
            joblib.dump(self.scaler, 
                       os.path.join(output_dir, 'scaler.joblib'))
            
        # Save configuration
        config = {
            'apply_yj': self.apply_yj,
            'apply_scaler': self.apply_scaler
        }
        joblib.dump(config, os.path.join(output_dir, 'preprocessor_config.joblib'))
    
    @classmethod
    def load_transformers(cls, model_dir: str) -> 'FeaturePreprocessor':
        """Load saved transformers."""
        config = joblib.load(os.path.join(model_dir, 'preprocessor_config.joblib'))
        preprocessor = cls(**config)
        
        if config['apply_yj']:
            preprocessor.yj_transformer = joblib.load(
                os.path.join(model_dir, 'yj_transformer.joblib'))
            
        if config['apply_scaler']:
            preprocessor.scaler = joblib.load(
                os.path.join(model_dir, 'scaler.joblib'))
            
        return preprocessor

    def describe_transformations(self, X: pd.DataFrame) -> Dict:
        """Describe the effect of transformations on the data."""
        stats = {
            'original': X.describe(),
        }
        
        if self.apply_yj:
            yj_data = pd.DataFrame(
                self.yj_transformer.transform(X),
                columns=X.columns
            )
            stats['after_yj'] = yj_data.describe()
            
        if self.apply_scaler:
            if self.apply_yj:
                scaler_input = yj_data
            else:
                scaler_input = X
                
            scaled_data = pd.DataFrame(
                self.scaler.transform(scaler_input),
                columns=X.columns
            )
            stats['after_scaling'] = scaled_data.describe()
            
        return stats
