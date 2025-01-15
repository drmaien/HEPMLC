import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
from sklearn.preprocessing import PowerTransformer, StandardScaler

class FeatureAnalyzer:
    def __init__(self, data: pd.DataFrame, feature_cols: List[str], label_cols: List[str]):
        """Initialize with data and column names."""
        self.data = data
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.features = data[feature_cols]
        self.labels = data[label_cols]
        
    def generate_feature_stats(self) -> Dict:
        """Generate basic statistics for each feature."""
        stats = {}
        for col in self.feature_cols:
            stats[col] = {
                'mean': self.features[col].mean(),
                'std': self.features[col].std(),
                'min': self.features[col].min(),
                'max': self.features[col].max(),
                'skew': self.features[col].skew(),
                'kurtosis': self.features[col].kurtosis()
            }
        return stats
    
    def plot_feature_distributions(self, output_dir: str = None):
        """Plot distribution of each feature."""
        n_features = len(self.feature_cols)
        fig, axes = plt.subplots(n_features, 2, figsize=(15, 5*n_features))
        
        for i, feature in enumerate(self.feature_cols):
            # Histogram
            sns.histplot(self.features[feature], ax=axes[i,0])
            axes[i,0].set_title(f'{feature} Distribution')
            
            # Box plot
            sns.boxplot(x=self.features[feature], ax=axes[i,1])
            axes[i,1].set_title(f'{feature} Box Plot')
            
        plt.tight_layout()
        if output_dir:
            plt.savefig(f"{output_dir}/feature_distributions.png")
        plt.show()
    
    def plot_correlation_matrix(self, output_dir: str = None):
        """Plot correlation matrix of features."""
        corr = self.features.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0)
        plt.title('Feature Correlation Matrix')
        
        if output_dir:
            plt.savefig(f"{output_dir}/correlation_matrix.png")
        plt.show()
    
    def analyze_preprocessing_effects(self):
        """Analyze effects of different preprocessing steps."""
        # Original data
        original = self.features.copy()
        
        # Apply Yeo-Johnson
        pt = PowerTransformer(method='yeo-johnson')
        yj_transformed = pd.DataFrame(
            pt.fit_transform(original),
            columns=self.feature_cols
        )
        
        # Apply StandardScaler
        scaler = StandardScaler()
        scaled = pd.DataFrame(
            scaler.fit_transform(original),
            columns=self.feature_cols
        )
        
        # Apply both
        both = pd.DataFrame(
            scaler.fit_transform(pt.fit_transform(original)),
            columns=self.feature_cols
        )
        
        return {
            'original': original,
            'yeo_johnson': yj_transformed,
            'standard_scaled': scaled,
            'both': both
        }
