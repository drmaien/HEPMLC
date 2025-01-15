import os
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import seaborn as sns

class ScannerRunner:
    def __init__(self, build_path: str = None):
        """Initialize scanner with path to ScannerS build directory"""
        if build_path is None:
            # Get the absolute path to build directory
            current_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
            self.build_path = os.path.dirname(current_dir)
        else:
            self.build_path = build_path

        # Map model names to their executables
        self.model_executables = {
            'N2HDMDarkD.ini': 'N2HDMDarkD',
            'R2HDM_T1.ini': 'R2HDM',
            'N2HDMDarkSD.ini': 'N2HDMDarkSD',
            'C2HDM_T1.ini': 'C2HDM',
            'C2HDM_T2.ini': 'C2HDM',
            'N2HDMDarkS_T1.ini': 'N2HDMDarkS', #only this type compatible with micrOMEGAs as per ScannerS paper.
            'C2HDM_FL.ini': 'C2HDM',
            'C2HDM_LS.ini': 'C2HDM',
            'CPVDM.ini': 'CPVDM',
            'CxSMBroken.ini': 'CxSMBroken',
            'CxSMDark.ini': 'CxSMDark',
            'N2HDMBroken_T2.ini': 'N2HDMBroken', #this can be changed to other types.
            'R2HDM_FL.ini': 'R2HDM',
            'R2HDM_LS.ini': 'R2HDM',
            'R2HDM_T2.ini': 'R2HDM',
            'TRSMBroken.ini': 'TRSMBroken'
        }

    def run_scan(self, model: str, n_points: int, output_file: str) -> str:
        """Run a scan for specified model and number of points"""
        if not model.endswith('.ini'):
            model += '.ini'
            
        if model not in self.model_executables:
            raise ValueError(f"Unknown model: {model}")
            
        executable = self.model_executables[model]
        config_file = os.path.join('example_input', model)
        
        # Construct command
        cmd = f'./{executable} {output_file} --config {config_file} scan -n {n_points}'
        
        # Run the command
        try:
            subprocess.run(cmd, shell=True, check=True, cwd=self.build_path)
            return output_file
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Scan failed with error: {e}")

    def analyze_class_balance(self, data_file: str, labels: List[str]) -> Dict:
        """Analyze class balance for given labels"""
        # Construct full path to data file in build directory
        data_path = os.path.join(self.build_path, data_file)
        
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found: {data_path}")
            
        df = pd.read_csv(data_path, sep='\t')
        
        # Rest of the method remains the same
        single_label_stats = {}
        for label in labels:
            counts = df[label].value_counts()
            single_label_stats[label] = counts.to_dict()
        
        # Multi-label analysis (if more than one label)
        if len(labels) > 1:
            df['combined_label'] = df[labels].apply(
                lambda x: '_'.join(x.astype(str)), axis=1
            )
            multi_label_counts = df['combined_label'].value_counts()
            
        return {
            'single_label': single_label_stats,
            'multi_label': multi_label_counts.to_dict() if len(labels) > 1 else None
        }

    def plot_class_distribution(self, stats: Dict, title: str = "Class Distribution"):
        """Plot class distribution statistics"""
        single_label_stats = stats['single_label']
        n_labels = len(single_label_stats)
        
        # Single label plots
        fig, axes = plt.subplots(1, n_labels, figsize=(5*n_labels, 4))
        if n_labels == 1:
            axes = [axes]
            
        for ax, (label, counts) in zip(axes, single_label_stats.items()):
            counts_series = pd.Series(counts)
            counts_series.plot(kind='bar', ax=ax)
            
            # Set title and labels
            ax.set_title(f'{label}')
            ax.set_ylabel("Count")
            ax.set_xlabel("Class")
            
            # Add value labels on bars
            for i, v in enumerate(counts_series):
                ax.text(i, v, str(v), ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Multi-label plot
        if stats['multi_label']:
            plt.figure(figsize=(12, 5))
            multi_label_df = pd.Series(stats['multi_label'])
            multi_label_df.plot(kind='bar')
            plt.title("Joint Class Distribution")
            plt.xlabel("Class Combinations")
            plt.ylabel("Count")
            plt.xticks(rotation=45)
            
            # Add value labels on bars
            for i, v in enumerate(multi_label_df):
                plt.text(i, v, str(v), ha='center', va='bottom')
                
            plt.tight_layout()
            
    def recommend_additional_scans(self, stats: Dict) -> List[str]:
        """Provide recommendations for additional scans based on class balance"""
        recommendations = []
        threshold = 0.1  # 10% threshold
        min_total_points = 1000  # Minimum recommended total points
        
        # First check total number of points
        first_label_counts = list(stats['single_label'].values())[0]
        total_points = sum(first_label_counts.values())
        if total_points < min_total_points:
            recommendations.append(
                f"WARNING: Total points ({total_points}) is very low. "
                f"Recommend running at least {min_total_points} points for reliable statistics."
            )
        
        # Check single label balance
        for label, counts in stats['single_label'].items():
            total = sum(counts.values())
            
            # Check if any class is missing entirely
            possible_classes = {0, 1}  # Binary classification
            missing_classes = possible_classes - set(counts.keys())
            if missing_classes:
                recommendations.append(
                    f"Label {label} is severely imbalanced: missing class(es) {missing_classes}"
                )
                
            # Check class ratios
            for class_val, count in counts.items():
                ratio = count/total
                if ratio <= threshold:
                    recommendations.append(
                        f"Label {label} is imbalanced: class {class_val} has only {count} points ({ratio:.1%})"
                    )
        
        # Check multi-label balance if available
        if stats['multi_label']:
            total = sum(stats['multi_label'].values())
            
            # Calculate number of possible combinations (2^n for n labels)
            n_labels = len(stats['single_label'])
            possible_combos = 2**n_labels
            actual_combos = len(stats['multi_label'])
            
            if actual_combos < possible_combos:
                recommendations.append(
                    f"Joint class distribution is incomplete: found {actual_combos} out of {possible_combos} possible combinations"
                )
                
            for combo, count in stats['multi_label'].items():
                ratio = count/total
                if ratio <= threshold:
                    recommendations.append(
                        f"Joint class {combo} is underrepresented: only {count} points ({ratio:.1%})"
                    )
        
        return recommendations if recommendations else ["Class balance looks good! No additional scans needed."]
