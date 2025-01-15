import configparser
import os
from typing import Dict, List, Tuple

class ModelReader:
    def __init__(self, scanner_path: str):
        """Initialize ModelReader with path to ScannerS input files."""
        self.scanner_path = scanner_path
        
        self.model_names = {
            'N2HDMDarkSD.ini': 'N2HDM Fully Dark Phase',
            'R2HDM_T1.ini': 'Real 2HDM Type 1',
            'C2HDM_T1.ini': 'Complex 2HDM Type 1',
            'C2HDM_T2.ini': 'Complex 2HDM Type 2',
            'N2HDMDarkD.ini': 'N2HDM Dark Doublet Phase',
            'N2HDMDarkS_T1.ini': 'N2HDM Dark Singlet Phase Type 1',
            'C2HDM_FL.ini': 'Complex 2HDM Flipped',
            'C2HDM_LS.ini': 'Complex 2HDM Lepton Specific',
            'CPVDM.ini': 'CP-Violating Dark Matter',
            'CxSMBroken.ini': 'Complex Singlet Broken Phase',
            'CxSMDark.ini': 'Complex Singlet Dark Phase',
            'N2HDMBroken_T2.ini': 'N2HDM Broken Phase Type 2',
            'R2HDM_FL.ini': 'Real 2HDM Flipped',
            'R2HDM_LS.ini': 'Real 2HDM Lepton Specific',
            'R2HDM_T2.ini': 'Real 2HDM Type 2',
            'TRSMBroken.ini': 'TRSM Broken Phase'
        }

    def get_available_models(self) -> Dict[str, str]:
        """Return dictionary of available models with their clear names."""
        if not os.path.exists(self.scanner_path):
            raise FileNotFoundError(f"Scanner path not found: {self.scanner_path}")
        
        available_models = {}
        for ini_file in os.listdir(self.scanner_path):
            if ini_file.endswith('.ini') and ini_file in self.model_names:
                available_models[ini_file] = self.model_names[ini_file]
        return available_models

    def read_model(self, ini_file: str) -> Tuple[Dict, Dict]:
        """Read model configuration from ini file."""
        if not ini_file.endswith('.ini'):
            ini_file += '.ini'
        
        config_path = os.path.join(self.scanner_path, ini_file)
        
        # Read constraints from the top of the file
        constraints = {}
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith(';'):
                    if '=' in line and '[scan]' not in line:
                        key, value = line.split('=')
                        key = key.strip()
                        value = value.strip()
                        if value.lower() in ['apply', 'ignore', 'skip']:
                            constraints[key] = value

        # Read features from [scan] section
        features = {}
        scan_section = False
        with open(config_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line == '[scan]':
                    scan_section = True
                    continue
                if scan_section and line and not line.startswith(';'):
                    if '=' in line:
                        key, value = line.split('=')
                        key = key.strip()
                        value = value.strip()
                        try:
                            min_val, max_val = map(float, value.split())
                            features[key] = {'min': min_val, 'max': max_val}
                        except (ValueError, TypeError):
                            continue
        
        return features, constraints

    def update_constraint(self, ini_file: str, constraint: str, value: str):
        """Update constraint value in ini file."""
        if value.lower() not in ['apply', 'ignore', 'skip']:
            raise ValueError("Constraint value must be 'apply', 'ignore', or 'skip'")
        
        config_path = os.path.join(self.scanner_path, ini_file)
        
        # Read all lines from the file
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # Find and update the constraint
        constraint_updated = False
        for i, line in enumerate(lines):
            if line.strip() and not line.startswith(';'):
                if '=' in line and '[scan]' not in line:
                    key, _ = line.split('=')
                    if key.strip() == constraint:
                        lines[i] = f"{constraint} = {value}\n"
                        constraint_updated = True
                        break
        
        # Write back to file
        with open(config_path, 'w') as f:
            f.writelines(lines)

    def update_feature_range(self, ini_file: str, feature: str, min_val: float, max_val: float):
        """Update feature range in ini file."""
        config_path = os.path.join(self.scanner_path, ini_file)
        
        # Read all lines from the file
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # Find the [scan] section and update the feature
        in_scan_section = False
        feature_updated = False
        
        for i, line in enumerate(lines):
            line = line.strip()
            if line == '[scan]':
                in_scan_section = True
                continue
            
            if in_scan_section and line and not line.startswith(';'):
                if '=' in line:
                    key, _ = line.split('=')
                    if key.strip() == feature:
                        lines[i] = f"{feature} = {min_val} {max_val}\n"
                        feature_updated = True
                        break
        
        # Write back to file
        with open(config_path, 'w') as f:
            f.writelines(lines)
