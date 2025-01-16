# HEPMLC 

HEPMLC is a tool designed for training multi-label classifiers to predict theoretical and experimental constraints in extensions of the Standard Model. 

This version works with the public tool ScannerS. However, data preprocessing, optimization, training, and evaluation modules work with any approprately labeled dataset, with features (input parameters) and binary labels (constraints valid/invalid).

## Description

HEPMLC provides a streamlined workflow for:
- Selecting and configuring physics models from ScannerS
- Generating training data.
- Class balance analysis and recommendations 
- Training deep learning models for multi-label classification
- Evaluating model performance with comprehensive metrics

The last three steps work with any labeled dataset regardless of the physics tool.

The tool is based on the work described in [arXiv:2409.05453](https://arxiv.org/abs/2409.05453).

## Prerequisites

- Python 3.8+
- TensorFlow 2.x
- ScannerS ([Installation Guide](https://gitlab.com/jonaswittbrodt/ScannerS))
- Additional Python packages:
 - numpy
 - pandas
 - scikit-learn 
 - matplotlib
 - seaborn
 - optuna

## Installation

1. If you do not already have labeled data, you can start by installing the public tool ScannerS following their [installation guide](https://gitlab.com/jonaswittbrodt/ScannerS).

2. If you decide to use HEPMLC with ScannerS, then clone HEPMLC into your ScannerS build directory:
```bash
cd ScannerS/build
git clone https://github.com/drmaien/HEPMLC.git
```

## Example Usage
Detailed workflow exploiting all HEPMLC functionalities can be found in HEPMLC/notebooks

For a basic example:
```bash
cd HEPMLC
jupyter notebook HEPMLC.ipynb 
```

Follow the notebook sections to:
1. Select a physics model
2. Generate training data
3. Train and evaluate the classifier
4. Access results in the Results directory

## Citation
If you use HEPMLC in your research, please cite:
- Maien Binjonaid, Multilabel Classification of Parameter Constraints in BSM Extensions using Deep Learning. [2409.05453]( 	
https://doi.org/10.48550/arXiv.2409.05453).

If you use HEPMLC with ScannerS for data generation, then please also cite:
- [ScannerS](https://gitlab.com/jonaswittbrodt/ScannerS) and its associated packages/papers. 

Please also cite relevant data and ML packages, referenced in 2409.05453.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Contact
For questions or issues, please open an issue on GitHub or contact me at maien@ksu.edu.sa.
