import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from sklearn.metrics import classification_report, confusion_matrix
from scipy.spatial.distance import jensenshannon

class ModelTrainer:
    def __init__(self, 
                 model_builder,
                 feature_cols: List[str],
                 label_cols: List[str],
                 output_dir: str):
        """Initialize trainer with model builder and configuration."""
        self.model_builder = model_builder
        self.feature_cols = feature_cols
        self.label_cols = label_cols
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def train(self, 
              X_train: np.ndarray,
              y_train: np.ndarray,
              X_val: np.ndarray,
              y_val: np.ndarray,
              params: Dict,
              epochs: int = 10) -> tf.keras.Model:
        """Train final model with best parameters."""
        # Build model
        model = self.model_builder.build_model(params)
        
        # Callbacks
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_subset_accuracy',
                patience=5,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ModelCheckpoint(
                os.path.join(self.output_dir, 'best_model.keras'),
                monitor='val_subset_accuracy',
                save_best_only=True
            ),
            tf.keras.callbacks.CSVLogger(
                os.path.join(self.output_dir, 'training_history.csv')
            )
        ]
        
        # Train model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=params['batch_size'],
            callbacks=callbacks,
            verbose=1
        )
        
        # Save training history
        with open(os.path.join(self.output_dir, 'training_history.json'), 'w') as f:
            json.dump(history.history, f)
            
        return model
    
    def evaluate(self, model: tf.keras.Model, X_test: np.ndarray, y_test):
        """Comprehensive model evaluation."""
        # Convert to numpy arrays if needed
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values
        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values
        # Generate predictions
        y_pred_proba = model.predict(X_test)
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Save predictions
        pd.DataFrame(y_pred, columns=self.label_cols).to_csv(
            os.path.join(self.output_dir, 'test_predictions.csv'),
            index=False
        )
        
        # Individual label evaluation
        self._evaluate_individual_labels(y_test, y_pred)
        
        # Powerset evaluation
        self._evaluate_powerset(y_test, y_pred)
        
        # Plot confusion matrices
        self._plot_confusion_matrices(y_test, y_pred)
        
        # Distribution analysis
        self._analyze_label_distributions(y_test, y_pred)

    def _evaluate_individual_labels(self, y_true, y_pred):
        """Evaluate each label separately."""
        # Convert DataFrame to numpy array if needed
        if isinstance(y_true, pd.DataFrame):
            y_true = y_true.values
        if isinstance(y_pred, pd.DataFrame):
            y_pred = y_pred.values
            
        for i, label in enumerate(self.label_cols):
            report = classification_report(
                y_true[:, i],
                y_pred[:, i],
                output_dict=True
            )
            
            # Save report
            pd.DataFrame(report).T.to_csv(
                os.path.join(self.output_dir, f'classification_report_{label}.csv')
            )
            
    def _plot_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Plot confusion matrices for each label."""
        for i, label in enumerate(self.label_cols):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {label}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.savefig(os.path.join(self.output_dir, f'confusion_matrix_{label}.png'))
            plt.close()
            
    def _evaluate_powerset(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Evaluate model using powerset approach."""
        # Convert to powerset classes
        def to_powerset_class(y):
            return ['_'.join(map(str, row)) for row in y]
            
        y_true_powerset = to_powerset_class(y_true)
        y_pred_powerset = to_powerset_class(y_pred)
        
        # Generate classification report
        report = classification_report(
            y_true_powerset,
            y_pred_powerset,
            output_dict=True
        )
        
        # Save report
        pd.DataFrame(report).T.to_csv(
            os.path.join(self.output_dir, 'powerset_classification_report.csv')
        )
        
    def _analyze_label_distributions(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Analyze and compare label distributions."""
        # Joint distribution analysis
        def get_distribution(y):
            unique, counts = np.unique(y, axis=0, return_counts=True)
            return dict(zip(map(tuple, unique), counts))
            
        true_dist = get_distribution(y_true)
        pred_dist = get_distribution(y_pred)
        
        # Compute Jensen-Shannon divergence
        all_combinations = list(set(true_dist.keys()) | set(pred_dist.keys()))
        p = np.array([true_dist.get(c, 0) for c in all_combinations])
        q = np.array([pred_dist.get(c, 0) for c in all_combinations])
        
        p = p / p.sum()
        q = q / q.sum()
        
        js_div = jensenshannon(p, q)
        
        # Save distribution analysis
        with open(os.path.join(self.output_dir, 'distribution_analysis.txt'), 'w') as f:
            f.write(f'Jensen-Shannon Divergence: {js_div}\n\n')
            f.write('Distribution Comparison:\n')
            for combo in all_combinations:
                f.write(f'{combo}: True={true_dist.get(combo, 0)}, '
                       f'Pred={pred_dist.get(combo, 0)}\n')
