import os
import optuna
import tensorflow as tf
import numpy as np
from typing import Dict, Any, Tuple
from optuna.trial import Trial
import pandas as pd
import json

class OptunaCallback(tf.keras.callbacks.Callback):
    """Custom callback for Optuna optimization."""
    def __init__(self, trial: Trial, monitor: str = 'val_subset_accuracy'):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.best_value = float('-inf')

    def on_epoch_end(self, epoch: int, logs: Dict = None):
        current_value = logs.get(self.monitor)
        if current_value is None:
            return
        
        # Report value to Optuna
        self.trial.report(current_value, epoch)
        self.best_value = max(self.best_value, current_value)

        # Handle pruning
        if self.trial.should_prune():
            raise optuna.TrialPruned(f"Trial pruned at epoch {epoch}")

class ModelOptimizer:
    def __init__(self, 
                 model_builder,
                 X_train, y_train,
                 X_val, y_val,
                 output_dir: str,
                 n_trials: int = 10):
        """Initialize optimizer with data and parameters."""
        self.model_builder = model_builder
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.output_dir = output_dir
        self.n_trials = n_trials
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
    def suggest_parameters(self, trial: Trial) -> Dict[str, Any]:
        """Suggest hyperparameters for the trial."""
        params = {
            'n_layers': trial.suggest_int('n_layers', 2, 4),
            'activation': trial.suggest_categorical('activation', ['relu', 'leaky_relu']),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.1, 0.5),
            'apply_batch_norm': trial.suggest_categorical('apply_batch_norm', [True, False]),
            'optimizer': trial.suggest_categorical('optimizer', ['adam', 'nadam']),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'batch_size': trial.suggest_int('batch_size', 512, 2048),
            'regularization': trial.suggest_categorical('regularization', [None, 'l2']),
        }
        
        # Add layer-specific parameters
        for i in range(params['n_layers']):
            params[f'n_units_{i}'] = trial.suggest_int(f'n_units_{i}', 128, 1024)
            
        # Add regularization parameter if needed
        if params['regularization'] == 'l2':
            params['reg_lambda'] = trial.suggest_float('reg_lambda', 1e-5, 1e-2, log=True)
        else:
            params['reg_lambda'] = 0.0
            
        return params
        
    def objective(self, trial: Trial) -> float:
        """Objective function for optimization."""
        # Get parameters for this trial
        params = self.suggest_parameters(trial)
        
        # Build model
        model = self.model_builder.build_model(params)
        
        # Create callbacks
        callbacks = [
            OptunaCallback(trial),
            tf.keras.callbacks.EarlyStopping(
                monitor='val_subset_accuracy',
                patience=10,
                restore_best_weights=True
            )
        ]
        
        try:
            # Train model
            history = model.fit(
                self.X_train, self.y_train,
                validation_data=(self.X_val, self.y_val),
                epochs=20,  # Maximum epochs
                batch_size=params['batch_size'],
                callbacks=callbacks,
                verbose=1
            )
            
            # Get best validation subset accuracy
            best_val_accuracy = max(history.history['val_subset_accuracy'])
            
            # Save trial results
            trial_dir = os.path.join(self.output_dir, f'trial_{trial.number}')
            os.makedirs(trial_dir, exist_ok=True)
            
            # Save history
            with open(os.path.join(trial_dir, 'history.json'), 'w') as f:
                json.dump(history.history, f)
            
            # Save parameters
            with open(os.path.join(trial_dir, 'parameters.json'), 'w') as f:
                json.dump(params, f)
            
            return best_val_accuracy
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial failed: {e}")
            raise optuna.TrialPruned()
            
    def optimize(self) -> optuna.Study:
        """Run optimization."""
        # Create study
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=10,
            interval_steps=1
        )
        
        study = optuna.create_study(
            direction='maximize',
            pruner=pruner,
            study_name='model_optimization'
        )
        
        # Run optimization
        study.optimize(
            self.objective,
            n_trials=self.n_trials,
            catch=(Exception,)
        )
        
        # Save study results
        study.trials_dataframe().to_csv(
            os.path.join(self.output_dir, 'study_results.csv')
        )
        
        # Save best parameters
        with open(os.path.join(self.output_dir, 'best_parameters.json'), 'w') as f:
            json.dump(study.best_params, f)
            
        return study
