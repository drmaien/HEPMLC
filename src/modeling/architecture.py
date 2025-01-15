import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Activation
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam, Nadam
from tensorflow.keras.losses import BinaryFocalCrossentropy
from typing import List, Dict, Any
from .metrics import subset_accuracy, hamming_loss, matthews_correlation, macro_f1_score

class ModelBuilder:
    def __init__(self, input_shape: tuple, num_outputs: int):
        """Initialize model builder with input and output dimensions."""
        self.input_shape = input_shape
        self.num_outputs = num_outputs

    def build_model(self, config: Dict[str, Any]) -> tf.keras.Model:
        """
        Build model based on configuration dictionary.
        
        config should include:
        - n_layers: number of hidden layers
        - n_units_[i]: number of units in layer i
        - activation: activation function
        - dropout_rate: dropout rate
        - apply_batch_norm: whether to use batch normalization
        - optimizer: 'adam' or 'nadam'
        - learning_rate: optimizer learning rate
        - regularization: None or 'l2'
        - reg_lambda: l2 regularization factor
        """
        model = Sequential()
        model.add(Input(shape=self.input_shape))

        # Add hidden layers
        for i in range(config['n_layers']):
            units = config[f'n_units_{i}']
            reg = l2(config['reg_lambda']) if config['regularization'] == 'l2' else None
            
            model.add(Dense(units, kernel_regularizer=reg))
            model.add(Activation(config['activation']))
            
            if config['apply_batch_norm']:
                model.add(BatchNormalization())
                
            model.add(Dropout(config['dropout_rate']))

        # Output layer
        model.add(Dense(self.num_outputs, activation='sigmoid'))

        # Configure optimizer
        if config['optimizer'] == 'adam':
            optimizer = Adam(learning_rate=config['learning_rate'])
        else:
            optimizer = Nadam(learning_rate=config['learning_rate'])

        # Compile model
        model.compile(
            optimizer=optimizer,
            loss=BinaryFocalCrossentropy(gamma=2.0, from_logits=False),
            metrics=[
                subset_accuracy,
                hamming_loss,
                matthews_correlation,
                macro_f1_score
            ]
        )

        return model

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """Get default model configuration."""
        return {
            'n_layers': 3,
            'n_units_0': 875,
            'n_units_1': 938,
            'n_units_2': 402,
            'activation': 'relu',
            'dropout_rate': 0.117,
            'apply_batch_norm': True,
            'optimizer': 'adam',
            'regularization': None,
            'reg_lambda': 0.05,
            'learning_rate': 0.000263
        }
