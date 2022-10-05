import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tcn import TCN
from .losses import MyWeightedBinaryCrossentropy, MyWeightedCategoricalCrossentropy, MyWeightedMeanSquaredError
import os, sys
sys.path.append("../utils")
from utils.data_utils import SAMPLING_FREQUENCY, MAP_INDEX_EVENTS, MAP_INDEX_PHASES

def get_model(
        num_input_channels: int = 12,
        mode: str = "gait_events",
        task: str = "classification",
        **kwargs):

    # Get number of output classes
    num_output_classes = len(MAP_INDEX_EVENTS) if mode == "gait_events" else len(MAP_INDEX_PHASES)

    # Get mapping from index to label
    map_index_labels = MAP_INDEX_EVENTS if mode == "gait_events" else MAP_INDEX_PHASES

    # Define the layers
    input_layer = Input(shape=(None, num_input_channels), name="input_layer")
    tcn_layer = TCN(
        use_batch_norm=True,
        use_skip_connections=True,
        return_sequences=True,
        name="tcn_layer")(input_layer)
    if task == "classification":
        output_layer = Dense(units=num_output_classes, activation="softmax", name="output_layer")(tcn_layer)
    else:
        output_layer = [Dense(units=1, activation="sigmoid", name=lbl)(tcn_layer) for lbl in map_index_labels.values() if lbl != "null"]

    # Create model
    model = Model(inputs=input_layer, outputs=output_layer, name="tcn_model")

    # Define the loss function(s)
    if task == "classification":
        weights = [[1. for _ in range(len(list(map_index_labels.keys())))]]
        weights[0][0] = 1./100
        loss = MyWeightedCategoricalCrossentropy(weights=weights)
    else:
        loss = {lbl: MyWeightedMeanSquaredError(weights=1./100) for lbl in map_index_labels.values() if lbl != "null"}

    # Compile the model
    model.compile(loss=loss, optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4))
    return model
