import numpy as np
import os, random, sys, time, datetime
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import matplotlib.pyplot as plt
import tensorflow as tf
from utils.data_utils import SAMPLING_FREQUENCY, DATASET_DIR, MAP_INDEX_EVENTS, MAP_INDEX_PHASES
from utils.data_utils import split_train_test
from utils.data_loader import get_data_loader
from custom.models import get_model
gpus = tf.config.list_physical_devices(device_type="GPU")
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    assert tf.config.experimental.get_memory_growth(gpus[0])
except:
    pass

# Globals
BASE_CHECKPOINT_PATH = "./runs/test"
MODE = "gait_events"
ML_TASK = "classification"
LABELS_AS = "binary"
TOLERANCE = int(0.050 * SAMPLING_FREQUENCY)
WIN_LEN = int(6 * SAMPLING_FREQUENCY)

def main(mode: str = "gait_events",
        task: str = "classification",
        labels_as: str = "binary",
        tolerance: int = int(0.050 * SAMPLING_FREQUENCY),
        win_len: int = int(3 * SAMPLING_FREQUENCY), 
        step_len: int = None):

    # Get mapping from index to labels
    map_index_labels = MAP_INDEX_EVENTS if mode == "gait_events" else MAP_INDEX_PHASES

    # Get number of output classes
    num_classes = len(map_index_labels)

    if task == "classification" and labels_as == "gauss":
        raise ValueError(f"For a classification task we only allow binary (0/1) labels.")

    # Get list of train, val and test files
    train_files, val_files, test_files = split_train_test(DATASET_DIR, seed=123)

    # Create train and val dataset
    if task == "classification":
        train_ds = tf.data.Dataset.from_generator(
            get_data_loader(train_files, win_len=win_len, step_len=step_len, mode=mode, task=task, labels_as=labels_as, tolerance=tolerance),
            output_signature=(tf.TensorSpec(shape=(win_len, 12)),
                tf.TensorSpec(shape=(win_len, num_classes))))
        val_ds = tf.data.Dataset.from_generator(
            get_data_loader(val_files, win_len=win_len, mode=mode, task=task, labels_as=labels_as, tolerance=tolerance),
            output_signature=(tf.TensorSpec(shape=(win_len, 12)),
                tf.TensorSpec(shape=(win_len, num_classes))))
    elif task == "regression":
        train_ds = tf.data.Dataset.from_generator(
            get_data_loader(train_files, win_len=win_len, step_len=step_len, mode=mode, task=task, labels_as=labels_as, tolerance=tolerance),
            output_signature=(tf.TensorSpec(shape=(win_len, 12)),
                {lbl: tf.TensorSpec(shape=(win_len, 1)) for lbl in map_index_labels.values() if lbl != "null"}))
        val_ds = tf.data.Dataset.from_generator(
            get_data_loader(val_files, win_len=win_len, mode=mode, task=task, labels_as=labels_as, tolerance=tolerance),
            output_signature=(tf.TensorSpec(shape=(win_len, 12)),
                {lbl: tf.TensorSpec(shape=(win_len, 1)) for lbl in map_index_labels.values() if lbl != "null"}))
    
    # Create batches of train and validation data
    train_ds = train_ds.shuffle(buffer_size=128).batch(batch_size=32).repeat()
    val_ds = val_ds.batch(batch_size=32)

    # Get a modelprint(f"{'*'*65:s}")
    model = get_model(num_input_channels=12, mode=mode, task=task)
    model.summary()

    # Define callbacks
    checkpoint_path = os.path.join(BASE_CHECKPOINT_PATH, f"04")
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)
    csv_logger = tf.keras.callbacks.CSVLogger(
        filename=os.path.join(checkpoint_path, "learning_curves.tsv"), separator="\t", append=False)
    model_cp = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=False, save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=8, mode="min")

    # Fit model to data
    history = model.fit(
        train_ds,
        epochs=200,
        steps_per_epoch=295,
        validation_data=val_ds,
        validation_steps=None,
        callbacks=[csv_logger, model_cp, early_stopping])
    return


if __name__ == "__main__":
    main(win_len=WIN_LEN, step_len=WIN_LEN//2, tolerance=TOLERANCE)
