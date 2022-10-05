import numpy as np
import os, sys, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tcn import TCN
import matplotlib.pyplot as plt
from utils.data_utils import SAMPLING_FREQUENCY, MAP_INDEX_EVENTS, MAP_INDEX_PHASES
from utils.data_utils import split_train_test, _load_data
from custom.losses import MyWeightedCategoricalCrossentropy

gpus = tf.config.list_physical_devices(device_type="GPU")
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    assert tf.config.experimental.get_memory_growth(gpus[0])
except:
    pass

# Globals
DATASET_DIR = "/home/robbin/Datasets/Mobilise-D/rawdata/MicroWB"
BASE_CHECKPOINT_PATH = "./runs/test"
MODE = "gait_events"
ML_TASK = "classification"
LABELS_AS = "binary"
TOLERANCE = int(0.050 * SAMPLING_FREQUENCY)
WIN_LEN = int(6 * SAMPLING_FREQUENCY)

def main():
    # Get mapping from index to labels
    map_index_labels = MAP_INDEX_EVENTS if MODE == "gait_events" else MAP_INDEX_PHASES
    
    # Get number of output classes
    num_classes = len(map_index_labels)
    
    # Get list of train, val and test files
    train_files, val_files, test_files = split_train_test(DATASET_DIR, seed=123)
    
    # Get a trained model
    trained_model = tf.keras.models.load_model(
        os.path.join(BASE_CHECKPOINT_PATH, "01"),
        custom_objects={"MyWeightedCategoricalCrossentropy": MyWeightedCategoricalCrossentropy}
    )
    trained_model.summary()
    
    # Loop over test files
    for idx_file, file_name in enumerate(test_files):
        with open(file_name, 'rb') as infile:
            data, labels = _load_data(file_name,
                                      tracked_points=["LeftFoot", "RightFoot"],
                                      incl_magn=False,
                                      mode=MODE,
                                      task=ML_TASK,
                                      labels_as=LABELS_AS,
                                      tolerance=TOLERANCE)
        
        # Make predictions
        predictions = trained_model.predict(data[None, ...])
        
        iplot = 1
        if iplot:
            fig, axs = plt.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [3, 1, 3, 1]})
            axs[0].fill_between(np.arange(labels.shape[0]), 
                                labels[:, 1]*data[:, 3:6].max().max()*1.2,
                                labels[:, 1]*data[:, 3:6].min().min()*1.2,
                                fc="green", alpha=0.2)
            axs[0].fill_between(np.arange(labels.shape[0]), 
                                labels[:, 2]*data[:, 3:6].max().max()*1.2,
                                labels[:, 2]*data[:, 3:6].min().min()*1.2,
                                fc="blue", alpha=0.2)
            axs[0].plot(np.arange(data.shape[0]), data[:, 3:6])
            axs[1].plot(np.arange(labels.shape[0]), labels[:, 1], c="green", alpha=0.2)
            axs[1].plot(np.arange(labels.shape[0]), labels[:, 2], c="blue", alpha=0.2)
            axs[1].plot(np.arange(predictions[0].shape[0]), predictions[0][:, 1], c="green")
            axs[1].plot(np.arange(predictions[0].shape[0]), predictions[0][:, 2], c="blue")
            axs[2].fill_between(np.arange(labels.shape[0]), 
                                labels[:, 3]*data[:, 9:12].max().max()*1.2,
                                labels[:, 3]*data[:, 9:12].min().min()*1.2,
                                fc="green", alpha=0.2)
            axs[2].fill_between(np.arange(labels.shape[0]), 
                                labels[:, 4]*data[:, 9:12].max().max()*1.2,
                                labels[:, 4]*data[:, 9:12].min().min()*1.2,
                                fc="blue", alpha=0.2)
            axs[2].plot(np.arange(data.shape[0]), data[:, 9:12])
            axs[3].plot(np.arange(labels.shape[0]), labels[:, 3], c="green", alpha=0.2)
            axs[3].plot(np.arange(labels.shape[0]), labels[:, 4], c="blue", alpha=0.2)
            axs[3].plot(np.arange(predictions[0].shape[0]), predictions[0][:, 3], c="green")
            axs[3].plot(np.arange(predictions[0].shape[0]), predictions[0][:, 4], c="blue")
            plt.tight_layout()
            plt.show()
            break
    return 

if __name__ == "__main__":
    main()