from gettext import find
from pydoc import visiblename
import numpy as np
import os, sys, random
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import tensorflow as tf
from tcn import TCN
from utils.data_utils import DATASET_DIR, SAMPLING_FREQUENCY, MAP_INDEX_EVENTS, MAP_INDEX_PHASES
from utils.data_utils import split_train_test, _load_data
from utils.eval_utils import match_events
from custom.losses import MyWeightedCategoricalCrossentropy

gpus = tf.config.list_physical_devices(device_type="GPU")
try:
    tf.config.experimental.set_memory_growth(gpus[0], True)
    assert tf.config.experimental.get_memory_growth(gpus[0])
except:
    pass

# Globals
BASE_CHECKPOINT_PATH = ".\\runs\\test"
MODE = "gait_events"
ML_TASK = "classification"
LABELS_AS = "binary"
TOLERANCE = int(0.050 * SAMPLING_FREQUENCY)
WIN_LEN = int(6 * SAMPLING_FREQUENCY)


def main():
    # Get mapping from index to labels
    map_index_labels = MAP_INDEX_EVENTS if MODE=="gait_events" else MAP_INDEX_PHASES

    # Get number of output classes
    num_classes = len(map_index_labels)

    # Get train, val and test files
    _, _, test_files = split_train_test(DATASET_DIR, test_size=0.2, seed=123)

    # Get models
    trained_models = [
        tf.keras.models.load_model(
            os.path.join(BASE_CHECKPOINT_PATH, folder_name),
            custom_objects={"MyWeightedCategoricalCrossentropy": MyWeightedCategoricalCrossentropy}
        ) for folder_name in os.listdir(BASE_CHECKPOINT_PATH) if os.path.isdir(os.path.join(BASE_CHECKPOINT_PATH, folder_name))
    ]

    # Get a random file
    random.seed(43)
    random_test_file = random.sample(test_files, k=1)[0]

    # Load data and labels
    data, labels = _load_data(
        file_path=random_test_file,
        tracked_points=["LeftFoot", "RightFoot"],
        incl_magn=False,
        mode=MODE,
        task=ML_TASK,
        labels_as=LABELS_AS,
        tolerance=0
    )

    # Dictionary of annotated indices
    labels_idx = {evnt: np.argwhere(labels[:, idx_evnt]==1)[:,0] for idx_evnt, evnt in map_index_labels.items() if evnt != "null"}

    # Make predictions
    predictions = [
        model.predict(data[None, ...]) for model in trained_models
    ]

    # Visualize
    fig, axs = plt.subplots(4, 2, sharex=True, gridspec_kw={"height_ratios": [3, 3, 1, 1]})
    axs[0, 0].plot(np.arange(len(data)), data[:, 0:3], lw=1)
    axs[0, 1].plot(np.arange(len(data)), data[:, 6:9], lw=1)
    axs[1, 0].plot(np.arange(len(data)), data[:, 3:6], lw=1)
    axs[1, 1].plot(np.arange(len(data)), data[:, 9:], lw=1)
    axs[2, 0].plot(np.arange(len(labels)), labels[:, 1], lw=1)
    axs[3, 0].plot(np.arange(len(labels)), labels[:, 2], lw=1)
    axs[2, 1].plot(np.arange(len(labels)), labels[:, 3], lw=1)
    axs[3, 1].plot(np.arange(len(labels)), labels[:, 4], lw=1)
    for preds in predictions:
        axs[2, 0].plot(np.arange(len(preds[0])), preds[0][:, 1], lw=2)
        axs[3, 0].plot(np.arange(len(preds[0])), preds[0][:, 2], lw=2)
        axs[2, 1].plot(np.arange(len(preds[0])), preds[0][:, 3], lw=2)
        axs[3, 1].plot(np.arange(len(preds[0])), preds[0][:, 4], lw=2)
    for row in axs:
        for col in row:
            col.spines["top"].set_visible(False)
            col.spines["right"].set_visible(False)
            col.spines["bottom"].set_position("zero")
            col.grid(visible=True, which="both", axis="both", alpha=0.2, ls=":")
    axs[2, 0].set_ylim((-.2, 1.2))
    axs[2, 1].set_ylim((-.2, 1.2))
    axs[3, 0].set_ylim((-.2, 1.2))
    axs[3, 1].set_ylim((-.2, 1.2))
    axs[3, 1].set_xlim((0, len(data)))
    plt.show()
    return

if __name__ == "__main__":
    main()