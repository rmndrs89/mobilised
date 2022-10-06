from gettext import find
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

DATASET_DIR = "D:\\Datasets\\Mobilise-D\\rawdata\\MicroWB"


def main():
    # Get mapping from index to labels
    map_index_labels = MAP_INDEX_EVENTS if MODE=="gait_events" else MAP_INDEX_PHASES

    # Get number of output classes
    num_classes = len(map_index_labels)

    # Get train, val and test files
    _, _, test_files = split_train_test(DATASET_DIR, test_size=0.2, seed=123)

    # Get models
    trained_model_0 = tf.keras.models.load_model(
        os.path.join(BASE_CHECKPOINT_PATH, "00"), 
        custom_objects={"MyWeightedCategoricalCrossentropy": MyWeightedCategoricalCrossentropy})
    
    trained_model_1 = tf.keras.models.load_model(
        os.path.join(BASE_CHECKPOINT_PATH, "01"), 
        custom_objects={"MyWeightedCategoricalCrossentropy": MyWeightedCategoricalCrossentropy})

    trained_model_0.summary()
    trained_model_1.summary()

    # Get a random file
    random.seed(43)
    random_test_file = random.sample(test_files, k=1)[0]

    # Load data and labels
    data, labels_0 = _load_data(
        file_path=random_test_file,
        tracked_points=["LeftFoot", "RightFoot"],
        incl_magn=False,
        mode=MODE,
        task=ML_TASK,
        labels_as=LABELS_AS,
        tolerance=0
    )

    _, labels_1 = _load_data(
        file_path=random_test_file,
        tracked_points=["LeftFoot", "RightFoot"],
        incl_magn=False,
        mode=MODE,
        task=ML_TASK,
        labels_as=LABELS_AS,
        tolerance=TOLERANCE
    )

    # Dictionary of annotated indices
    labels_idx = {evnt: np.argwhere(labels_0[:, idx_evnt]==1)[:,0] for idx_evnt, evnt in map_index_labels.items() if evnt != "null"}


    # Make predictoins
    predictions_0 = trained_model_0.predict(data[None, ...])
    predictions_1 = trained_model_1.predict(data[None, ...])

    # Find peak probabilities
    thr_proba = 0.5
    thr_distance = int(SAMPLING_FREQUENCY//2)
    peak_probs_0 = {evnt: [] for evnt in map_index_labels.values() if evnt != "null"}
    peak_probs_1 = {evnt: [] for evnt in map_index_labels.values() if evnt != "null"}
    for idx_evnt, evnt in map_index_labels.items():
        if evnt != "null":
            idx_pks, _ = find_peaks(predictions_0[0][:, idx_evnt], height=thr_proba, distance=thr_distance)
            peak_probs_0[evnt] = idx_pks
            idx_pks, _ = find_peaks(predictions_1[0][:, idx_evnt], height=thr_proba, distance=thr_distance)
            peak_probs_1[evnt] = idx_pks
    
    # Match peak probabilities with annotated events
    matched_events_0 = {evnt: dict() for evnt in peak_probs_0.keys()}
    for idx_evnt, evnt in map_index_labels.items():
        if evnt != "null":
            map_annotations_preds, map_preds_annotations = match_events(
                np.argwhere(labels_0[:, idx_evnt]==1)[:,0],
                peak_probs_0[evnt],
                tolerance=int(SAMPLING_FREQUENCY//4),
                do_print=False
            )
            matched_events_0[evnt]["a2p"] = map_annotations_preds
            matched_events_0[evnt]["p2a"] = map_preds_annotations

    # Plot file
    fig, axs = plt.subplots(4, 1, sharex=True, gridspec_kw={"height_ratios": [3, 3, 1, 1]}, figsize=(29.7/2.54, 21/2.54), num=os.path.split(random_test_file)[-1][:-4])
    axs[0].plot(np.arange(len(data)), data[:, 0:3])
    axs[1].plot(np.arange(len(data)), data[:, 3:6])
    for idx in np.argwhere(labels_0[:, 1]==1)[:, 0]:
        axs[0].axvline(idx, c="tab:red", alpha=0.2)
        axs[1].axvline(idx, c="tab:red", alpha=0.2)
        axs[2].axvline(idx, c="tab:red")
    for idx in np.argwhere(labels_0[:, 2]==1)[:, 0]:
        axs[0].axvline(idx, c="tab:green", alpha=0.2)
        axs[1].axvline(idx, c="tab:green", alpha=0.2)
        axs[3].axvline(idx, c="tab:green")
    axs[2].fill_between(np.arange(len(labels_1)), labels_1[:, 1], fc="tab:red", alpha=0.2)
    axs[2].plot(np.arange(len(predictions_0[0])), predictions_0[0][:, 1], c="tab:purple")
    # axs[2].plot(peak_probs_0["ICL"], predictions_0[0][peak_probs_0["ICL"], 1], ls="None", marker="o", mfc="None", mec="tab:purple", ms=8)
    # axs[2].plot(peak_probs_0["ICL"][matched_events_0["ICL"]["p2a"] > -1], predictions_0[0][peak_probs_0["ICL"][matched_events_0["ICL"]["p2a"] > -1], 1], ls="None", marker="o", mfc=(0, 1, 0), mec="k", ms=8)
    # axs[2].plot(peak_probs_0["ICL"][matched_events_0["ICL"]["p2a"] <= -1], predictions_0[0][peak_probs_0["ICL"][matched_events_0["ICL"]["p2a"] <= -1], 1], ls="None", marker="o", mfc=(1, 0, 0), mec="k", ms=8)
    axs[2].plot(np.arange(len(predictions_1[0])), predictions_1[0][:, 1], c="tab:pink")
    axs[3].fill_between(np.arange(len(labels_1)), labels_1[:, 2], fc="tab:red", alpha=0.2)
    axs[3].plot(np.arange(len(predictions_0[0])), predictions_0[0][:, 2], c="tab:purple")
    # axs[3].plot(peak_probs_0["FCL"], predictions_0[0][peak_probs_0["FCL"], 2], ls="None", marker="o", mfc="None", mec="tab:purple", ms=8)
    # axs[3].plot(peak_probs_0["FCL"][matched_events_0["FCL"]["p2a"] > -1], predictions_0[0][peak_probs_0["FCL"][matched_events_0["FCL"]["p2a"] > -1], 2], ls="None", marker="o", mfc=(0, 1, 0), mec="k", ms=8)
    # axs[3].plot(peak_probs_0["FCL"][matched_events_0["FCL"]["p2a"] <= -1], predictions_0[0][peak_probs_0["FCL"][matched_events_0["FCL"]["p2a"] <= -1], 2], ls="None", marker="o", mfc=(1, 0, 0), mec="k", ms=8)
    axs[3].plot(np.arange(len(predictions_1[0])), predictions_1[0][:, 2], c="tab:pink")
    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_position("zero")
        ax.grid(visible=True, which="both", axis="both", alpha=0.5, ls=":")
        for lbl in ax.get_xticklabels() + ax.get_yticklabels():
            lbl.set_fontsize(16)
    axs[2].set_ylim((-.1, 1.1))
    axs[3].set_ylim((-.1, 1.1))
    # axs[3].set_xlim((0, len(data)))
    axs[3].set_xlim((1100, 1400))
    axs[0].set_ylabel("acceleration, $a$ / g", fontsize=16)
    axs[1].set_ylabel(r"ang. vel., $\omega$ / deg/s", fontsize=16)
    axs[2].set_ylabel("Pr(ICL)", fontsize=16)
    axs[3].set_ylabel("Pr(FCL)", fontsize=16)
    axs[3].set_xlabel("time / samples", fontsize=16)
    plt.tight_layout()
    plt.show()
    return

if __name__ == "__main__":
    main()