from pydoc import visiblename
from re import I
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
from scipy.signal import find_peaks

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
    
    # Get a random file
    random_test_files = random.sample(test_files, 3)
    
    for idx_file, file_path in enumerate(random_test_files):
                
        # Load data and labels
        data, labels = _load_data(
            file_path=file_path,
            tracked_points=["LeftFoot", "RightFoot"],
            incl_magn=False,
            mode=MODE,
            task=ML_TASK,
            labels_as=LABELS_AS,
            tolerance=TOLERANCE
        )
        
        # Re-label exact gait event timings
        events = {label: [] for label in map_index_labels.values() if label != "null"}
        relabels = {label: {"pos": [], "neg": []} for label in map_index_labels.values() if label != "null"}
        for idx_label, label in map_index_labels.items():
            if label == "FCL":
                idxs_FCL = []
                
                # Find the `annotated search window`
                idx_pos = np.argwhere(np.diff(labels[:, idx_label]) > 0)[:, 0] + 1
                idx_neg = np.argwhere(np.diff(labels[:, idx_label]) < 0)[:, 0]
                
                # Find negative-to-posive zero-crossings in the data
                idx_zc = np.argwhere((data[1:, 4] * data[:-1, 4]) < 0)[:,0]
                idx_zc = idx_zc[data[idx_zc, 4] < 0]  # negative-to-positive zero-crossings
                
                # 
                if idx_pos[0] < idx_neg[0]:
                    for ii in range(len(idx_pos)):
                        f = np.argwhere((idx_zc > idx_pos[ii]) & (idx_zc < idx_neg[ii]))[:,0]
                        if len(f) > 0:
                            idxs_FCL.append(idx_zc[f[0]])
        idxs_FCL = np.array(idxs_FCL)
        
        # Make predictions
        predictions = trained_model.predict(data[None, ...])
        
        # Threshold predicted probabilities
        thresh_proba = 0.5
        predictions_thresh = {label: np.where(predictions[0][:, idx_label] > thresh_proba, 1.0, 0.0) for idx_label, label in map_index_labels.items() if label != "null"}
        idx_pks, _ = find_peaks(predictions_thresh["FCL"], height=0.99, plateau_size=TOLERANCE//2)
        
        # Visualize
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
            axs[0].plot(idxs_FCL, data[idxs_FCL, 4], ls="None", marker="x", mec="tab:orange", ms=8, mew=2)
            axs_ = axs[0].twinx()
            axs_.plot(np.arange(data.shape[0]), data[:, 0], c="tab:red")
            axs_.plot(np.arange(data.shape[0]), data[:, 1], c="tab:purple")
            axs_.plot(np.arange(data.shape[0]), data[:, 2], c="tab:brown")
            axs[1].plot(np.arange(labels.shape[0]), labels[:, 1], c="green", alpha=0.2)
            # axs[1].plot(relabels["ICL"]["pos"], labels[relabels["ICL"]["pos"], 1], ls="None", marker="o", mfc="None", mec="green", ms=8)
            # axs[1].plot(relabels["ICL"]["neg"], labels[relabels["ICL"]["pos"], 1], ls="None", marker="o", mfc="green", mec="green", ms=8)
            axs[1].plot(np.arange(labels.shape[0]), labels[:, 2], c="blue", alpha=0.2)
            # axs[1].plot(relabels["FCL"]["pos"], labels[relabels["FCL"]["pos"], 2], ls="None", marker="o", mfc="None", mec="blue", ms=8)
            # axs[1].plot(relabels["FCL"]["neg"], labels[relabels["FCL"]["pos"], 2], ls="None", marker="o", mfc="blue", mec="blue", ms=8)
            axs[1].plot(np.arange(predictions[0].shape[0]), predictions[0][:, 1], c="green")
            axs[1].plot(np.arange(len(predictions_thresh["ICL"])), predictions_thresh["ICL"], c="green", ls="--")
            axs[1].plot(np.arange(predictions[0].shape[0]), predictions[0][:, 2], c="blue")
            axs[1].plot(np.arange(len(predictions_thresh["FCL"])), predictions_thresh["FCL"], c="blue", ls="--")
            axs[1].plot(idx_pks, predictions_thresh["FCL"][idx_pks], ls="None", marker="x", mec="blue", ms=8)
            
            axs[2].fill_between(np.arange(labels.shape[0]), 
                                labels[:, 3]*data[:, 9:12].max().max()*1.2,
                                labels[:, 3]*data[:, 9:12].min().min()*1.2,
                                fc="green", alpha=0.2)
            axs[2].fill_between(np.arange(labels.shape[0]), 
                                labels[:, 4]*data[:, 9:12].max().max()*1.2,
                                labels[:, 4]*data[:, 9:12].min().min()*1.2,
                                fc="blue", alpha=0.2)
            axs[2].plot(np.arange(data.shape[0]), data[:, 9:12])
            axs_ = axs[2].twinx()
            axs_.plot(np.arange(data.shape[0]), data[:, 6], c="tab:red")
            axs_.plot(np.arange(data.shape[0]), data[:, 7], c="tab:purple")
            axs_.plot(np.arange(data.shape[0]), data[:, 8], c="tab:brown")
            axs[3].plot(np.arange(labels.shape[0]), labels[:, 3], c="green", alpha=0.2)
            axs[3].plot(np.arange(labels.shape[0]), labels[:, 4], c="blue", alpha=0.2)
            axs[3].plot(np.arange(predictions[0].shape[0]), predictions[0][:, 3], c="green")
            axs[3].plot(np.arange(len(predictions_thresh["ICR"])), predictions_thresh["ICR"], c="green", ls="--")
            axs[3].plot(np.arange(predictions[0].shape[0]), predictions[0][:, 4], c="blue")
            axs[3].plot(np.arange(len(predictions_thresh["FCR"])), predictions_thresh["FCR"], c="blue", ls="--")
            
            for idx_ax in range(len(axs)):
                axs[idx_ax].spines["top"].set_visible(False)
                axs[idx_ax].spines["right"].set_visible(False)
                if idx_ax in [0, 2]:
                    axs[idx_ax].spines["bottom"].set_position("zero")
                axs[idx_ax].grid(alpha=0.1, ls=":")
            
            axs[-1].set_xlim((0, data.shape[0]))
            plt.tight_layout()
            plt.show()
    return 

if __name__ == "__main__":
    main()