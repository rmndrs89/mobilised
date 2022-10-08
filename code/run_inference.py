from unittest import result
import numpy as np
import os, sys, random
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
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
BASE_CHECKPOINT_PATH = "./runs/test" if sys.platform == "linux" else ".\\runs\\test"
MODE = "gait_events"
ML_TASK = "classification"
LABELS_AS = "binary"
TOLERANCE = int(0.050 * SAMPLING_FREQUENCY)
WIN_LEN = int(6 * SAMPLING_FREQUENCY)


def main(visualize: bool = False):
    # Get mapping from index to labels
    map_index_labels = MAP_INDEX_EVENTS if MODE=="gait_events" else MAP_INDEX_PHASES
    map_labels_index = {v: k for k, v in map_index_labels.items()}  # inverse mapping

    # Get number of output classes
    num_classes = len(map_index_labels)

    # Get train, val and test files
    _, _, test_files = split_train_test(DATASET_DIR, test_size=0.2, seed=123)
    
    # Get subjects dataframe
    subjects_df = pd.read_csv(os.path.join(DATASET_DIR, "subjects.tsv"), sep="\t", header=0)

    # Get models
    runs_dirs = [d for d in os.listdir(BASE_CHECKPOINT_PATH) if os.path.isdir(os.path.join(BASE_CHECKPOINT_PATH, d))]
    runs_dirs.sort()
    trained_models = [
        tf.keras.models.load_model(
            os.path.join(BASE_CHECKPOINT_PATH, run_dir),
            custom_objects={"MyWeightedCategoricalCrossentropy": MyWeightedCategoricalCrossentropy}
        ) for run_dir in runs_dirs[2:4]
    ]
    
    # Initialize list of output dataframe
    output_dfs = [pd.DataFrame() for _ in trained_models]

    # Loop over the test files
    for idx_file, test_file_path in  enumerate(test_files):
        
        # Load data and labels
        data, labels = _load_data(
            file_path=test_file_path,
            tracked_points=["LeftFoot", "RightFoot"],
            normalize=False,
            incl_magn=False,
            mode=MODE,
            task=ML_TASK,
            labels_as=LABELS_AS,
            tolerance=0
        )

        # Get a dictionary of annotated indices of gait events
        labels_idx = {evnt: np.argwhere(labels[:, idx_evnt]==1)[:,0] for idx_evnt, evnt in map_index_labels.items() if evnt != "null"}

        # Make predictions on the test data
        predictions = [model.predict(data[None, ...]) for model in trained_models]
    
        # For each trained model, get a dictionary of predicted indices of gait events
        thr_proba = 0.5  # minimum probability
        peak_probs_idx = [{evnt: [] for evnt in list(labels_idx.keys())} for _ in trained_models]
        for p, preds in enumerate(predictions):  # iterate over the predictions for each model
            for v in list(peak_probs_idx[p].keys()):  # v: {`ICL`, `FCL`, `ICR`, `FCR`}
                i = map_labels_index[v]  # which column to look at for the current event type
                idx_pks, _ = find_peaks(predictions[p][0][:, i], height=thr_proba, distance=int(SAMPLING_FREQUENCY//2))
                peak_probs_idx[p][v] = idx_pks
        
        # Map indices of peak probabilities to labels, and vice versa
        map_labels_preds, map_preds_labels = [{evnt: [] for evnt in list(labels_idx.keys())} for mdl in trained_models], [{evnt: [] for evnt in list(labels_idx.keys())} for mdl in trained_models]
        for p, probs in enumerate(peak_probs_idx):
            for evnt in list(probs.keys()):
                map_labels_preds[p][evnt], map_preds_labels[p][evnt] = match_events(labels_idx[evnt], peak_probs_idx[p][evnt], tolerance=int(SAMPLING_FREQUENCY//4))
    
        # For each trained model,
        for p, probs in enumerate(peak_probs_idx):
            
            # Create pandas DataFrame to store results
            results_df = pd.DataFrame()
            
            # Loop over the event types
            for evnt in list(labels_idx.keys()):
                true_pos_df = pd.DataFrame(data=np.hstack(
                    (labels_idx[evnt][map_labels_preds[p][evnt] > -1][..., None], 
                        peak_probs_idx[p][evnt][map_preds_labels[p][evnt] > -1][..., None])), columns=["ref", "pred"])
                true_pos_df = pd.concat((true_pos_df, pd.DataFrame(data=[[evnt] for _ in range(len(true_pos_df))], columns=["event_type"])), axis=1, ignore_index=True)
                false_neg_df = pd.DataFrame(data=np.hstack(
                    (labels_idx[evnt][map_labels_preds[p][evnt] <= -1][..., None],
                        np.array([np.nan for _ in labels_idx[evnt][map_labels_preds[p][evnt] <= -1]])[..., None])), columns=["ref", "pred"])
                false_neg_df = pd.concat((false_neg_df, pd.DataFrame(data=[[evnt] for _ in range(len(false_neg_df))], columns=["event_type"])), axis=1, ignore_index=True)
                false_pos_df = pd.DataFrame(data=np.hstack(
                    (np.array([np.nan for _ in peak_probs_idx[p][evnt][map_preds_labels[p][evnt] <= -1]])[..., None],
                        peak_probs_idx[p][evnt][map_preds_labels[p][evnt] <= -1][..., None])), columns=["ref", "pred"])
                false_pos_df = pd.concat((false_pos_df, pd.DataFrame(data=[[evnt] for _ in range(len(false_pos_df))], columns=["event_type"])), axis=1, ignore_index=True)
                events_df = pd.concat((true_pos_df, false_neg_df, false_pos_df), axis=0, ignore_index=True)
                results_df = pd.concat((results_df, events_df), axis=0, ignore_index=True)
            results_df.columns = ["ref", "pred", "event_type"]
            results_df["sub_id"] = [os.path.split(test_file_path)[-1][:8] for _ in range(len(results_df))]
            results_df["cohort"] = [subjects_df[subjects_df["sub_id"]==os.path.split(test_file_path)[-1][:8]]['cohort'].values[0] for _ in range(len(results_df))]
            results_df["file_path"] = [test_file_path for _ in range(len(results_df))]
            results_df = results_df[["file_path", "sub_id", "cohort", "event_type", "ref", "pred"]]
            
            # Add to overal output dataframes
            output_dfs[p] = pd.concat((output_dfs[p], results_df), axis=0, ignore_index=True)
        
        # Loop over the dataframe
        for iter, df in enumerate(output_dfs):
            time_diff = df["pred"] - df["ref"]
            df["time_diff"] = time_diff
            
            # Write to file
            df.to_csv(f"output_model-{iter+2:02d}.tsv", sep="\t", header=True, index=False)

        # Visualize
        visualize = visualize
        if visualize:
            fig, axs = plt.subplots(4, 2, sharex=True, gridspec_kw={"height_ratios": [3, 3, 1, 1]})
            axs[0, 0].plot(np.arange(len(data)), data[:, 0:3], lw=2, alpha=0.4)
            axs[0, 0].plot(np.arange(len(data)), np.linalg.norm(data[:, 0:3], axis=1), lw=1, c=(0, 0, 0))
            axs[0, 1].plot(np.arange(len(data)), data[:, 6:9], lw=2, alpha=0.4)
            axs[0, 1].plot(np.arange(len(data)), np.linalg.norm(data[:, 6:9], axis=1), lw=1, c=(0, 0, 0))
            axs[1, 0].plot(np.arange(len(data)), data[:, 3:6], lw=2, alpha=0.4)
            axs[1, 1].plot(np.arange(len(data)), data[:, 9:], lw=2, alpha=0.4)
            for evnt in list(labels_idx.keys()):
                for idx_evnt in labels_idx[evnt]:
                    if evnt == "ICL":
                        axs[0,0].axvline(idx_evnt, c="r", ls='--', alpha=0.3)
                        axs[1,0].axvline(idx_evnt, c="r", ls='--', alpha=0.3)
                        axs[2,0].axvline(idx_evnt, c="r", ls='--')
                    elif evnt == "FCL":
                        axs[0,0].axvline(idx_evnt, c="r", ls=':', alpha=0.3)
                        axs[1,0].axvline(idx_evnt, c="r", ls=':', alpha=0.3)
                        axs[3,0].axvline(idx_evnt, c="r", ls=':')
                    elif evnt == "ICR":
                        axs[0,1].axvline(idx_evnt, c="r", ls='--', alpha=0.3)
                        axs[1,1].axvline(idx_evnt, c="r", ls='--', alpha=0.3)
                        axs[2,1].axvline(idx_evnt, c="r", ls='--')
                    else:  # evnt == "FCR":
                        axs[0,1].axvline(idx_evnt, c="r", ls=':', alpha=0.3)
                        axs[1,1].axvline(idx_evnt, c="r", ls=':', alpha=0.3)
                        axs[3,1].axvline(idx_evnt, c="r", ls=':')
                    
            for preds in predictions:
                axs[2, 0].plot(np.arange(len(preds[0])), preds[0][:, 1], lw=2)
                axs[3, 0].plot(np.arange(len(preds[0])), preds[0][:, 2], lw=2)
                axs[2, 1].plot(np.arange(len(preds[0])), preds[0][:, 3], lw=2)
                axs[3, 1].plot(np.arange(len(preds[0])), preds[0][:, 4], lw=2)
            
            for p in range(len(peak_probs_idx)):
                axs[2,0].plot(peak_probs_idx[p]["ICL"], predictions[p][0][peak_probs_idx[p]["ICL"], 1], ls="None", marker="o")
            for p in range(len(peak_probs_idx)):
                axs[3,0].plot(peak_probs_idx[p]["FCL"], predictions[p][0][peak_probs_idx[p]["FCL"], 2], ls="None", marker="o")
            for p in range(len(peak_probs_idx)):
                axs[2,1].plot(peak_probs_idx[p]["ICR"], predictions[p][0][peak_probs_idx[p]["ICR"], 3], ls="None", marker="o")
            for p in range(len(peak_probs_idx)):
                axs[3,1].plot(peak_probs_idx[p]["FCR"], predictions[p][0][peak_probs_idx[p]["FCR"], 4], ls="None", marker="o")
                
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
    main(visualize=False)