from cgi import test
from unittest import result
import numpy as np
import os, sys, random
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import tensorflow as tf
from tcn import TCN
from utils.data_utils import DATASET_DIR, SAMPLING_FREQUENCY, MAP_INDEX_EVENTS, MAP_INDEX_PHASES
from utils.data_utils import split_train_test, _load_data
from utils.eval_utils import match_events, get_stride_params
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
    gait_events_df = [pd.DataFrame() for _ in trained_models]
    stride_params_df = [pd.DataFrame() for _ in trained_models]

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
            for event_type in list(peak_probs_idx[p].keys()):  # v: {`ICL`, `FCL`, `ICR`, `FCR`}
                i = map_labels_index[event_type]  # which column to look at for the current event type
                idx_pks, _ = find_peaks(predictions[p][0][:, i], height=thr_proba, distance=int(SAMPLING_FREQUENCY//2))
                peak_probs_idx[p][event_type] = idx_pks
        
        # Map indices of peak probabilities to labels, and vice versa
        map_labels_preds, map_preds_labels = [{evnt: [] for evnt in list(labels_idx.keys())} for mdl in trained_models], [{evnt: [] for evnt in list(labels_idx.keys())} for mdl in trained_models]
        for p, probs in enumerate(peak_probs_idx):
            for evnt in list(probs.keys()):
                map_labels_preds[p][evnt], map_preds_labels[p][evnt] = match_events(labels_idx[evnt], peak_probs_idx[p][evnt], tolerance=int(SAMPLING_FREQUENCY//4))
            
        # For each trained model,
        for p, probs in enumerate(peak_probs_idx):
            
            # Parse stride parameters
            dd = get_stride_params(labels_idx=labels_idx, 
                                   preds_idx=peak_probs_idx[p],
                                   map_labels_preds=map_labels_preds[p],
                                   map_preds_labels=map_preds_labels[p])
            
            # Add subject id and cohort
            dd = pd.concat((pd.DataFrame(
                {
                    "file_path": [test_file_path for _ in range(len(dd))],
                    "sub_id": [os.path.split(test_file_path)[-1][:8] for _ in range(len(dd))],
                    "cohort": [subjects_df[(subjects_df["sub_id"]==os.path.split(test_file_path)[-1][:8])]["cohort"].values[0] for _ in range(len(dd))]
                }
            ), dd), axis=1, ignore_index=True)
            
            # Accummulate in overall gait phases dataframe
            dd.columns = ["file_path", "sub_id", "cohort", "gait_phase", "side", "ref", "pred"]
            stride_params_df[p] = pd.concat((stride_params_df[p], dd), axis=0, ignore_index=True)
            del dd
            
            # Parse individual gait events
            results_df = pd.DataFrame()
            
            # Loop over the event types
            for evnt in list(labels_idx.keys()):
                true_pos_df = pd.DataFrame(data=np.hstack(
                    (labels_idx[evnt][map_labels_preds[p][evnt] > -1][..., None], 
                        peak_probs_idx[p][evnt][map_preds_labels[p][evnt] > -1][..., None])), columns=["ref", "pred"])
                # true_pos_df = pd.concat((true_pos_df, pd.DataFrame(data=[[evnt] for _ in range(len(true_pos_df))], columns=["event_type"])), axis=1, ignore_index=True)
                true_pos_df = pd.concat((pd.DataFrame({
                    "gait_event": [evnt[:2] for _ in range(len(true_pos_df))],
                    "side": [evnt[-1] for _ in range(len(true_pos_df))]
                }), true_pos_df), axis=1, ignore_index=True)
                false_neg_df = pd.DataFrame(data=np.hstack(
                    (labels_idx[evnt][map_labels_preds[p][evnt] <= -1][..., None],
                        np.array([np.nan for _ in labels_idx[evnt][map_labels_preds[p][evnt] <= -1]])[..., None])), columns=["ref", "pred"])
                # false_neg_df = pd.concat((false_neg_df, pd.DataFrame(data=[[evnt] for _ in range(len(false_neg_df))], columns=["event_type"])), axis=1, ignore_index=True)
                false_neg_df = pd.concat((pd.DataFrame({
                    "gait_event": [evnt[:2] for _ in range(len(false_neg_df))],
                    "side": [evnt[-1] for _ in range(len(false_neg_df))]
                }), false_neg_df), axis=1, ignore_index=True)
                false_pos_df = pd.DataFrame(data=np.hstack(
                    (np.array([np.nan for _ in peak_probs_idx[p][evnt][map_preds_labels[p][evnt] <= -1]])[..., None],
                        peak_probs_idx[p][evnt][map_preds_labels[p][evnt] <= -1][..., None])), columns=["ref", "pred"])
                # false_pos_df = pd.concat((false_pos_df, pd.DataFrame(data=[[evnt] for _ in range(len(false_pos_df))], columns=["event_type"])), axis=1, ignore_index=True)
                false_pos_df = pd.concat((pd.DataFrame({
                    "gait_event": [evnt[:2] for _ in range(len(false_pos_df))],
                    "side": [evnt[-1] for _ in range(len(false_pos_df))]
                }), false_pos_df), axis=1, ignore_index=True)
                events_df = pd.concat((true_pos_df, false_neg_df, false_pos_df), axis=0, ignore_index=True)
                results_df = pd.concat((results_df, events_df), axis=0, ignore_index=True)
            results_df.columns = ["gait_event", "side", "ref", "pred"]
            results_df["sub_id"] = [os.path.split(test_file_path)[-1][:8] for _ in range(len(results_df))]
            results_df["cohort"] = [subjects_df[subjects_df["sub_id"]==os.path.split(test_file_path)[-1][:8]]['cohort'].values[0] for _ in range(len(results_df))]
            results_df["file_path"] = [test_file_path for _ in range(len(results_df))]
            results_df = results_df[["file_path", "sub_id", "cohort", "gait_event", "side", "ref", "pred"]]
            
            # Add to overal output dataframes
            gait_events_df[p] = pd.concat((gait_events_df[p], results_df), axis=0, ignore_index=True)
    
    # Write dataframe(s) to text file(s)
    for out in zip(gait_events_df, stride_params_df, ["02", "03"]):
        out[0].to_csv(f"gait_events_{out[2]}.tsv", sep="\t", header=True, index=False)
        out[1].to_csv(f"stride_params_{out[2]}.tsv", sep="\t", header=True, index=False)
    return

if __name__ == "__main__":
    main(visualize=False)