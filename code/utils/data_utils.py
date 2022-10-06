import numpy as np
import pandas as pd
import os, random
import tensorflow as tf

SAMPLING_FREQUENCY = 100.0  # Hz
# DATASET_DIR = "/gxfs_work1/cau/sukne964/Datasets/Mobilised-D/rawdata/MicroWB"
# DATASET_DIR = "/home/robbin/Datasets/Mobilise-D/rawdata/MicroWB"
DATASET_DIR = "D:\\Datasets\\Mobilise-D\\rawdata\\MicroWB"
MAP_INDEX_EVENTS = {idx: event_name for idx, event_name in enumerate(["null", "ICL", "FCL", "ICR", "FCR"])}
MAP_INDEX_PHASES = {idx: gait_phase for idx, gait_phase in enumerate(["null", "LeftSwing", "RightSwing"])}

def set_seed(seed: int = None):
    if seed is None:
        seed = np.random.choice(2**32)
    random.seed(seed)
    np.random.seed(seed)
    return

def split_train_test(path=DATASET_DIR, test_size=0.2, seed=None):
    # Set seed, for reproducible sets
    set_seed(seed)

    # Get demographics dataframe
    subjects_df = pd.read_csv(os.path.join(path, "subjects.tsv"), sep="\t", header=0)

    # Get list of valid subject ids
    sub_ids = [s for s in os.listdir(path) if s.startswith("sub-") and os.path.isdir(os.path.join(path, s))]

    # Sort subjects by cohort
    subs_by_cohort = {c: [] for c in subjects_df["cohort"].unique()}
    for idx_sub, sub_id in enumerate(sub_ids):
        coh = subjects_df[subjects_df["sub_id"]==sub_id]["cohort"].values[0]
        subs_by_cohort[coh].append(sub_id)

    # Divide subjects over test, val and train set
    test_sub_ids, val_sub_ids, train_sub_ids = [], [], []
    for cohort, subs in subs_by_cohort.items():
        N = int(len(subs) * test_size)
        random.shuffle(subs)
        test_sub_ids += subs[:N]
        val_sub_ids += subs[N:2*N]
        train_sub_ids += subs[2*N:]

    # Lookup available files
    train_files, val_files, test_files = [], [], []
    for s in train_sub_ids:
        file_names = [os.path.join(DATASET_DIR, s, f) for f in os.listdir(os.path.join(DATASET_DIR, s)) if f.endswith(".npy")]
        train_files += file_names
    for s in val_sub_ids:
        file_names = [os.path.join(DATASET_DIR, s, f) for f in os.listdir(os.path.join(DATASET_DIR, s)) if f.endswith(".npy")]
        val_files += file_names
    for s in test_sub_ids:
        file_names = [os.path.join(DATASET_DIR, s, f) for f in os.listdir(os.path.join(DATASET_DIR, s)) if f.endswith(".npy")]
        test_files += file_names
    return train_files, val_files, test_files


def _get_event_labels(annotations, labels_as="binary", tolerance=0):
    # Get the number of unique classes
    num_classes = len(MAP_INDEX_EVENTS)

    # Get one-hot encoded labels
    labels = tf.keras.utils.to_categorical(annotations, num_classes=num_classes)

    # If no tolerance is allowed, we are all set
    if tolerance == 0:
        return labels

    # Get length of data sequence a.k.a. number of time steps
    num_time_steps = len(annotations)

    # Parse annotations
    if labels_as == "binary":

        # Create mask
        mask = np.zeros((num_time_steps + 2 * tolerance + 1, num_classes, 2 * tolerance + 1))
        for i in range(2 * tolerance + 1):
            mask[i:i+num_time_steps,:,i] = labels
        mask = np.any(mask, axis=-1, keepdims=False).astype("float32")
        mask[:,0] = np.logical_not(np.any(mask[:,1:], axis=-1, keepdims=False)).astype("float32")
        labels = mask[tolerance:tolerance+num_time_steps,:]
    elif labels_as == "gauss":
        h = tolerance
        gs = np.exp((-4*np.log(2)*(np.arange(-2*h, 2*h+1)**2))/(h**2))

        for i in range(1, num_classes):
            for j in np.argwhere(labels[:, i]==1)[:,0]:
                a = j - len(gs)//2
                b = j + len(gs)//2 + 1
                if (a >= 0) and (b < num_time_steps):
                    labels[a:b, i] = gs
                elif (a < 0) and (b < num_time_steps):
                    labels[:b, i] = gs[-a:]
                elif (a >= 0)  and (b >= num_time_steps):
                    labels[a:, i] = gs[:len(np.arange(a, num_time_steps))]
                else:
                    raise ValueError(f"Invalid indices for mapping a Gaussian.")
    return labels

def _get_phase_labels(annotations):
    # Preallocate labels array
    num_classes = len(MAP_INDEX_PHASES)
    labels = np.zeros((len(annotations), num_classes-1))

    # Get indices from events
    idxs_events = {evnt: np.argwhere(annotations[:,0]==idx)[:,0] for idx, evnt in MAP_INDEX_EVENTS.items() if evnt != "null"}

    # Parse left swings
    if idxs_events["ICL"][0] < idxs_events["FCL"][0]:
        labels[:idxs_events["ICL"][0],0] = 1
    if idxs_events["FCL"][-1] > idxs_events["ICL"][-1]:
        labels[idxs_events["FCL"][-1]:,0] = 1
    for i in range(len(idxs_events["FCL"])):
        j = np.argwhere(idxs_events["ICL"] > idxs_events["FCL"][i])[:,0]
        if len(j) > 0: labels[idxs_events["FCL"][i]:idxs_events["ICL"][j[0]],0] = 1

    # Parse right swings
    if idxs_events["ICR"][0] < idxs_events["FCL"][0]:
        labels[:idxs_events["ICR"][0],1] = 1
    if idxs_events["FCR"][-1] > idxs_events["ICR"][-1]:
        labels[idxs_events["FCR"][-1]:,1] = 1
    for i in range(len(idxs_events["FCR"])):
        j = np.argwhere(idxs_events["ICR"] > idxs_events["FCR"][i])[:,0]
        if len(j) > 0: labels[idxs_events["FCR"][i]:idxs_events["ICR"][j[0]],1] = 1

    # Add labels for null class
    labels = np.hstack((np.logical_not(np.any(labels, axis=-1, keepdims=True)), labels))
    return labels.astype("float32")

def _get_labels(annotations, mode="gait_events", task="classification", labels_as="binary", tolerance=0):
    if mode == "gait_events":
        labels = _get_event_labels(annotations, labels_as=labels_as, tolerance=tolerance)
    elif mode == "gait_phases":
        labels = _get_phase_labels(annotations)
    else:
        raise ValueError(f"Unrecognized mode `{mode:s}`, please choose from `gait_events` or `gait_phases`.")
    return labels

def _load_data(file_path, 
        tracked_points=["LeftFoot", "RightFoot"],
        incl_magn: bool = False,
        mode: str = "gait_events",
        task: str = "classification",
        labels_as: str = "binary",
        tolerance: int = 0):
    if len(tracked_points)==0: return

    # Load data and annotations as numpy array
    with open(file_path, 'rb') as infile:
        data_annotations = np.load(infile)

    # Load channel names in pandas DataFrame
    with open(file_path[:-29]+"_channels.tsv", 'rb') as tsv_file:
        channels_df = pd.read_csv(tsv_file, sep="\t", header=0)

    # Split data and annotations
    data = data_annotations[:,:-1]
    annotations = data_annotations[:,-1:]

    # Select only data from tracked points
    if incl_magn:
        idx_cols = [idx for idx, tp in enumerate(channels_df["tracked_point"]) if tp in tracked_points]
    else:
        idx_cols = [idx for idx, tp in enumerate(channels_df["tracked_point"]) if (tp in tracked_points) and (channels_df["type"][idx] != "Mag")]
    data = data[:, idx_cols]

    # Get labels from annotations
    labels = _get_labels(annotations, mode=mode, task=task, labels_as=labels_as, tolerance=tolerance)
    return data, labels
