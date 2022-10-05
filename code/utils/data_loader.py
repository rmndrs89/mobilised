from .data_utils import SAMPLING_FREQUENCY, DATASET_DIR, MAP_INDEX_EVENTS, MAP_INDEX_PHASES
from .data_utils import _load_data
import tensorflow as tf

def get_data_loader(file_paths, 
        win_len: int,
        step_len: int = None, 
        mode: str = "gait_events",
        task: str = "classification",
        labels_as: str = "binary",
        tolerance: int = 0):

    # Set step length, if not given
    step_len = win_len if step_len is None else step_len

    # Get mapping from index to label
    map_index_labels = MAP_INDEX_EVENTS if mode == "gait_events" else MAP_INDEX_PHASES

    # Create data loader
    def data_loader():

        # Loop over the files
        for idx_file, file_path in enumerate(file_paths):

            # Get current data and labels
            data, labels = _load_data(file_path, mode=mode, task=task, labels_as=labels_as, tolerance=tolerance)

            # Create sequences of equal length
            for idx in range(0, len(data) - win_len + 1, step_len):
                data_ = data[idx:idx+win_len, :]
                if task == "classification":
                    # For a classification task, labels is an array-like
                    # with shape (num_time_steps, num_classes) where
                    # num_classes includes the `null` class
                    labels_ = labels[idx:idx+win_len, :]
                elif task == "regression":
                    # For regression task, labels is a dictionary
                    # with a (num_time_steps, 1) array-like for each
                    # non-null output class
                    labels_ = {lbl: labels[idx:idx+win_len, idx_lbl][..., None] for idx_lbl, lbl in map_index_labels.items() if lbl != "null"}
                else:
                    raise ValueError(f"Invalid machine learning task defined (`{task:s}`), choose from `classification` and `regression`.")
                yield data_, labels_
    return data_loader

if __name__ == "__main__":
    get_data_loader()
