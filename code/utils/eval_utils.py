import numpy as np
import pandas as pd
from utils.data_utils import SAMPLING_FREQUENCY

def match_events(annotated_idx, preds_idx, tolerance=0, do_print=False):
    """
    Match the discrete-time indices from the reference system,
    `annotated_idx`, with the indices as detected by the neural
    network, `preds_idx`.

    Parameters
    ----------
    annotated_idx : ndarray
        The indices corresponding to the reference gait events.
    preds_idx : ndarray
        The indices corresponding to the predicted gait events.
    tolerance : int
        The number of samples that the annotated and predicted
        indices can be away from one another.
    do_print : bool
        Whether to print the matched indices to the terminal.
    """

    # Compute the pair-wise absolute differences
    abs_diff = np.abs(np.subtract.outer(annotated_idx, preds_idx))

    # Map each annotated event to the nearest predicted event
    map_annotations_preds = np.argmin(abs_diff, axis=1)
    
    # Map each predicted event to the nearest annotated event
    map_preds_annotations = np.argmin(abs_diff, axis=0)

    # There can only be unique mappings
    for m in np.unique(map_annotations_preds):
        indices = np.argwhere(map_annotations_preds==m)[:,0]
        
        # If multiple labelled events map to the same predicted event
        if len(indices) > 1:
            
            # Loop over the indices
            for i in indices:
                if map_preds_annotations[m] == i:
                    continue # to next of the indices
                else:
                    map_preds_annotations[np.argwhere(map_preds_annotations==i)[:,0]] = -1
                    map_annotations_preds[i] = -1

    # Do the same for mappings from predicted to labelled events
    for m in np.unique(map_preds_annotations):
        indices = np.argwhere(map_preds_annotations==m)[:,0]
        if len(indices) > 1:
            for i in indices:
                if map_annotations_preds[m] == i:
                    continue # to next
                else:
                    map_annotations_preds[np.argwhere(map_annotations_preds==i)[:,0]] = -1
                    map_preds_annotations[i] = -1
    
    # Additional check
    if len(map_annotations_preds[map_annotations_preds > -1]) > len(map_preds_annotations[map_preds_annotations > -1]):
        map_annotations_preds[map_annotations_preds > -1] = np.where(map_preds_annotations[map_annotations_preds[map_annotations_preds > -1]] > -1, 
                                                                     map_annotations_preds[map_annotations_preds > -1], # keep the current mapping
                                                                     -1) # set to -1
    elif len(map_annotations_preds[map_annotations_preds > -1]) < len(map_preds_annotations[map_preds_annotations > -1]):
        map_preds_annotations[map_preds_annotations > -1] = np.where(map_annotations_preds[map_preds_annotations[map_preds_annotations > -1]] > -1,
                                                                     map_preds_annotations[map_preds_annotations > -1], # keep the current mapping
                                                                     -1) # set to -1
    else:
        pass # nothing to do
    
    # Filter those event that have a valid counterpart
    annotated_idx_valid = annotated_idx[map_preds_annotations[map_preds_annotations>=0]]
    preds_idx_valid = preds_idx[map_annotations_preds[map_annotations_preds>=0]]
    
    # Get absolute differences for the matched (valid) events
    abs_diff_valid = np.abs((preds_idx_valid - annotated_idx_valid))
    
    # Update the mappings
    indices = np.argwhere(map_annotations_preds > -1)[:,0]
    map_annotations_preds[indices] = np.where(abs_diff_valid <= tolerance, map_annotations_preds[indices], -1)
    indices = np.argwhere(map_preds_annotations > -1)[:,0]
    map_preds_annotations[indices] = np.where(abs_diff_valid <= tolerance, map_preds_annotations[indices], -1)
    
    if do_print:
        print(
            np.hstack((
                annotated_idx[map_annotations_preds > -1][..., None],
                preds_idx[map_preds_annotations > -1][..., None],
                (preds_idx[map_preds_annotations > -1] - annotated_idx[map_annotations_preds > -1])[..., None]
            ))
        )
    return map_annotations_preds, map_preds_annotations

def get_stride_params(labels_idx, 
                      preds_idx,
                      map_labels_preds,
                      map_preds_labels):
    
    # Initialize placeholders to store results
    results_dict = {"gait_phase": [], "side": [], "ref": [], "pred": []}
    
    # Left strides
    for i in range(1, len(labels_idx["ICL"])):
        
        # If both the current and previous event
        # have a valid mapping (i.e., true positive)
        if (map_labels_preds["ICL"][i] > -1) and (map_labels_preds["ICL"][i-1] > -1):

            # Find the final contact in between
            f = np.argwhere((labels_idx["FCL"] > labels_idx["ICL"][i-1]) & (labels_idx["FCL"] < labels_idx["ICL"][i]))[:,0]
            if (len(f)==1) and (map_labels_preds["FCL"][f[0]] > -1):

                # Reference timings
                stance_time_ref = (labels_idx["FCL"][f[0]] - labels_idx["ICL"][i-1])/SAMPLING_FREQUENCY
                swing_time_ref = (labels_idx["ICL"][i] - labels_idx["FCL"][f[0]])/SAMPLING_FREQUENCY
                stride_time_ref = (labels_idx["ICL"][i] - labels_idx["ICL"][i-1])/SAMPLING_FREQUENCY
                
                # Predicted timings
                stance_time_pred = (preds_idx["FCL"][map_labels_preds["FCL"][f[0]]] - preds_idx["ICL"][map_labels_preds["ICL"][i-1]])/SAMPLING_FREQUENCY
                swing_time_pred = (preds_idx["ICL"][map_labels_preds["ICL"][i]] - preds_idx["FCL"][map_labels_preds["FCL"][f[0]]])/SAMPLING_FREQUENCY
                stride_time_pred = (preds_idx["ICL"][map_labels_preds["ICL"][i]] - preds_idx["ICL"][map_labels_preds["ICL"][i-1]])/SAMPLING_FREQUENCY
                
                # Add to placeholders
                results_dict["gait_phase"] += ["stance", "swing", "stride"]
                results_dict["side"] += ["left" for _ in range(3)]
                results_dict["ref"] += [stance_time_ref, swing_time_ref, stride_time_ref]
                results_dict["pred"] += [stance_time_pred, swing_time_pred, stride_time_pred]
                
    # Right strides
    for i in range(1, len(labels_idx["ICR"])):
        
        # If both the current and previous event
        # have a valid mapping (i.e., true positive)
        if (map_labels_preds["ICR"][i] > -1) and (map_labels_preds["ICR"][i-1] > -1):

            # Find the final contact in between
            f = np.argwhere((labels_idx["FCR"] > labels_idx["ICR"][i-1]) & (labels_idx["FCR"] < labels_idx["ICR"][i]))[:,0]
            if (len(f)==1) and (map_labels_preds["FCR"][f[0]] > -1):

                # Reference timings
                stance_time_ref = (labels_idx["FCR"][f[0]] - labels_idx["ICR"][i-1])/SAMPLING_FREQUENCY
                swing_time_ref = (labels_idx["ICR"][i] - labels_idx["FCR"][f[0]])/SAMPLING_FREQUENCY
                stride_time_ref = (labels_idx["ICR"][i] - labels_idx["ICR"][i-1])/SAMPLING_FREQUENCY
                
                # Predicted timings
                stance_time_pred = (preds_idx["FCR"][map_labels_preds["FCR"][f[0]]] - preds_idx["ICR"][map_labels_preds["ICR"][i-1]])/SAMPLING_FREQUENCY
                swing_time_pred = (preds_idx["ICR"][map_labels_preds["ICR"][i]] - preds_idx["FCR"][map_labels_preds["FCR"][f[0]]])/SAMPLING_FREQUENCY
                stride_time_pred = (preds_idx["ICR"][map_labels_preds["ICR"][i]] - preds_idx["ICR"][map_labels_preds["ICR"][i-1]])/SAMPLING_FREQUENCY
                
                # Add to placeholders
                results_dict["gait_phase"] += ["stance", "swing", "stride"]
                results_dict["side"] += ["right" for _ in range(3)]
                results_dict["ref"] += [stance_time_ref, swing_time_ref, stride_time_ref]
                results_dict["pred"] += [stance_time_pred, swing_time_pred, stride_time_pred]
    return pd.DataFrame(results_dict)