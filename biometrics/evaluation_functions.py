import numpy as np


def compute_frr_far(data, data_class_labels, threshold):
    false_rejected_list = []
    false_accepted_list = []
    for i in range(len(data)):
        accepted = np.zeros_like(data[i])
        accepted[data[i] > threshold] = 1
        genuine_indexes = np.array(data_class_labels == data_class_labels[i])
        false_rejected = 1 - (accepted[genuine_indexes == True].sum() / (len(accepted[genuine_indexes == True]) - 1))
        false_accepted = accepted[genuine_indexes == False].sum() / len(accepted[genuine_indexes == False])
        false_rejected_list.append(false_rejected)
        false_accepted_list.append(false_accepted)
    cur_frr = np.array(false_rejected_list).mean() * 100
    cur_far = np.array(false_accepted_list).mean() * 100
    return cur_frr, cur_far


def compute_frr_at_far_points(list_frr, list_far):
    frr_at_01 = None
    frr_at_1 = None
    frr_at_10 = None
    list_frr_r = list(reversed(list_frr))
    list_far_r = list(reversed(list_far))
    for i in range(len(list_frr_r)):
        if list_far_r[i] >= 0.1 and frr_at_01 is None:
            frr_at_01 = list_frr_r[i]
        if list_far_r[i] >= 1 and frr_at_1 is None:
            frr_at_1 = list_frr_r[i]
        if list_far_r[i] >= 10 and frr_at_10 is None:
            frr_at_10 = list_frr_r[i]
            break
    return frr_at_01, frr_at_1, frr_at_10


def find_eer(data, data_class_labels):
    frr_list = []
    far_list = []
    min_difference = np.inf
    eer_threshold = 0
    eer_threshold_index = 0
    count = 0
    for threshold in np.arange(0, 1, 0.001):
        cur_frr, cur_far = compute_frr_far(data, data_class_labels, threshold)
        frr_list.append(cur_frr)
        far_list.append(cur_far)
        if np.abs(cur_frr - cur_far) <= min_difference:
            min_difference = np.abs(cur_frr - cur_far)
            eer_threshold = threshold
            eer_threshold_index = count
        count += 1
    return frr_list, far_list, eer_threshold_index, eer_threshold


def get_scores(data, data_class_labels):
    genuine_scores_list = None
    impostor_scores_list = None
    for i in range(len(data)):
        genuine_indexes = np.array(data_class_labels == data_class_labels[i])
        genuine_scores = data[i][genuine_indexes == True]
        genuine_scores = genuine_scores[~np.isnan(genuine_scores)]
        impostor_scores = data[i][genuine_indexes == False]
        genuine_scores_list = genuine_scores if genuine_scores_list is None else np.append(genuine_scores_list, genuine_scores)
        impostor_scores_list = impostor_scores if impostor_scores_list is None else np.append(impostor_scores_list, impostor_scores)
    return genuine_scores_list, impostor_scores_list
