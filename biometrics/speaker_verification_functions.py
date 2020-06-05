import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from numpy.linalg import inv

import utils


def gmm_global_threshold(features, label_dict):
    runs_eer = []
    runs_hter = []
    
    for experiment_i in range(5):
        train_set, development_set, test_set, train_dev_set = utils.shuffle_split_data(features, label_dict)
        train_x, train_y, development_x, development_y, test_x, test_y, train_dev_x, train_dev_y = utils.get_datasets(train_set, development_set, test_set, train_dev_set)
        nb_of_components = 11
        
        all_gmms = build_GMMs(train_set, nb_of_components, label_dict)
        dist_matrix = compute_dist_matrix(development_x, all_gmms, label_dict)
        cur_eer, cur_threshold = compute_eer(dist_matrix, development_y, label_dict)
        runs_eer.append(cur_eer)
        
        if experiment_i == 0:
            utils.plot_scores(dist_matrix, development_y, "First Section", "e1", label_dict)
            frr_list, far_list, threshold_list = compute_frr_far_list(dist_matrix, development_y, label_dict)
            utils.plot_far_frr(frr_list, far_list, threshold_list, "First Section", "e1")
        
        print(f"Threshold:{cur_threshold}")
        
        all_gmms = build_GMMs(train_dev_set, nb_of_components, label_dict)
        dist_matrix = compute_dist_matrix(test_x, all_gmms, label_dict)
        cur_frr, cur_far = compute_frr_far(dist_matrix, test_y, cur_threshold, label_dict)
        cur_hter = (cur_frr + cur_far) / 2
        runs_hter.append(cur_hter)
        
        print(f"EERs:{np.array(runs_eer)}, HTERs:{np.array(runs_hter)}")
        
        print(f"Average EER:{np.array(runs_eer).mean():.4f}, std:{np.array(runs_eer).std():.4f}")
        print(f"Average HTER:{np.array(runs_hter).mean():.4f}, std:{np.array(runs_hter).std():.4f}")


def gmm_client_specific(features, label_dict):
    runs_eer = []
    runs_hter = []

    for _ in range(5):
        train_set, development_set, test_set, train_dev_set = utils.shuffle_split_data(features, label_dict)
        train_x, train_y, development_x, development_y, test_x, test_y, train_dev_x, train_dev_y = utils.get_datasets(train_set, development_set, test_set, train_dev_set)
        nb_of_components = 11

        all_gmms = build_GMMs(train_set, nb_of_components, label_dict)
        dist_matrix = compute_dist_matrix(development_x, all_gmms, label_dict)
        cur_eers, cur_thresholds = compute_eer_client_threshold(dist_matrix, development_y, label_dict)
        runs_eer.append(np.mean(cur_eers))

        print(f"Client thresholds:{np.array(cur_thresholds)}")

        all_gmms = build_GMMs(train_dev_set, nb_of_components, label_dict)
        dist_matrix = compute_dist_matrix(test_x, all_gmms, label_dict)

        client_hters = []
        for i in range(len(label_dict)):
            cur_dm = dist_matrix[:,i]
            genuine_indexes = (test_y == i)
            client_threshold = cur_thresholds[i]
            cur_frr, cur_far = compute_frr_far_client(cur_dm, genuine_indexes, client_threshold)
            client_hters.append((cur_frr + cur_far) / 2)

        cur_hter = np.mean(client_hters)
        runs_hter.append(cur_hter)

        print(f"EERs:{np.array(runs_eer)}, HTERs:{np.array(runs_hter)}")

    print(f"Average EER:{np.array(runs_eer).mean():.4f}, std:{np.array(runs_eer).std():.4f}")
    print(f"Average HTER:{np.array(runs_hter).mean():.4f}, std:{np.array(runs_hter).std():.4f}")


def ubm(features, label_dict):
    runs_eer = []
    runs_hter = []
    
    for experiment_i in range(5):
        train_set, development_set, test_set, train_dev_set = utils.shuffle_split_data(features, label_dict)
        train_x, train_y, development_x, development_y, test_x, test_y, train_dev_x, train_dev_y = utils.get_datasets(train_set, development_set, test_set, train_dev_set)
        nb_of_components = 11
        
        nb_of_components_background = 15
        
        all_gmms = build_GMMs(train_set, nb_of_components, label_dict)
        all_ubms = build_UBMs(train_set, nb_of_components_background, label_dict)
        dist_matrix = compute_dist_matrix_with_ubm(development_x, all_gmms, all_ubms, label_dict)
        cur_eers, cur_thresholds = compute_eer_client_threshold(dist_matrix, development_y, label_dict)
        runs_eer.append(np.mean(cur_eers))
        
        if experiment_i == 0:
            utils.plot_scores(dist_matrix, development_y, "Second Section", "e2", label_dict)
            frr_list, far_list, threshold_list = compute_frr_far_list(dist_matrix, development_y, label_dict)
            utils.plot_far_frr(frr_list, far_list, threshold_list, "Second Section", "e2")
        
        print(f"Client thresholds:{np.array(cur_thresholds)}")
        
        all_gmms = build_GMMs(train_dev_set, nb_of_components, label_dict)
        all_ubms = build_UBMs(train_dev_set, nb_of_components_background, label_dict)
        dist_matrix = compute_dist_matrix_with_ubm(test_x, all_gmms, all_ubms, label_dict)
        
        client_hters = []
        for i in range(len(label_dict)):
            cur_dm = dist_matrix[:,i]
            genuine_indexes = (test_y == i)
            client_threshold = cur_thresholds[i]
            cur_frr, cur_far = compute_frr_far_client(cur_dm, genuine_indexes, client_threshold)
            client_hters.append((cur_frr + cur_far) / 2)
        
        cur_hter = np.mean(client_hters)
        runs_hter.append(cur_hter)
        
        print(f"EERs:{np.array(runs_eer)}, HTERs:{np.array(runs_hter)}")
    
    print(f"Average EER:{np.array(runs_eer).mean():.4f}, std:{np.array(runs_eer).std():.4f}")
    print(f"Average HTER:{np.array(runs_hter).mean():.4f}, std:{np.array(runs_hter).std():.4f}")


def build_GMMs(dataset, nb_of_components, label_dict):
    return [GaussianMixture(n_components=nb_of_components)
            .fit(dataset[dataset["Label"] == subject_i].drop('Label', axis=1).to_numpy()) 
            for subject_i in range(len(label_dict))]


def build_UBMs(dataset, nb_of_components, label_dict):
    return [GaussianMixture(n_components=nb_of_components)
            .fit(dataset[dataset["Label"] != subject_i].drop('Label', axis=1).to_numpy()) 
            for subject_i in range(len(label_dict))]


def compute_dist_matrix(dataset_x, all_gmms, label_dict):
    dist_matrix = np.zeros(shape=(len(dataset_x), len(label_dict)))
    for subject_i in range(len(dataset_x)):
        cur_subject = dataset_x[subject_i]
        class_distances = []
        for class_i in range(len(label_dict)):
            class_gmm = all_gmms[class_i]
            min_dist = np.inf
            for gmm_comp in range(class_gmm.n_components):
                cur_dist = distance.mahalanobis(cur_subject, class_gmm.means_[gmm_comp], inv(class_gmm.covariances_)[gmm_comp])
                if cur_dist < min_dist:
                    min_dist = cur_dist
            class_distances.append(min_dist)
        dist_matrix[subject_i,:] = class_distances
    return dist_matrix


def compute_dist_matrix_with_ubm(dataset_x, all_gmms, all_ubms, label_dict):
    dist_matrix = np.zeros(shape=(len(dataset_x), len(label_dict)))
    for subject_i in range(len(dataset_x)):
        cur_subject = dataset_x[subject_i]
        class_distances = []
        for class_i in range(len(label_dict)):
            class_gmm = all_gmms[class_i]
            min_dist = np.inf
            for gmm_comp in range(class_gmm.n_components):
                cur_dist = distance.mahalanobis(cur_subject, class_gmm.means_[gmm_comp], inv(class_gmm.covariances_)[gmm_comp])
                if cur_dist < min_dist:
                    min_dist = cur_dist
            class_ubm = all_ubms[class_i]
            ubm_min_dist = np.inf
            for gmm_comp in range(class_ubm.n_components):
                cur_dist = distance.mahalanobis(cur_subject, class_ubm.means_[gmm_comp], inv(class_ubm.covariances_)[gmm_comp])
                if cur_dist < ubm_min_dist:
                    ubm_min_dist = cur_dist
            class_distances.append(min_dist + (1 / ubm_min_dist))
        dist_matrix[subject_i,:] = class_distances
    return dist_matrix


def compute_frr_far(dist_matrix, class_labels, threshold, label_dict):
    false_rejected_list = []
    false_accepted_list = []
    for i in range(len(label_dict)):
        accepted = np.zeros_like(dist_matrix)
        accepted[dist_matrix < threshold] = True
        accepted = accepted[:,i]
        genuine_indexes = (class_labels == i)
        false_rejected = 1 - (accepted[genuine_indexes == True].sum() / (len(accepted[genuine_indexes == True])))
        false_accepted = accepted[genuine_indexes == False].sum() / len(accepted[genuine_indexes == False])
        false_rejected_list.append(false_rejected)
        false_accepted_list.append(false_accepted)
    threshold_frr = np.mean(false_rejected_list)
    threshold_far = np.mean(false_accepted_list)
    return threshold_frr, threshold_far


def compute_frr_far_client(dist_matrix, genuine_indexes, threshold):
    accepted = np.zeros_like(dist_matrix)
    accepted[dist_matrix < threshold] = True
    false_rejected = 1 - (accepted[genuine_indexes == True].sum() / (len(accepted[genuine_indexes == True])))
    false_accepted = accepted[genuine_indexes == False].sum() / len(accepted[genuine_indexes == False])
    return false_rejected, false_accepted


def compute_eer(dist_matrix, dataset_y, label_dict):
    min_difference = np.inf
    eer_threshold = 0
    for threshold in np.arange(4.5, 5.5, 0.005):
        cur_frr, cur_far = compute_frr_far(dist_matrix, dataset_y, threshold, label_dict)
        if np.abs(cur_frr - cur_far) <= min_difference:
            min_difference = np.abs(cur_frr - cur_far)
            eer_threshold = threshold
        else:
            break
    return (cur_frr + cur_far) / 2, eer_threshold


def compute_frr_far_list(dist_matrix, dataset_y, label_dict):
    frr_list = []
    far_list = []
    threshold_list = []
    for threshold in np.arange(2, 8, 0.04):
        cur_frr, cur_far = compute_frr_far(dist_matrix, dataset_y, threshold, label_dict)
        frr_list.append(cur_frr)
        far_list.append(cur_far)
        threshold_list.append(threshold)
    return frr_list, far_list, threshold_list


def compute_eer_client_threshold(dist_matrix, dataset_y, label_dict):
    eer_thresholds = []
    eers = []
    for i in range(len(label_dict)):
        cur_dm = dist_matrix[:,i]
        min_difference = np.inf
        eer_threshold = 0
        genuine_indexes = (dataset_y == i)
        for threshold in np.arange(4.5, 6.0, 0.005):
            cur_frr, cur_far = compute_frr_far_client(cur_dm, genuine_indexes, threshold)
            if np.abs(cur_frr - cur_far) <= min_difference:
                min_difference = np.abs(cur_frr - cur_far)
                eer_threshold = threshold
            else:
                break
        eer_thresholds.append(eer_threshold)
        eers.append((cur_frr + cur_far) / 2)
    return eers, eer_thresholds
