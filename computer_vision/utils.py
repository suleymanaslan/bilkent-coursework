import numpy as np
import cv2


def find_keypoints(img_list):
    sift = cv2.xfeatures2d.SIFT_create()
    
    keypoints = []
    descriptors = []
    img_keypoints = []
    
    for img in img_list:
        cur_keypoints, cur_descriptors = sift.detectAndCompute(img, None)
        keypoints.append(cur_keypoints)
        descriptors.append(cur_descriptors)
        img_keypoints.append(cv2.drawKeypoints(img, cur_keypoints, None))
    
    return keypoints, descriptors, img_keypoints


def normalize_descriptor(descriptor):
    return (descriptor - np.min(descriptor)) / (np.max(descriptor) - np.min(descriptor))


def create_distance_matrix(descriptor1, descriptor2):
    distance_matrix = np.zeros([descriptor1.shape[0], descriptor2.shape[0]])
    for i in range(descriptor1.shape[0]):
        dist = np.linalg.norm(descriptor2 - descriptor1[i], axis=1)
        distance_matrix[i,:] = dist
    return distance_matrix


def get_eligible_matches(distance_matrix, descriptors, nndr_threshold):
    min_key_points_val = np.min((descriptors[0].shape[0], descriptors[1].shape[0]))
    min_key_points_ind = 1
    
    min_idx = distance_matrix.argmin(axis=min_key_points_ind)
    min_vals = distance_matrix.min(axis=min_key_points_ind)
    if min_key_points_ind == 1:
        distance_matrix[np.arange(len(distance_matrix)), min_idx] = np.inf
    elif min_key_points_ind == 0:
        distance_matrix[min_idx, np.arange(descriptors[1].shape[0])] = np.inf
    min_idx2 = distance_matrix.argmin(axis=min_key_points_ind)
    min_vals2 = distance_matrix.min(axis=min_key_points_ind)
    
    min_distance = np.concatenate([np.expand_dims(min_vals, axis=0), np.expand_dims(min_vals2, axis=0)], axis=0)
    min_indices = np.concatenate([np.expand_dims(min_idx, axis=0), np.expand_dims(min_idx2, axis=0)], axis=0)
    
    all_matches = [[] for _ in range(len(min_idx))]
    for i in range(len(min_vals)):
        for j in range(2):
            all_matches[i].append(cv2.DMatch(i, min_indices[j, i], min_distance[j, i]))
    
    eligible_matches = []
    for i in range(len(min_vals)):
        if all_matches[i][0].distance < nndr_threshold * all_matches[i][1].distance:
            eligible_matches.append(all_matches[i][0])
    
    return eligible_matches


def find_matches(descriptors, use_nndr=True, nndr_threshold=0.35, number_of_matches=100):
    if use_nndr:
        distance_matrix = create_distance_matrix(descriptors[0], descriptors[1])
        eligible_matches = get_eligible_matches(distance_matrix, descriptors, nndr_threshold)
        matched_points = [(eligible_matches[i].queryIdx, eligible_matches[i].trainIdx) for i in range(len(eligible_matches))]
        matches1to2 = [cv2.DMatch(i, i, 0) for i in range(len(eligible_matches))]
        
        return matched_points, matches1to2, eligible_matches
    
    else:
        pairwise_distances = create_distance_matrix(normalize_descriptor(descriptors[0]), normalize_descriptor(descriptors[1]))
        matched_points = []
        for _ in range(number_of_matches):
            min_index = np.argmin(pairwise_distances)
            first_point = min_index // pairwise_distances.shape[1]
            second_point = min_index % pairwise_distances.shape[1]
            matched_points.append((first_point, second_point))
            pairwise_distances[min_index // pairwise_distances.shape[1],:] = np.inf
            pairwise_distances[:,min_index % pairwise_distances.shape[1]] = np.inf
        matches1to2 = [cv2.DMatch(i, i, 0) for i in range(number_of_matches)]
        
        return matched_points, matches1to2
