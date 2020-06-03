import numpy as np
import cv2


def rectify_images(img_list, filtered_keypoints):
    img_size = (img_list[0].shape[1], img_list[0].shape[0])
    points_l = [np.array([kp.pt for kp in filtered_keypoints[0]]).reshape(-1, 1, 2).astype(np.float32)]
    points_r = [np.array([kp.pt for kp in filtered_keypoints[1]]).reshape(-1, 1, 2).astype(np.float32)]
    
    pattern_points = np.zeros((points_l[0].shape[0],3)).astype(np.float32)
    pattern_points[:,:-1] = points_l[0].reshape(-1, 2)
    _, cam_mat_l, dist_coeff_l, cam_mat_r, dist_coeff_r, rot_mat, trans_vec, _, _ = cv2.stereoCalibrate([pattern_points], points_l, points_r, 
                                                                                                        None, None, None, None, img_size, flags=cv2.CALIB_FIX_INTRINSIC)
    
    rect_trans_l, rect_trans_r, _, _, _, _, _ = cv2.stereoRectify(cam_mat_l, dist_coeff_l, cam_mat_r, dist_coeff_r, img_size, rot_mat, trans_vec, 
                                                                  flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.5)
    
    map_l_x, map_l_y = cv2.initUndistortRectifyMap(cam_mat_l, dist_coeff_l, rect_trans_l, cam_mat_l, img_size, cv2.CV_32FC1)
    map_r_x, map_r_y = cv2.initUndistortRectifyMap(cam_mat_r, dist_coeff_r, rect_trans_r, cam_mat_r, img_size, cv2.CV_32FC1)
    left_img_rectified = cv2.remap(img_list[0], map_l_x, map_l_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    right_img_rectified = cv2.remap(img_list[1], map_r_x, map_r_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
    
    return left_img_rectified, right_img_rectified


def compute_descriptors(img_l, img_r):    
    sift = cv2.xfeatures2d.SIFT_create()
    
    keypoints_l = []
    keypoints_r = []
    
    for i in range(img_l.shape[1]):
        for j in range(img_l.shape[0]):
            keypoints_l.append(cv2.KeyPoint(i, j, 1))
            keypoints_r.append(cv2.KeyPoint(i, j, 1))
    
    _, descriptors_l = sift.compute(img_l, keypoints_l)
    _, descriptors_r = sift.compute(img_r, keypoints_r)
    
    max_i = keypoints_l[-1].pt[0]
    max_j = keypoints_l[-1].pt[-1]
    
    return keypoints_r, descriptors_l, descriptors_r, max_i, max_j


def is_valid_keypoint(i, j, max_i, max_j):
    return (i >= 0 and i <= max_i and j >= 0 and j <= max_j)


def descriptor_distance(descriptor_l, descriptor_r):
    return np.sum(np.abs(descriptor_l - descriptor_r))


def match_point(keypoints_r, descriptors_l, descriptors_r, point_l, max_disp, max_i, max_j, compute_right_img=False):
    window_size = 7
    min_dist = np.inf
    match_ix = (0, 0)
    max_y = keypoints_r[-1].pt[1]
    i_range = np.arange(point_l[0]-max_disp,point_l[0]+max_disp+1).astype(np.int32)
    i_range = i_range[i_range >= 0]
    i_range = i_range[i_range <= keypoints_r[-1].pt[0]]
    j_range = range(-(window_size-1)//2, (window_size+1)//2)
    k_range = range(-(window_size-1)//2, (window_size+1)//2)
    for i in i_range:
        cur_dist = 0
        dist_count = 0
        for j in j_range:
            for k in k_range:
                if not is_valid_keypoint(point_l[0]+j, point_l[1]+k, max_i, max_j) or not is_valid_keypoint(i+j, point_l[1]+k, max_i, max_j):
                    continue
                if compute_right_img:
                    desc_l_ix = int((i+j)*(max_y+1))+int(point_l[1]+k)
                    desc_r_ix = int((point_l[0]+j)*(max_y+1))+int(point_l[1]+k)
                else:
                    desc_l_ix = int((point_l[0]+j)*(max_y+1))+int(point_l[1]+k)
                    desc_r_ix = int((i+j)*(max_y+1))+int(point_l[1]+k)
                cur_dist += descriptor_distance(descriptors_l[desc_l_ix], descriptors_r[desc_r_ix])
                dist_count += 1
        cur_dist / dist_count
        if cur_dist < min_dist:
            min_dist = cur_dist
            match_ix = (i, point_l[1])
    return match_ix
