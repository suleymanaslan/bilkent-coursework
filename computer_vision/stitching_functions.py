import numpy as np
import cv2


def find_homography(eligible_matches, kp1, kp2, step_size, residual_stopping_threshold):
    init_translation = True
    max_iteration = 100000
    
    H = np.random.rand(3, 3).astype(np.float64)
    H[0, 0] = 1 + np.random.rand(1, 1)
    H[1, 1] = 1 + np.random.rand(1, 1)
    H[2, 2] = 1
    
    P0 = np.zeros([3, len(eligible_matches)])
    P1 = np.zeros([3, len(eligible_matches)])
    
    for i in range(len(eligible_matches)):
        p0 = kp1[eligible_matches[i].queryIdx]
        p1 = kp2[eligible_matches[i].trainIdx]
        P0[:, i] = np.array([p0.pt[0], p0.pt[1], 1])
        P1[:, i] = np.array([p1.pt[0], p1.pt[1], 1])
    
    P0_ravel = P0.transpose()[:, :2].ravel()
    P1_ravel = P1.transpose()[:, :2].ravel()
    
    if init_translation:
        arrays = [np.identity(2) for _ in range(len(eligible_matches))]
        J = np.concatenate((arrays), axis=0)
        p_star = np.matmul(np.matmul(np.linalg.inv(np.matmul(J.transpose(), J)), J.transpose()), P1_ravel - P0_ravel)
        H[0, 2] = p_star[0]
        H[1, 2] = p_star[1]
    
    for step in range(max_iteration):
        HP = np.matmul(H, P1)
        
        HP_homogeneous2cartesian = np.transpose([HP[0, :] / HP[2, :], HP[1, :] / HP[2, :]])
        predicted = HP_homogeneous2cartesian.ravel()
        res = -P0_ravel + predicted
        
        if np.abs(np.sum(res)) < residual_stopping_threshold:
            break
        
        J = np.zeros([2 * len(eligible_matches), 9])
        for i in range(len(eligible_matches)):
            J_i = np.zeros([2, 9])
            J_i[0, 0] = P1[0, i] / HP[2, i]
            J_i[0, 1] = P1[1, i] / HP[2, i]
            J_i[0, 2] = P1[2, i] / HP[2, i]
            
            J_i[1, 3] = P1[0, i] / HP[2, i]
            J_i[1, 4] = P1[1, i] / HP[2, i]
            J_i[1, 5] = P1[2, i] / HP[2, i]
            
            J_i[0, 6] = -P1[0, i] * HP[0, i] / (HP[2, i] ** 2)
            J_i[0, 7] = -P1[1, i] * HP[0, i] / (HP[2, i] ** 2)
            J_i[0, 8] = -P1[2, i] * HP[0, i] / (HP[2, i] ** 2)
            
            J_i[1, 6] = -P1[0, i] * HP[1, i] / (HP[2, i] ** 2)
            J_i[1, 7] = -P1[1, i] * HP[1, i] / (HP[2, i] ** 2)
            J_i[1, 8] = -P1[2, i] * HP[1, i] / (HP[2, i] ** 2)
            
            J[2 * i:2 * i + 2, :] = J_i
        
        delta_X = np.matmul(np.matmul(np.linalg.inv(np.matmul(J.transpose(), J)), J.transpose()), res)
        delta_X_reshaped = delta_X.reshape([3, 3])
        
        H = H - step_size * delta_X_reshaped
        H = H / H[2, 2]
    
    H = H / H[2, 2]
    HP = np.matmul(H, P1)
    
    HP_homogeneous2cartesian = np.transpose([HP[0, :] / HP[2, :], HP[1, :] / HP[2, :]])
    predicted = HP_homogeneous2cartesian.ravel()
    res = -P0_ravel + predicted
    
    return H


def perspective_transform(img_list, homography):
    height1, width1 = img_list[0].shape[:2]
    height2, width2 = img_list[1].shape[:2]
    pts = np.concatenate((np.float32([[0, 0], [0, height1], [width1, height1], [width1, 0]]).reshape(-1, 1, 2), 
                          cv2.perspectiveTransform(np.float32([[0, 0], [0, height2], [width2, height2], [width2, 0]]).reshape(-1, 1, 2), homography)), axis=0)
    [xmin, ymin] = np.int32(np.min(pts, axis=0).squeeze() - 0.5)
    [xmax, ymax] = np.int32(np.max(pts, axis=0).squeeze() + 0.5)
    t = [-xmin, -ymin]
    ht = np.array([[1, 0, t[0]], [0, 1, t[1]], [0, 0, 1]])
    return t, ht, ymax, ymin, xmax, xmin, height1, width1


def warp_perspective(img_list, ymax, ymin, xmax, xmin, ht, homography):
    stitch_r = np.zeros((ymax - ymin, xmax - xmin, 3)).astype(np.uint8)
    trans_mat = cv2.invert(ht.dot(homography))[1]
    for i in range(ymax - ymin):
        for j in range(xmax - xmin):
            img_i = int((trans_mat[1][0] * j + trans_mat[1][1] * i + trans_mat[1][2]) / (trans_mat[2][0] * j + trans_mat[2][1] * i + trans_mat[2][2]))
            img_j = int((trans_mat[0][0] * j + trans_mat[0][1] * i + trans_mat[0][2]) / (trans_mat[2][0] * j + trans_mat[2][1] * i + trans_mat[2][2]))
            if img_i >= 0 and img_j >= 0 and img_i < img_list[1].shape[0] and img_j < img_list[1].shape[1]:
                stitch_r[i][j] = img_list[1][img_i][img_j]
    
    return stitch_r


def alpha_blending(stitch_r, stitch_l):
    stitch_mask = np.ones_like(stitch_r, dtype=np.float32)
    stitch_mask[stitch_r > 0] = 0
    stitch_mask[stitch_l == 0] = 1
    start_col = np.min(np.where(stitch_mask == 0)[1])
    end_col = np.max(np.where(stitch_mask == 0)[1])
    for i in range(stitch_mask.shape[1]):
        col_value = np.ones(stitch_mask[:,i].shape) * (i - start_col) / (end_col - start_col)
        col_value[np.logical_and(stitch_r[:,i] > 0, stitch_l[:,i] == 0)] = 1
        col_value[np.logical_and(stitch_r[:,i] == 0, stitch_l[:,i] > 0)] = 0
        col_value = np.clip(col_value, 0.0, 1.0)
        stitch_mask[:,i] = col_value
    result = np.uint8((stitch_r * stitch_mask) + (stitch_l * (1 - stitch_mask)))
    return result


def clip_image(result, t):
    clipped_img = result[t[1]:,:]
    rows, cols = np.where(clipped_img[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    clipped_img = clipped_img[min_row:max_row, min_col:max_col, :]
    return clipped_img
