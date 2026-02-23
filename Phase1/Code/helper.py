import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
import random
import argparse

def extract_features(img, x_coords, y_coords, patch_size=40):
    if len(img.shape) == 3:
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    features = []
    offset = patch_size // 2 
    # Padding to account for corners
    padded_img = cv.copyMakeBorder(gray, offset, offset, offset, offset, cv.BORDER_REFLECT)
    
    for x, y in zip(x_coords, y_coords):
        # Shift coords from padding
        padded_x, padded_y = int(x) + offset, int(y) + offset
        
        # Take a patch of size 40x40 centered around the keypoint
        patch = padded_img[padded_y - offset : padded_y + offset, padded_x - offset : padded_x + offset]
        
        # Apply Gaussian blur
        blurred = cv.GaussianBlur(patch, (5, 5), 0)
        
        # Sub-sample the blurred output to 8x8
        sub_sampled = cv.resize(blurred, (8, 8), interpolation=cv.INTER_AREA)
        
        # Reshape to obtain a 64x1 vector
        vector = sub_sampled.flatten() 
        
        #  Standardize the vector (zero mean, variance of 1)
        mean = np.mean(vector)
        std = np.std(vector)
        
        if std == 0:
            std = 1e-5 
            
        standardized_vector = (vector - mean) / std
        
        features.append(standardized_vector)
        
    return np.array(features)

def apply_anms(corner_img, N_best):
    # Find all local maxima using peak_local_max
    coordinates = peak_local_max(corner_img, min_distance=1, threshold_rel=0.01)
    
    # (x,y)coordinates of local maxima
    y_coords = coordinates[:, 0]
    x_coords = coordinates[:, 1]
    scores = corner_img[y_coords, x_coords]
    
    N_strong = len(scores)
    
    # Initialize r_i = infinity for i = [1 : N_strong]
    r = np.full(N_strong, np.inf)

    for i in range(N_strong):
        better_points_idx = np.where(scores > scores[i])[0]
        
        if len(better_points_idx) > 0:
            dx = x_coords[better_points_idx] - x_coords[i]
            dy = y_coords[better_points_idx] - y_coords[i]
            ed = dx**2 + dy**2
            r[i] = np.min(ed)

    best_indices = np.argsort(r)[::-1][:N_best]
    best_x = x_coords[best_indices]
    best_y = y_coords[best_indices]
    
    return best_x, best_y

def match_features(features1, features2, x1, y1, x2, y2, ratio_thresh=0.8):
    matches = []
    
    cv_kp1 = [cv.KeyPoint(float(x), float(y), 1) for x, y in zip(x1, y1)]
    cv_kp2 = [cv.KeyPoint(float(x), float(y), 1) for x, y in zip(x2, y2)]
    
    for i in range(len(features1)):
        diff = features1[i] - features2
        
        # Calculate SSD for this single point against all points in Image 2
        distances = np.sum(diff ** 2, axis=1)
        
        # Get the indices of the sorted distances (lowest to highest)
        sorted_indices = np.argsort(distances)
        
        best_idx = sorted_indices[0]
        second_best_idx = sorted_indices[1]
        
        best_dist = distances[best_idx]
        second_best_dist = distances[second_best_idx]
        
        # Take the ratio of best match to second best match
        if (best_dist / (second_best_dist + 1e-5)) < ratio_thresh:
            match = cv.DMatch(_queryIdx=i, _trainIdx=best_idx, _distance=best_dist)
            matches.append(match)
            
    return cv_kp1, cv_kp2, matches


def apply_ransac(matches, kp1, kp2, N_max=2000, tau=5.0):
    """
    RANSAC algorithm to estimate robust homography and reject outliers.
    """
    # Extract coordinates from the DMatch objects
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches])
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches])
    
    num_matches = len(matches)
    max_inliers = 0
    best_inliers_idx = []
    
    src_pts_homogeneous = np.column_stack((src_pts, np.ones(num_matches)))
    
    for i in range(N_max):
        # Select four feature pairs at random
        idx = random.sample(range(num_matches), 4)
        p1 = src_pts[idx]
        p2 = dst_pts[idx]
        
        # Compute homography H between the previously picked point pairs
        H = cv.getPerspectiveTransform(p1, p2)
        
        # Compute Sum of Squared Differences (SSD)
        transformed_pts = np.dot(H, src_pts_homogeneous.T).T
        
        epsilon = 1e-8
        transformed_x = transformed_pts[:, 0] / (transformed_pts[:, 2] + epsilon)
        transformed_y = transformed_pts[:, 1] / (transformed_pts[:, 2] + epsilon)
        transformed_pts_2d = np.column_stack((transformed_x, transformed_y))

        ssd = np.sum((dst_pts - transformed_pts_2d) ** 2, axis=1)
        
        # Compute inliers where SSD < tau
        inlier_idx = np.where(ssd < tau)[0]
        
        # Keep largest set of inliers
        if len(inlier_idx) > max_inliers:
            max_inliers = len(inlier_idx)
            best_inliers_idx = inlier_idx
            
    #  Re-compute least-squares H estimate on all of the inliers
    best_src_pts = src_pts[best_inliers_idx]
    best_dst_pts = dst_pts[best_inliers_idx]
    final_H, _ = cv.findHomography(best_src_pts, best_dst_pts, 0)
    inlier_matches = [matches[i] for i in best_inliers_idx]
    
    return final_H, inlier_matches
