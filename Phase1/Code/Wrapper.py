#!/usr/bin/evn python

"""
CMSC733 Spring 2019: Classical and Deep Learning Approaches for
Geometric Computer Vision
Project1: MyAutoPano: Phase 1 Starter Code

Author(s): 
Chahat Deep Singh (chahat@terpmail.umd.edu) 
PhD Student in Computer Science,
University of Maryland, College Park

Nitin J. Sanket (nitinsan@terpmail.umd.edu)
PhD Candidate in Computer Science,
University of Maryland, College Park
"""

# Code starts here:
import os
import gc
import helper
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
import argparse
# Add any python libraries here

# 

def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--imgpath', default="../Data/Train/Set1", help='Include the image path')
	Args = Parser.parse_args()
	imgpath = Args.imgpath

	"""
	Read a set of images for Panorama stitching

	"""

	files = sorted([os.path.join(imgpath, f) for f in os.listdir(imgpath)])

	images = []
	for file in files:
		img = cv.imread(file)

		# Resize
		max_width = 800
		if img.shape[1] > max_width:
			scale = max_width / img.shape[1]
			img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))

		# Cylindrical projection
		h, w = img.shape[:2]
		f = w * 1.0
		cylindrical_img = helper.cylindrical_warp(img, f)
		images.append(cylindrical_img)

	C_imgs = []
	all_features = []
	all_x_best = []
	all_y_best = []
	all_matches = []
	homographies = []
	valid_image_indices = [] 


	# Change these parameters for feature matching. 
	
	N_best = 3000 #2500 for dataset 2
	ratio_thresh = 0.9 #0.9 for dataset 2

	"""
	Corner Detection
	Save Corner detection output as corners.png
	"""

	for img, file in zip(images, files):
		# Detect corners
		gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		gray = np.float32(gray)
		dst = cv.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
		dst = cv.dilate(dst, None)
		C_imgs.append(dst)

		img_corners = img.copy()
		img_corners[dst > 0.01 * dst.max()] = [0, 0, 255]

		# Save image
		filename = os.path.basename(file)
		save_path = os.path.join("../Output", f"corners_{filename}")
		cv.imwrite(save_path, img_corners)

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""
	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""
	
	# Running ANMS and extracting features
	for img, dst, file in zip(images, C_imgs, files):

		best_x, best_y = helper.apply_anms(dst, N_best)

		features = helper.extract_features(img, best_x, best_y)

		all_features.append(features)
		all_x_best.append(best_x)
		all_y_best.append(best_y)

		# Draw the points
		img_anms = img.copy()
		for x, y in zip(best_x, best_y):
		    cv.circle(img_anms, (x, y), radius=2, color=(0, 255, 0), thickness=-1)
		
		# Save ANMS image
		filename = os.path.basename(file)
		save_path = os.path.join("../Output", f"anms_{filename}")
		cv.imwrite(save_path, img_anms)

	"""
	Feature Matching + RANSAC, Estimate Homography
	Save Feature Matching output as matching.png
	"""

	MIN_INLIERS = 60 
	anchor_idx = 0
	
	for next_idx in range(1, len(images)):
		f1, f2 = all_features[anchor_idx], all_features[next_idx]
		x1, y1 = all_x_best[anchor_idx], all_y_best[anchor_idx]
		x2, y2 = all_x_best[next_idx], all_y_best[next_idx]
		
		# Perform matching
		kp1, kp2, good_matches = helper.match_features(f1, f2, x1, y1, x2, y2, ratio_thresh=ratio_thresh)
		print(len(good_matches))

		H, inlier_matches = (None, [])
		
		if len(good_matches) >= MIN_INLIERS:
			H, inlier_matches = helper.apply_ransac(good_matches, kp1, kp2, N_max=4000, tau=2.0)
		
		if len(inlier_matches) < MIN_INLIERS:
			print(f"Skipping Image {next_idx}. Not enough inliers {len(inlier_matches)}")
			
			if len(valid_image_indices) == 0:
				anchor_idx = next_idx
			
			continue

		if len(valid_image_indices) == 0:
			valid_image_indices.append(anchor_idx)
			
		valid_image_indices.append(next_idx)
		homographies.append(H)
		all_matches.append((kp1, kp2, good_matches))
		
		img1, img2 = images[anchor_idx], images[next_idx]
		match_img = cv.drawMatches(
			img1, kp1, 
			img2, kp2, 
			good_matches, 
			None, 
			flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
		)

		ransac_img = cv.drawMatches(
			img1, kp1, 
			img2, kp2, 
			inlier_matches, 
			None, 
			flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
		)

		save_path = os.path.join("../Output", f"ransac_{anchor_idx}_to_{next_idx}.png")
		cv.imwrite(save_path, ransac_img)
		save_path = os.path.join("../Output", f"matching_{anchor_idx}_to_{next_idx}.png")
		cv.imwrite(save_path, match_img)
		
		anchor_idx = next_idx

	images = [images[idx] for idx in valid_image_indices]

	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

	# Determine the middle image 
	num_images = len(images)
	ref_idx = num_images // 2
	
	H_global = [None] * num_images
	H_global[ref_idx] = np.eye(3)

	for i in range(ref_idx, num_images - 1):
		H_inv = np.linalg.inv(homographies[i])
		H_global[i+1] = H_global[i].dot(H_inv)

	for i in range(ref_idx - 1, -1, -1):
		H_global[i] = H_global[i+1].dot(homographies[i])

	num_linked_images = len(H_global)

	if num_linked_images < 2:
		print("Error: Not enough images were successfully matched to create a panorama.")
		return

	# Find the global bounding box for the new canvas
	x_min_global, y_min_global = np.inf, np.inf
	x_max_global, y_max_global = -np.inf, -np.inf

	for i in range(num_linked_images):
		h, w = images[i].shape[:2]
		corners = np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2)
		transformed_corners = cv.perspectiveTransform(corners, H_global[i])

		# Use floor and ceil to ensure we don't truncate fractional pixel boundaries
		x_min = np.floor(transformed_corners[:, 0, 0].min())
		y_min = np.floor(transformed_corners[:, 0, 1].min())
		x_max = np.ceil(transformed_corners[:, 0, 0].max())
		y_max = np.ceil(transformed_corners[:, 0, 1].max())

		x_min_global = min(x_min_global, x_min)
		y_min_global = min(y_min_global, y_min)
		x_max_global = max(x_max_global, x_max)
		y_max_global = max(y_max_global, y_max)

	# Convert to integers for canvas creation
	x_min_global = int(x_min_global)
	y_min_global = int(y_min_global)
	x_max_global = int(x_max_global)
	y_max_global = int(y_max_global)

	canvas_w = x_max_global - x_min_global
	canvas_h = y_max_global - y_min_global


	# Check if final dimension too large
	MAX_DIMENSION = 15000 
	if canvas_w > MAX_DIMENSION or canvas_h > MAX_DIMENSION or canvas_w <= 0 or canvas_h <= 0:
		canvas_w = min(max(canvas_w, 1), MAX_DIMENSION)
		canvas_h = min(max(canvas_h, 1), MAX_DIMENSION)

	# Global translation matrix
	translation_dist = [-x_min_global, -y_min_global]
	H_translation = np.array([[1, 0, translation_dist[0]],
								[0, 1, translation_dist[1]],
								[0, 0, 1]], dtype=np.float64)

	output_pano = np.zeros((canvas_h, canvas_w, 3), dtype=images[0].dtype)

	global_mask = np.zeros((canvas_h, canvas_w), dtype=bool)

	# Warp and blend using Distance Transform 
	
	# output_pano_float needs 3 channels for BGR
	output_pano_float = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
	weight_pano = np.zeros((canvas_h, canvas_w, 1), dtype=np.float32)

	for i in range(num_linked_images):
		H_final = H_translation.dot(H_global[i])
		warped_img = cv.warpPerspective(images[i], H_final, (canvas_w, canvas_h))

		gray_warped = cv.cvtColor(warped_img, cv.COLOR_BGR2GRAY)
		_, binary_mask = cv.threshold(gray_warped, 0, 255, cv.THRESH_BINARY)

		dist_transform = cv.distanceTransform(binary_mask, cv.DIST_L2, 3)

		max_dist = dist_transform.max()
		if max_dist > 0:
			dist_transform = dist_transform / max_dist

		# Reshape to (H, W, 1) to use NumPy broadcasting instead of cv.merge
		alpha_mask = dist_transform[:, :, np.newaxis].astype(np.float32)
		output_pano_float += warped_img.astype(np.float32) * alpha_mask
		weight_pano += alpha_mask

		# Force garbage collection
		del warped_img, gray_warped, binary_mask, dist_transform, alpha_mask
		gc.collect()

	# Avoid division by zero in empty areas
	weight_pano[weight_pano == 0] = 1.0

	# The 1-channel weight_pano broadcasts perfectly across the 3-channel output array
	output_pano_float /= weight_pano

	output_pano = np.clip(output_pano_float, 0, 255).astype(np.uint8)
	
	# Free up the massive float arrays before saving
	del output_pano_float, weight_pano
	gc.collect()

	cv.imwrite(os.path.join("../Output", "mypano.png"), output_pano)

    
if __name__ == '__main__':
    main()
 
