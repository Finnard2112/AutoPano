#!/usr/bin/env python

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

import numpy as np
import cv2 as cv
# Add any python libraries here
import os
import helper
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
import argparse



def main():
	# Add any Command Line arguments here
	# Parser = argparse.ArgumentParser()
	# Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
	
	# Args = Parser.parse_args()
	# NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""
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
		max_width = 800
		if img.shape[1] > max_width:
			scale = max_width / img.shape[1]
			img = cv.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
		images.append(img)

	C_imgs = []
	all_features = []
	all_x_best = []
	all_y_best = []
	all_matches = []
	homographies = []


	# Change these parameters for feature matching. 
	
	N_best = 2500 #2500 for dataset 2
	ratio_thresh = 0.9 #0.9 for dataset 2
	
	"""
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
	for i in range(len(images) - 1):
		patch1, patch2 = get_patches(images[i], images[i+1]) 
		
		# Your model prediction (e.g., HomographyNet or similar)
		# This replaces the entire Corner/ANMS/Matching/RANSAC block
		H_predicted = model.predict(patch1, patch2) 
		
		homographies.append(H_predicted)
	
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

	# 4. Warp and blend 
	for i in range(num_linked_images):
		H_final = H_translation.dot(H_global[i])
		warped_img = cv.warpPerspective(images[i], H_final, (canvas_w, canvas_h))

		gray_warped = cv.cvtColor(warped_img, cv.COLOR_BGR2GRAY)
		current_mask = gray_warped > 0

		# Add non-overlapping regions
		new_pixels_mask = current_mask & ~global_mask
		output_pano[new_pixels_mask] = warped_img[new_pixels_mask]

		# Average the overlapping region
		overlap = current_mask & global_mask
		
		pano_overlap = output_pano[overlap].astype(np.uint16)
		warp_overlap = warped_img[overlap].astype(np.uint16)
		output_pano[overlap] = ((pano_overlap + warp_overlap) // 2).astype(output_pano.dtype)

		# Update the global tracking mask
		global_mask |= current_mask

		del warped_img, gray_warped, current_mask, new_pixels_mask, overlap, pano_overlap, warp_overlap

	cv.imwrite(os.path.join("../Output", "mypano.png"), output_pano)


	
if __name__ == '__main__':
	main()
 
