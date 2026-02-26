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
from matplotlib import pyplot as plt
from skimage.feature import peak_local_max
import argparse
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from Network.Network import HomographyModel



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
	Parser.add_argument('--imgpath', default="../../Phase1/Data/Train/Set1", help='Include the image path')
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
	Obtain Homography using Deep Learning Model (Supervised)
	"""
	num_images = len(images)
	ref_idx = num_images // 2
	
	# Load the trained model
	tf.reset_default_graph()
	ImageSize = [128, 128, 6]
	MiniBatchSize = 1
	
	ImgPH = tf.placeholder(tf.float32, shape=(MiniBatchSize, ImageSize[0], ImageSize[1], ImageSize[2]))
	prLogits, prSoftMax = HomographyModel(ImgPH, ImageSize, MiniBatchSize)
	
	Saver = tf.train.Saver()
	
	with tf.Session() as sess:
		Saver.restore(sess, "../Checkpoints/49model.ckpt")
		
		# Compute homographies between consecutive images
		for i in range(num_images - 1):
			# Get original image dimensions
			h_orig, w_orig = images[i].shape[:2]
			
			# Calculate scaling factors
			scale_x = w_orig / 128.0
			scale_y = h_orig / 128.0
			
			# Resize images to 128x128
			img1_resized = cv.resize(images[i], (128, 128))
			img2_resized = cv.resize(images[i+1], (128, 128))
			
			# Stack images (reference first, then image to warp)
			stacked_img = np.dstack([img1_resized, img2_resized]).astype(np.float32)
			stacked_img = (stacked_img / 127.5) - 1.0
			stacked_img = np.expand_dims(stacked_img, axis=0)

			# --- Sanity Check: Save to Disk ---
			# 1. Ensure a directory exists for these checks
			debug_dir = "../Debug_Inference"
			if not os.path.exists(debug_dir):
				os.makedirs(debug_dir)

			# 2. Extract the two images from the 6-channel stack
			# Input shape is (1, 128, 128, 6)
			check_img = (np.squeeze(stacked_img) * 255.0).astype(np.uint8)
			img1_view = check_img[:, :, 0:3] # First 3 channels
			img2_view = check_img[:, :, 3:6] # Last 3 channels

			# 3. Stack them horizontally for easy viewing
			side_by_side = np.hstack((img1_view, img2_view))

			# 4. Save with index (i is your loop variable)
			debug_filename = os.path.join(debug_dir, f"input_pair_{i:03d}.png")
			cv.imwrite(debug_filename, side_by_side)
			# ----------------------------------
			
			# Run inference to get 128x128-scale displacements
			H_4pt_small = sess.run(prSoftMax, feed_dict={ImgPH: stacked_img})
			H_4pt_small = H_4pt_small.reshape((4, 2)) * 32.0
			
			# Scale the displacements to full resolution
			# scale_x = w_orig / 128.0
			# scale_y = h_orig / 128.0
			H_4pt_large = H_4pt_small * np.array([scale_x, scale_y])
			
			# Define corners for the full resolution image
			pts_src_large = np.float32([[0, 0], [w_orig, 0], [w_orig, h_orig], [0, h_orig]])
			pts_dst_large = (pts_src_large + H_4pt_large).astype(np.float32).reshape(4, 2)			
			# Get the homography matrix for the large image
			H = cv.getPerspectiveTransform(pts_src_large, pts_dst_large)
			
			homographies.append(H)
	
	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	
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

		# --- NEW: Save each warped image to disk ---
		warped_debug_dir = "../Warped_Steps"
		if not os.path.exists(warped_debug_dir):
			os.makedirs(warped_debug_dir)
		
		warped_filename = os.path.join(warped_debug_dir, f"warped_img_{i:03d}.png")
		cv.imwrite(warped_filename, warped_img)
		# -------------------------------------------

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
 
