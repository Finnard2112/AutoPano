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



def main():
	# Add any Command Line arguments here
	# Parser = argparse.ArgumentParser()
	# Parser.add_argument('--NumFeatures', default=100, help='Number of best features to extract from each image, Default:100')
	
	# Args = Parser.parse_args()
	# NumFeatures = Args.NumFeatures

	"""
	Read a set of images for Panorama stitching
	"""
	N = 5000
	#read the first N images
	images = [cv.imread(f'/nfshomes/hw987/cmsc733_Vision/YourDirectoryID_p1/Phase2/Data/Train/{i}.jpg') for i in range(1, N+1)]

	labels_file = open('./TxtFiles/LabelsTrain.txt', 'w')
	dirnames_file = open('./TxtFiles/DirNamesTrain.txt', 'w')

	for i, image in enumerate(images):
		# should be (426, 640, 3)
		height, width = image.shape[:2]

		#maximum possible perturbation
		rho = 32
		#patch size
		p_w = 128
		p_h = 128

		# Must be at least (patch_size + 2*rho) in both dimensions
		if height <= (p_h + 2*rho) or width <= (p_w + 2*rho):
			print(f"Skipping image {i+1}: dimensions ({width}x{height}) too small for patch.")
			continue

		
		# 3. Generate a random top-left corner (x, y)
		x1 = np.random.randint(rho, width - p_w - rho)
		y1 = np.random.randint(rho, height - p_h - rho)

		# 4. Calculate coordinates for all 4 corners
		x2, y2 = x1 + p_w, y1 + p_h
		
		corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.int32)

		patch_original = image[corners[0, 1]:corners[2, 1], corners[0, 0]:corners[2, 0]]

		corners_perturbed = np.array([[x1 + np.random.randint(-rho, rho), y1 + np.random.randint(-rho, rho)], [x2 + np.random.randint(-rho, rho), y1 + np.random.randint(-rho, rho)], [x2 + np.random.randint(-rho, rho), y2 + np.random.randint(-rho, rho)], [x1 + np.random.randint(-rho, rho), y2 + np.random.randint(-rho, rho)]], dtype=np.int32)

		patch_perturbed = image[corners_perturbed[0, 1]:corners_perturbed[2, 1], corners_perturbed[0, 0]:corners_perturbed[2, 0]]


		homography = cv.getPerspectiveTransform(corners_perturbed.astype(np.float32), corners.astype(np.float32))
		#this is the homography that maps the perturbed patch to the original patch
		warped_image = cv.warpPerspective(image, homography, (width, height))
		warped_patch = warped_image[corners[0, 1]:corners[2, 1], corners[0, 0]:corners[2, 0]]

		#save patch_original and warped_patch
		cv.imwrite(f'../Data/Patches/Original/{i+1}.jpg', patch_original)
		cv.imwrite(f'../Data/Patches/Warped/{i+1}.jpg', warped_patch)




		#save patch_original and warped_patch
		corner_displacement = corners_perturbed - corners
		# stacked_images = np.concatenate((patch_original, warped_patch), axis=2)
		
		
		label_values = corner_displacement.flatten()
		labels_file.write(' '.join(map(str, label_values)) + '\n')
		
		dirnames_file.write(f'{i+1}\n')

	labels_file.close()
	dirnames_file.close()
	
	"""
	Obtain Homography using Deep Learning Model (Supervised and Unsupervised)
	"""
	
	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""
	
if __name__ == '__main__':
	main()
 
