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
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import argparse
# Add any python libraries here



def main():
	# Add any Command Line arguments here
	Parser = argparse.ArgumentParser()
	Parser.add_argument('--imgpath', default="../Data/Train/Set1", help='Include the image path')
	Args = Parser.parse_args()
	imgpath = Args.imgpath

	"""
	Read a set of images for Panorama stitching

	"""
	files = [os.path.join(imgpath, f) for f in os.listdir(imgpath)]
	images = [cv.imread(file) for file in files]

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
		img[dst > 0.01 * dst.max()] = [0, 0, 255]

		# Save image
		filename = os.path.basename(file)
		save_path = os.path.join("../Output", f"harris_{filename}")
		cv.imwrite(save_path, img)

	"""
	Perform ANMS: Adaptive Non-Maximal Suppression
	Save ANMS output as anms.png
	"""

	"""
	Feature Descriptors
	Save Feature Descriptor output as FD.png
	"""

	"""
	Feature Matching
	Save Feature Matching output as matching.png
	"""


	"""
	Refine: RANSAC, Estimate Homography
	"""


	"""
	Image Warping + Blending
	Save Panorama output as mypano.png
	"""

    
if __name__ == '__main__':
    main()
 
