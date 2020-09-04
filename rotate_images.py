# Input: Folder of images
# Output: New folder with images rotated by a set amount
# Mark Whitty
# UNSW
# V1 20190204 Tested, briefly and seems to work ok.
# May have 1 pixel offset for rotated images due to rounding

from __future__ import print_function
import os  # For PATH etc.  https://docs.python.org/2/library/os.html
import glob  # For Unix style finding pathnames matching a pattern (like regexp)
import numpy as np
import cv2
import time
from datetime import datetime




# Display barcode and QR code location for a single detected object
def display(im, decodedObject):
    if decodedObject is None:
        return
    # Only deal with first QR code found in the image
    points = decodedObject.polygon

    # If the points do not form a quad, find convex hull
    if len(points) > 4:
        hull = cv2.convexHull(np.array([point for point in points], dtype=np.float32))
        hull = list(map(tuple, np.squeeze(hull)))
    else:
        hull = points

    # Number of points in the convex hull
    n = len(hull)

    # Draw the convex hull
    for j in range(0, n):
        cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

    # Display results
    cv2.imshow("Results", im)
    #cv2.waitKey(0)


# Check whether any file exists in a given directory
# https://stackoverflow.com/questions/33463325/python-check-if-any-file-exists-in-a-given-directory
def does_file_exist_in_dir(path):
    return any(os.path.isfile(os.path.join(path, i)) for i in os.listdir(path))

# Rotate an image about its centre by a given number of degrees,
# not handling scaling and cropping
def maw_rotate_image(im, angle):
    (h, w) = im.shape[:2]
    centre = (w / 2, h / 2)

    M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    im_rotated = cv2.warpAffine(im, M, (w, h))
    return im_rotated


# Main
if __name__ == '__main__':

    file_input_path = "original_leafroll_decoded"
    file_output_path = file_input_path + "_rotated/"

    # Check if input directory exists and contains files
    if not os.path.exists(file_input_path):
        print("Input directory ", file_input_path, " does not exist")
        exit(1)

    if not does_file_exist_in_dir(file_input_path):
        print(" No files in input directory")
        exit(1)

    # Create target directory & all intermediate directories if don't exists
    if not os.path.exists(file_output_path):
        os.makedirs(file_output_path)
        print("Directory ", file_output_path, " created ")
    else:
        print("Warning: directory ", file_output_path, " already exists, existing files may be overwritten")

    num_input_files = 0
    num_correctly_rotated_files = 0

    # Read all files in directory that are in JPG format
    print(len(glob.glob(os.path.join(file_input_path, "*.jpg"))), " JPG files in ", file_input_path, " directory")
    for infile in glob.glob(os.path.join(file_input_path, "*.jpg")):
        # Read image
        im = cv2.imread(infile)
        num_input_files = num_input_files + 1

        im_rotated180 = maw_rotate(im, 180)

        num_correctly_rotated_files = num_correctly_rotated_files + 1

        #cv2.imshow("Rotated image", im_rotated180)
        #cv2.waitKey(0)

        new_filename = file_output_path + os.path.basename(infile)

        cv2.imwrite(new_filename, im_rotated180, )

    # Write out statistics to a text file
    print(os.path.basename(__file__) + " executed at " + str(datetime.now().isoformat(' ')) + '\n')
    print("Read " + str(num_input_files) + " images from " + file_input_path + '\n')
    print("Correctly rotated " + str(num_correctly_rotated_files) + " image(s) and unable to parse " + str(num_input_files-num_correctly_rotated_files) + " image(s)\n")

