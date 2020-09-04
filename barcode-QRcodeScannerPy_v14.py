# Input: Folder of images with QR codes
# Output: Folder with QR codes recognised and files renamed according to QR code content
# Mark Whitty
# UNSW
# https://github.com/markwhittyunsw/QR_image_renamer
# Derived from https://github.com/spmallick/learnopencv/tree/master/barcode-QRcodeScanner

from __future__ import print_function
import math  # For handling degrees and radians

import os  # For PATH etc.  https://docs.python.org/2/library/os.html
import glob  # For Unix style finding pathnames matching a pattern (like regexp)
import shutil  # For file copying
import pyzbar.pyzbar as pyzbar
from pyzbar.pyzbar import ZBarSymbol
import numpy as np
import cv2
import time
from datetime import datetime
import sys
import re  # Regular expressions used for file type matching
from numpy import array
from tkinter import filedialog  # For a GUI for user specified directories
from tkinter import *

def decode(im):
    # Find only QR codes (remove the second argument to find other symbols)
    decodedObjects = pyzbar.decode(im, symbols=[ZBarSymbol.QRCODE])

    # Print results
    # Check QR code exists in image
    if len(decodedObjects) > 0:
        max_size = 0
        for index, obj in enumerate(decodedObjects):
            # Check it is a QR code
            if obj.type != 'QRCODE':
                continue
        obj = decodedObjects[index]
        print('Decoded image:', infile)
        print('Type : ', obj.type)
        # noting that QR code data includes 'b' and two apostraphes, presumably to indicate data type, so these are not printed
        print('Data : ', str(obj.data)[2:-1], '\n')
    else:
        print("No object found in image named", infile, "\n")
        obj = None
    return obj

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
    #for j in range(0, n):
    #    cv2.line(im, hull[j], hull[(j + 1) % n], (255, 0, 0), 3)

    # Display results
    #cv2.imshow("Results", im)
    #cv2.waitKey(0)

# Resize image according to given maximum height or width, returning the resized image and
# the scaling factor used
def resize_max(im, max_size):
    height, width = im.shape[:2]
    scaling_factor = 1
    # only shrink if img is bigger than required
    if max_size < height or max_size < width:
        # get scaling factor
        scaling_factor = max_size / float(height)
        if max_size / float(width) < scaling_factor:
            scaling_factor = max_size / float(width)
        # resize image
        resized_im = cv2.resize(im, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
    else:
        resized_im = im
    return resized_im, scaling_factor


# Check whether any file exists in a given directory
# https://stackoverflow.com/questions/33463325/python-check-if-any-file-exists-in-a-given-directory
def does_file_exist_in_dir(path):
    return any(os.path.isfile(os.path.join(path, i)) for i in os.listdir(path))

# Main
if __name__ == '__main__':

    # Read target and output paths
    if len(sys.argv) < 2:
        # No command line input or output directories specified, use a GUI to ask the user to specify these
        file_input_path = "."
        file_output_path = "."
        root = Tk()
        file_input_path = filedialog.askdirectory(title="Location of images to be decoded", initialdir=".", parent=root)
        file_output_path = filedialog.askdirectory(title="Location where renamed images will be copied",
                                                   initialdir=file_input_path, parent=root)
    elif len(sys.argv) < 3:
        file_input_path = sys.argv[1]
        file_output_path = os.path.join(file_input_path,
                                        os.path.basename(os.path.abspath(file_input_path)) + "_decoded\\")
    else:
        file_input_path = sys.argv[1]
        file_output_path = sys.argv[2]

    max_image_dimension = 1600  # Maximum image dimension, if greater than this will be resized before processing
        # (output image remained unchanges), as the QR code decoder (pyzbar) has a size limit around 2000 pixels.

    # Check if input directory exists and contains files
    if not os.path.exists(file_input_path):
        print("Input directory ", file_input_path, " does not exist")
        exit(1)

    if not does_file_exist_in_dir(file_input_path):
        print("Warning: No files in input directory: ", file_input_path,)
        exit(1)

    # Read all files in directory that are in image format as allowed by OpenCV
    # https: // docs.opencv.org / 3.0 - beta / modules / imgcodecs / doc / reading_and_writing_images.html
    print(os.path.abspath(file_input_path))
    input_files = [f for f in os.listdir(file_input_path) if re.search(r'.*\.(jpg|png|bmp|dib|jpe|jpeg|jp2|tif|tiff)$', f)]
    input_files = list(map(lambda x: os.path.join(file_input_path, x), input_files))
    num_input_files = len(input_files)

    if num_input_files < 1:
        print("Warning: No image files in input directory: ", file_input_path)
        exit(0)

    # Create target directory & all intermediate directories if it doesn't exists
    if not os.path.exists(file_output_path):
        os.makedirs(file_output_path)
        print("Output directory ", file_output_path, " created ")
    else:
        print("Warning: output directory ", file_output_path, " already exists, existing files may be overwritten")

    while not os.path.exists(file_output_path):
        time.sleep(0.5)  # Short delay to ensure directory is created

    input_files_it = 0
    num_correctly_parsed_files = 0

    log_file = open(os.path.join(file_output_path, "decoding_log.txt"), "w+")
    log_file.write("Image number\tInput filename\tOutput filename\tStatus (0 = correctly read and parsed QR code, " +
                   "1 = Unable to parse code in image" +
                   "\tData\tType\tRect\tPolygon\n")

    print(str(num_input_files), " image files in ", file_input_path, " directory")
    if num_input_files > 10000:
        print("Warning: Halting execution as more than 10,000 files input. This program is not designed for such a" 
              " large volume of images.")
        exit(0)

    # Attempts to decode each image
    for infile in input_files:
        # Read image
        im = cv2.imread(infile)
        if im is None:
            print("Warning: image " + infile + " could not be read, trying next file")
            continue
        input_files_it = input_files_it + 1

        # Check size and if too large, resize it
        resized_im, scaling_factor = resize_max(im, max_image_dimension)

        decodedObject = decode(resized_im)
        if decodedObject is None:
            # If image is not correctly processed, copy file as is and append '_' to filename.
            # This will overwrite any existing files with this name (which are presumably useless)
            new_filename = os.path.join(file_output_path, ('_' + os.path.basename(infile)))
            shutil.copy2(infile, new_filename)
            log_file.write(str(input_files_it) + "\t" + os.path.basename(infile) + "\t" + "_" + os.path.basename(infile) \
                           + "\t" + "1" + "\t" + "Unable to parse code in image" + "\n")

            continue
        num_correctly_parsed_files = num_correctly_parsed_files + 1
        #display(resized_im, decodedObject)

        # Copy resized image and change its filename to match QR code
        # New filename, noting that QR code data includes 'b' and two apostraphes, presumably to indicate data type,
        # so these are stripped out
        qr_data = str(decodedObject.data)[2:-1]

        # Remove strange characters from filename and replace with underscores
        # Derived from list of special characters on Windows here:
        # https://en.wikipedia.org/wiki/Filename#Reserved_characters_and_words
        qr_data = qr_data.replace('/', '_').replace('\\', '_').replace('?', '_').replace('%', '_') \
            .replace('*', '_').replace(':', '_').replace('|', '_').replace('\"', '_').replace('<', '_') \
            .replace('>', '_').replace('.', '_').replace(' ', '_')

        new_filename = os.path.join(file_output_path, qr_data + os.path.splitext(infile)[1])
        file_iterator = 0

        # Copy original image to new file with filename set to decoded QR code value, incrementing a filename suffix if
        # a duplicate detected
        if os.path.exists(new_filename):
            # File already exists
            while os.path.exists(new_filename):
                new_filename = os.path.join(file_output_path, qr_data + "_" + str(file_iterator).zfill(3) + os.path.splitext(infile)[1])
                file_iterator += 1
        shutil.copy2(infile, new_filename)

        log_file.write(str(input_files_it) + "\t" + os.path.basename(infile) + "\t" +
                       os.path.basename(new_filename) + "\t0\t" + str(decodedObject.data) + "\t" +
                       str(decodedObject.type) + "\t" + str(decodedObject.rect) + "\t" +
                       str(decodedObject.polygon) + "\t\n")
        print("Completed image ", input_files_it, " of ", num_input_files, " [", int(float(input_files_it)/float(num_input_files)*100), "%]")

    # Write out statistics to a text file
    stats_file = open(os.path.join(file_output_path, "decoding_stats.txt"), "w+")
    stats_file.write(os.path.basename(__file__) + " executed at " + str(datetime.now().isoformat(' ')) + '\n')
    stats_file.write("Read " + str(input_files_it) + " images from " + file_input_path + '\n')
    stats_file.write("Correctly parsed QR codes in " + str(num_correctly_parsed_files) + " image(s) and unable to parse " \
                     + str(input_files_it-num_correctly_parsed_files) + " image(s)\n")
    stats_file.write("Images unable to be parsed have had an underscore prepended to their filename\n")
    stats_file.write("Wrote correctly parsed files to " + file_output_path + '\n')
    stats_file.close()

    log_file.close()
