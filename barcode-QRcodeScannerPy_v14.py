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
from datetime import datetime as dt
from fpdf import FPDF
import sys
import re  # Regular expressions used for file type matching
from numpy import array
from tkinter import filedialog  # For a GUI for user specified directories
from tkinter import *
from tabulate import tabulate
import pandas as pd
import pyqrcode


def read_qr_labels_from_file(filename):
    #f = open(filename, "r")
    # Select relevant columns (for testing only)
    #labels = f.readlines()
    #print(labels)

    qr_dataframe = pd.read_csv(filename, converters={i: str for i in range(100)})
    print(tabulate(qr_dataframe, headers='keys', tablefmt='psql'))

    return qr_dataframe


def decode(im, infile):
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
        # noting that QR code data includes 'b' and two apostraphes, presumably to indicate data type,
        # so these are not printed
        print('Data : ', str(obj.data)[2:-1], '\n')
    else:
        print("No object found in image named", infile, "\n")
        obj = None
    return obj


def display(im, decodedObject):
    # Display barcode and QR code location for a single detected object
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


def resize_max(im, max_size):
    # Resize image according to given maximum height or width, returning the resized image and
    # the scaling factor used
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


def does_file_exist_in_dir(path):
    # Check whether any file exists in a given directory
    # https://stackoverflow.com/questions/33463325/python-check-if-any-file-exists-in-a-given-directory
    return any(os.path.isfile(os.path.join(path, i)) for i in os.listdir(path))


def generate_QR_codes(qr_content, dest_path):
    # qr_content_human_string = qr_content.strftime('%Y-%m-%d_%H.%M.%S.%f%z')
    # Convert item to string, in case it isn't already.
    # time_seconds_since_epoch = original_qr_content.timestamp()
    # In some cases newlines need to be removed, others we need the correct type to be passed in
    # to be processed and displayed in associated text (eg. date).
    new_code = pyqrcode.create(qr_content, error='l', mode='binary', encoding=None)  #, version=10, )

    image_path = os.path.join(dest_path, (qr_content + '.png'))
    # new_png = new_code.png(image_path, scale=6, module_color=[0, 0, 0, 128], background=[0xff, 0xff, 0xff])
    # new_code.show()  # Very slow!

    # Display the image along with border
    code_size = len(new_code.code)
    image_array2 = np.ones((code_size+8, code_size+8), np.uint8)
    image_array1 = 1 - array(new_code.code, np.uint8)   # For some reason this needs to be inverted!
    image_array2[4:(4+code_size), 4:(4+code_size)] = image_array1
    # Add blank area at bottom for human readable text
    h, w = image_array2.shape
    blank_strip = np.ones((int(h/2), w), dtype=image_array2.dtype)
    image_array2 = np.r_[image_array2, blank_strip]

    image_array2 = image_array2*255  # Scale 0:1 values to 0:255

    scale_factor = 20  # Resize the image to be larger by simply scaling it up

    image_array2 = cv2.resize(image_array2, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    #cv2.imshow("New code: ", image_array2)
    #cv2.waitKey(0)

    y_text_pos = h*scale_factor
    x_text_pos = 0  #int(w*scale_factor/2)

    # Write content onto image
    #cv2.putText(image_array2, "Timestamp: YYYY-MM-DD_HH.MM.SS.uS", (x_text_pos, int(y_text_pos*1.03)),
    #            fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
    #cv2.putText(image_array2, human_string, (x_text_pos, y_text_pos),
    #    fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)


    cv2.imwrite(image_path, image_array2)


    #print(new_code.terminal(quiet_zone=1))
    return image_path


def add_page_to_PDF(pdf, image_path, total_pages, human_string):
    # Write content into a PDF, see https://www.blog.pythonlibrary.org/2018/06/05/creating-pdfs-with-pyfpdf-and-python/
    pdf.add_page()
    pdf.set_auto_page_break(False)
    pdf.set_font("times", size=16)
    QR_width_pdf = 210  # Width (and height) of QR code (including border) on PDF (cm, assuming A4)
    pdf.image(image_path, x=0, y=0, w=QR_width_pdf)
    # Add a blank cell the size of the image
    pdf.cell(QR_width_pdf, QR_width_pdf, txt="", ln=1, align="L")
    if len(human_string) > 8:
        print("barcode-QRcode-generator::add_page_to_PDF:: too many strings in code to display on page")
        pdf.close()
        exit(1)
    # if len(human_string) > 8:
    #     human_string_first_eight = human_string[0:8]
    #     human_string_remainder = human_string[8:]
    #     pdf.set_xy(10, 210)
    #     for line in human_string_first_eight:
    #         pdf.cell(100, 8, txt=line, ln=2, align="L")
    #     pdf.set_xy(105, 210)
    #     for line in human_string_remainder:
    #         pdf.cell(100, 8, txt=line, ln=2, align="L")
    else:
        pdf.set_xy(10, 210)
        for line in human_string:
            pdf.cell(100, 8, txt=line, ln=2, align="L")

    # Go to 1.5 cm from bottom to print footer
    pdf.set_y(-15)
    # Print centered page number
    pdf.cell(0, 10, "Page "+str(pdf.page_no())+" of "+str(total_pages), 0, 1, 'C')
    pdf.set_font("times", size=8)
    pdf.cell(100, 5, txt="QR code generated on " + dt.now().strftime('%Y-%m-%d_%H.%M.%S') + " by Mark Whitty", ln=1, align="L")

    return pdf


def add_title_page(pdf, total_pages, filename, headers):
    pdf.add_page()
    pdf.set_auto_page_break(False)
    pdf.set_font("times", size=16)
    #pdf.cell(100, 8, txt=Document Title, ln=2, align="L")
    pdf.multi_cell(0, 10, txt="This document containing QR codes was generated using the file " +
        filename + " on " + dt.now().strftime('%Y-%m-%d_%H.%M.%S') +
            "\n\nIt contained " + str(len(headers)) + " headers as follows:\n", align="L")

    for header_item in headers:
        pdf.cell(100, 10, txt=header_item, ln = 1, align="L")


def calculate_machine_and_human_string(headers, row_data):
    # Make machine readable content (no error checking done here on string lengths)
    # split_values = [x.strip() for x in row_data.split(',')]
    delimiter = "-"  # Delimiter used between fields in machine readable code
    machine_string = ""
    for j in range(len(headers) - 1):
        # machine_string = machine_string + headers[j] + delimiter + split_values[j] + delimiter
        machine_string = machine_string + headers[j] + delimiter + str(row_data.iloc[j]).strip() + delimiter

    machine_string = machine_string + headers[j + 1] + delimiter + str(row_data.iloc[j + 1]).strip()  # Last item don't add underscore

    human_string = []

    # Make human readable content
    for j in range(len(headers)):
        human_string.append(headers[j] + " = " + str(row_data.iloc[j]))
    # human_string[j+1] = headers[j+1] + " = " + split_values[j+1]  # Last item don't add newline
    
    return machine_string, human_string


def generate_QR_output_files(file_generated_images_output_path, file_qr_labels):
    qr_dataframe = read_qr_labels_from_file(file_qr_labels)
    if qr_dataframe.empty:
        print("Error: No data read from QR code label file")
        exit(1)

    start_time = dt.now()

    header_row = list(qr_dataframe)
    headers = [x.strip() for x in header_row]

    delimiter = "-"  # Delimiter used between fields in machine readable code

    pdf = FPDF()
    pdf.set_compression(True)
    pdf.set_author("Mark Whitty @ UNSW")
    pdf.set_title(os.path.basename(file_qr_labels)[:-4])
    total_pages = qr_dataframe.shape[0]  # Check this index
    add_title_page(pdf, total_pages, os.path.basename(file_qr_labels), headers)

    num_labels_generated = 0
    qr_machine_string_list = []

    # Generate output files based on labels
    for index, qr_row in qr_dataframe.iterrows():
        (machine_string, human_string) = calculate_machine_and_human_string(headers, qr_row)
        qr_machine_string_list.append(machine_string)  # Keep list of generated strings for later comparison

        image_path = generate_QR_codes(machine_string, file_generated_images_output_path)

        add_page_to_PDF(pdf, image_path, total_pages, human_string)
        try:
            os.remove(image_path)
        except OSError:
            pass
        num_labels_generated += 1

    pdf.output(os.path.join(file_generated_images_output_path, (os.path.basename(file_qr_labels[:-4]) + '.pdf')))
    pdf.close()

    # Add machine_strings to the qr_dataframe
    qr_dataframe["QR_Data_"] = qr_machine_string_list

    current_time = dt.now()
    print(str(num_labels_generated) + " images in " + str(((current_time - start_time).seconds)) + " seconds")
    return qr_dataframe


def parse_all_images(file_images_input_path):
    # Read all files in directory that are in image format as allowed by OpenCV
    # https: // docs.opencv.org / 3.0 - beta / modules / imgcodecs / doc / reading_and_writing_images.html
    print(os.path.abspath(file_images_input_path))
    input_image_files = [f for f in os.listdir(file_images_input_path) if
                         re.search(r'.*\.(jpg|png|bmp|dib|jpe|jpeg|jp2|tif|tiff|JPG|PNG|BMP|DIB|JPE|JPEG|JP2|TIF|TIFF)$', f)]
    input_image_files = list(map(lambda x: os.path.join(file_images_input_path, x), input_image_files))
    num_input_image_files = len(input_image_files)

    if num_input_image_files < 1:
        print("Warning: No image files in input directory: ", file_images_input_path)
        exit(0)

    input_image_files_it = 0
    num_correctly_parsed_files = 0

    parsed_images_list = []

    print(str(num_input_image_files), " image files in ", file_images_input_path, " directory")
    if num_input_image_files > 10000:
        print("Warning: Halting execution as more than 10,000 files input. This program is not designed for such a"
              " large volume of images.")
        exit(0)

    # Attempt to decode each image
    for infile in input_image_files:
        # Read image
        im = cv2.imread(infile)
        if im is None:
            print("Warning: image " + infile + " could not be read, trying next file")
            continue
        input_image_files_it = input_image_files_it + 1

        # Check size and if too large, resize it
        resized_im, scaling_factor = resize_max(im, max_image_dimension)

        decodedObject = decode(resized_im, infile)
        if decodedObject is None:
            # If image is not correctly processed, save the status in the dataframe as 1
            parsed_images_list.append([str(input_image_files_it), os.path.abspath(infile), 1, "", "", "", ""])
            continue
        num_correctly_parsed_files = num_correctly_parsed_files + 1
        # display(resized_im, decodedObject)

        # New record, noting that QR code data includes 'b' and two apostraphes, presumably to indicate data type,
        # so these are stripped out
        parsed_images_list.append([str(input_image_files_it), os.path.abspath(infile), 0, str(decodedObject.data)[2:-1],
                                    str(decodedObject.type), str(decodedObject.rect), str(decodedObject.polygon)])

        print("Completed image ", input_image_files_it, " of ", num_input_image_files, " [",
              int(float(input_image_files_it) / float(num_input_image_files) * 100), "%]")

    parsed_images_df = pd.DataFrame(parsed_images_list, columns=["Image number", "Input filename", "Status (0=success)",
                                                                 "QR_Data_", "Type", "Rect", "Polygon"])
    parsed_images_df.to_csv(os.path.join(file_images_input_path, "parsing_log_" +
                                         dt.now().strftime('%Y-%m-%d_%H.%M.%S') + ".txt"))

    # Only return correctly parsed images in the dataframe
    return parsed_images_df.loc[parsed_images_df['Status (0=success)'] == 0]


def check_duplicates(df, col_name):
    df['Duplicated'] = df.duplicated(subset=[col_name])
    print('Duplicated entries')
    print(df[df.duplicated(subset=[col_name])])

    return df


def check_parsed_images_against_original_list(original_qr_df, parsed_images_df):
    # Find entries in original list which have corresponding QR codes in parsed images
    # original_qr_df['image_exists'] = np.where(original_qr_df.QR_Data_ == parsed_images_df.QR_Data_, 'True', 'False')
    # https: // blog.softhints.com / pandas - compare - columns - in -two - dataframes /

    merged_df = pd.merge(original_qr_df, parsed_images_df, how="inner", on="QR_Data_")
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html

    # Add some logic for displaying the differences.
    return merged_df


# def compute_output_filename(qr_data, file_renamed_images_output_path, infile):
#     # Copy resized image and change its filename to match QR code
#
#     # Remove strange characters from filename and replace with underscores
#     # Derived from list of special characters on Windows here:
#     # https://en.wikipedia.org/wiki/Filename#Reserved_characters_and_words
#     qr_data = qr_data.replace('/', '_').replace('\\', '_').replace('?', '_').replace('%', '_') \
#         .replace('*', '_').replace(':', '_').replace('|', '_').replace('\"', '_').replace('<', '_') \
#         .replace('>', '_').replace('.', '_').replace(' ', '_')
#
#
#
#
# def apply_compute_output_filename(qr_df, file_renamed_images_output_path):
#     return compute_output_filename(qr_df['QR_Data_'], file_renamed_images_output_path, qr_df['Input filename'])


def rename_images(file_renamed_images_output_path, merged_df):
    # Create target directory & all intermediate directories if it doesn't exists
    if os.path.exists(file_renamed_images_output_path):
        print("Warning: output directory ", file_renamed_images_output_path,
              " already exists, existing files may be overwritten")

    # Calculate output filename for each image
    # merged_df['output_filename'] = merged_df['QR_Data_'].apply(apply_compute_output_filename,
    #                                                            args=file_renamed_images_output_path)

    for index, merged_image in merged_df.iterrows():
        qr_data = merged_image['QR_Data_']
        qr_data = qr_data.replace('/', '_').replace('\\', '_').replace('?', '_').replace('%', '_') \
            .replace('*', '_').replace(':', '_').replace('|', '_').replace('\"', '_').replace('<', '_') \
            .replace('>', '_').replace('.', '_').replace(' ', '_')
        new_filename = os.path.join(file_renamed_images_output_path,
                                    qr_data + os.path.splitext(merged_image['Input filename'])[1])

        shutil.copy2(merged_image['Input filename'], new_filename)

    # Write out statistics to a text file
    # stats_file = open(os.path.join(file_renamed_images_output_path, "decoding_stats.txt"), "w+")
    # stats_file.write(os.path.basename(__file__) + " executed at " + str(dt.now().isoformat(' ')) + '\n')
    # stats_file.write("Read " + str(input_image_files_it) + " images from " + file_images_input_path + '\n')
    # stats_file.write(
    #     "Correctly parsed QR codes in " + str(num_correctly_parsed_files) + " image(s) and unable to parse " \
    #     + str(input_image_files_it - num_correctly_parsed_files) + " image(s)\n")
    # stats_file.write("Images unable to be parsed have had an underscore prepended to their filename\n")
    # stats_file.write("Wrote correctly parsed files to " + file_renamed_images_output_path + '\n')
    # stats_file.close()

    return


if __name__ == '__main__':
    # Read target and output paths from a GUI (command line operation has been deprecated)
    root = Tk()
    file_qr_labels = filedialog.askopenfilename(title="Location of CSV file containing desired QR codes",
                                                initialdir="csv_input", initialfile="C:\\Users\\z3099851\\OneDrive - UNSW\\Code\\QR_image_renamer\\csv_input\\trial_records_test.csv", parent=root)
    file_generated_images_output_path = filedialog.askdirectory(title="Location where generated image files will be "
                                                                      "placed", initialdir="generated_codes", parent=root)

    file_images_input_path = filedialog.askdirectory(title="Location of images to be decoded",
                                                     initialdir="example_images", parent=root)
    file_renamed_images_output_path = filedialog.askdirectory(title="Location where renamed images will be copied",
                                                              initialdir="renamed_images", parent=root)

    max_image_dimension = 1600  # Maximum image dimension, if greater than this will be resized before processing
     # (output image remained unchanged), as the QR code decoder (pyzbar) has a size limit around 2000 pixels.

    # Check if input directory exists and contains files
    if not does_file_exist_in_dir(file_images_input_path):
        print("Warning: No files in input directory: ", file_images_input_path)
        exit(1)

    original_qr_df = generate_QR_output_files(file_generated_images_output_path, file_qr_labels)
    print("Checking duplicate QR codes from provided CSV")
    check_duplicates(original_qr_df, "QR_Data_")  # Just check for and display duplicated QR codes in input CSV
    # TODO: Remove duplicates in generated QR codes

    parsed_images_df = parse_all_images(file_images_input_path)
    print("Checking duplicates in parsed images")
    parsed_images_duplicates_checked_df = check_duplicates(parsed_images_df, "QR_Data_")
    # Add option to delete duplicate images? Dangerous...should let user choose.

    # Select only the first entry of any duplicates
    parsed_images_duplicates_removed_df = parsed_images_duplicates_checked_df.loc[
        parsed_images_duplicates_checked_df['Duplicated'] == False]

    merged_df = check_parsed_images_against_original_list(original_qr_df, parsed_images_duplicates_removed_df)
    merged_df.to_csv(os.path.join(file_images_input_path, "merged_log_" +
                                  dt.now().strftime('%Y-%m-%d_%H.%M.%S') + ".txt"))

    rename_images(file_renamed_images_output_path, merged_df)