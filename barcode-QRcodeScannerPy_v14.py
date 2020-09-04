# Input: Folder of images with QR codes
# Output: Folder with QR codes recognised and files renamed according to QR code content
# Mark Whitty
# UNSW
# v14 20190522 Trying to remove skew from QR codes once outline detected
# v13 20190515 Splitting QR code into quarters to identify which corner doesn't have the locating target.
#  Vastly improved thresholding across the code, as well as reasonably reliably detecting the orientation of the
#  code as well as blanking out the e-reader dimensions properly.
# v12 20190508 Improving feature detection for QR code orientation (still need to remove distortion)
# v11 20190506 Trying to correctly determine the orientation of the QR code, using template matching to find the corners
# v10 20190503 Tweaks for QR reader bounds + improved rotation
# v9 20190415 Trying to read more image file types in than .jpgs
# v8 20190409 Revising to add input folder as command line argument and preparing to run as .exe
#  Also changed area whited out to match angle of a tight rectangle around the QR code with proper scale in each direction
# V7 20190328 Adapting rotation and contrast enhancement for images from phone QR codes in Marlborough lab
# v6 20190207 Removing debugging lines and annotations
# v5 20190204 Integrates rotat ion and working on cropping out QR code + surroundings
# V4 20190131 Wait for user input on images unable to be parsed to debug reasons
# V3 20190131 Generates statistics and relationship between input and output filenames as well as decoded data
# V2 20190125 Just looks for QR codes, and handles file renaming better
# 20190125
# Derived from https://github.com/spmallick/learnopencv/tree/master/barcode-QRcodeScanner

from __future__ import print_function
import math  # For handling degrees and radians

#from shapely.geometry import Polygon  # Removed as Shapely does't properly compile with pyinstaller.
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
import maw_crop_leaf
#import pdb
import re  # Regular expressions used for file type matching
from PIL import Image  # For image rotation
from numpy import array

def decode(im):
    # Find only QR codes (remove the second argument to find other symbols)
    decodedObjects = pyzbar.decode(im, symbols=[ZBarSymbol.QRCODE])

    # Print results
    # Check QR code exists in image
    if len(decodedObjects) > 0:
        max_size = 0
        #max_polygon_index = 0
        for index, obj in enumerate(decodedObjects):
            # Check it is a QR code
            if obj.type != 'QRCODE':
                continue
            # Find first QR code and process it not just the largest one, as Shapely used for Polygon doesn't compile correctly using pyinstaller
            # Find largest object and treat it as the desired QR code
            # new_poly = Polygon(obj.polygon)
            #if new_poly.area > max_size:
            #    max_size = new_poly.area
            #    max_polygon_index = index
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

# Switch error reason based on human input
def switch_error(key):
    switcher = {
        '1': "Insufficient contrast",
        '2': "Variable lighting across code",
        '3': "Occlusion of code",
        '4': "Code bent",
        '5': "Code angle too acute",
        '6': "No code in image",
        '7': "Out of focus",
        '8': "Unknown"
    }
    return switcher.get(key, "Invalid key")

# Switch direction vector from QR corner detected
def switch_corner_orientation(corners):
    switcher = {
        0: 180,  # (0, -1), # Top left
        1: -90, # (-1, 0), # Top right
        2: 90,  # (1, 0), # Bottom left
        3: 0,  #(0, 1)
    }
    return switcher.get(corners, 0)  #(0,0))


# Test out ORB keypoints and draw them
def test_ORB(image):
    # Initiate ORB detector
    orb = cv2.ORB_create()
    #fast = cv2.FastFeatureDetector_create()
    # find the keypoints with ORB
    kp = orb.detect(image, None)
    # compute the descriptors with ORB
    kp, des = orb.compute(image, kp)
    # draw only keypoints location,not size and orientation
    for marker in kp:
        image = cv2.drawMarker(image, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
    #fast_image = cv2.drawKeypoints(th3, kp, color=(255, 0, 0))
    if True:
        cv2.imshow("ORB features", image)
        cv2.waitKey(0)
    return image

# Test out FAST keypoints and draw them
# https://docs.opencv.org/3.1.0/df/d0c/tutorial_py_fast.html
def test_FAST(image):
    # Initiate FAT detector
    fast = cv2.FastFeatureDetector_create()
    # find the keypoints with FAST
    kp = fast.detect(image, None)
    # draw only keypoint location, not size and orientation
    for marker in kp:
        image = cv2.drawMarker(image, tuple(int(i) for i in marker.pt), color=(0, 255, 0))
    if True:
        cv2.imshow("FAST features", image)
        cv2.waitKey(0)
    return image



# Rotate an image about its centre by a given number of degrees,
# not handling scaling and cropping
def maw_rotate_image(im, angle):
    #(h, w) = im.shape[:2]
    #centre = (w / 2, h / 2)
    #M = cv2.getRotationMatrix2D(centre, angle, 1.0)
    #im_rotated = cv2.warpAffine(im, M, (w, h))

    im2 = Image.fromarray(im)
    im_rotated = im2.rotate(angle, expand=True)
    im_rotated = array(im_rotated)
    return im_rotated

# https://stackoverflow.com/questions/32328179/opencv-3-0-python-lineiterator
def createLineIterator(P1, P2, img):
    # Produces and array that consists of the coordinates and intensities of each pixel in a line between two points
    #
    # Parameters:
    #     -P1: a numpy array that consists of the coordinate of the first point (x,y)
    #     -P2: a numpy array that consists of the coordinate of the second point (x,y)
    #     -img: the image being processed
    #
    # Returns:
    #     -it: a numpy array that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y,intensity])
    #
   #define local variables for readability
   imageH = img.shape[0]
   imageW = img.shape[1]
   P1X = P1[0]
   P1Y = P1[1]
   P2X = P2[0]
   P2Y = P2[1]

   #difference and absolute difference between points
   #used to calculate slope and relative location between points
   dX = P2X - P1X
   dY = P2Y - P1Y
   dXa = np.abs(dX)
   dYa = np.abs(dY)

   #predefine numpy array for output based on distance between points
   itbuffer = np.empty(shape=(np.maximum(dYa,dXa),3),dtype=np.float32)
   itbuffer.fill(np.nan)

   #Obtain coordinates along the line using a form of Bresenham's algorithm
   negY = P1Y > P2Y
   negX = P1X > P2X
   if P1X == P2X: #vertical line segment
       itbuffer[:,0] = P1X
       if negY:
           itbuffer[:,1] = np.arange(P1Y - 1,P1Y - dYa - 1,-1)
       else:
           itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
   elif P1Y == P2Y: #horizontal line segment
       itbuffer[:,1] = P1Y
       if negX:
           itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
       else:
           itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
   else: #diagonal line segment
       steepSlope = dYa > dXa
       if steepSlope:
           slope = dX.astype(np.float32)/dY.astype(np.float32)
           if negY:
               itbuffer[:,1] = np.arange(P1Y-1,P1Y-dYa-1,-1)
           else:
               itbuffer[:,1] = np.arange(P1Y+1,P1Y+dYa+1)
           itbuffer[:,0] = (slope*(itbuffer[:,1]-P1Y)).astype(np.int) + P1X
       else:
           slope = dY.astype(np.float32)/dX.astype(np.float32)
           if negX:
               itbuffer[:,0] = np.arange(P1X-1,P1X-dXa-1,-1)
           else:
               itbuffer[:,0] = np.arange(P1X+1,P1X+dXa+1)
           itbuffer[:,1] = (slope*(itbuffer[:,0]-P1X)).astype(np.int) + P1Y

   #Remove points outside of image
   colX = itbuffer[:,0]
   colY = itbuffer[:,1]
   itbuffer = itbuffer[(colX >= 0) & (colY >=0) & (colX<imageW) & (colY<imageH)]

   #Get intensities from img ndarray
   itbuffer[:,2] = img[itbuffer[:,1].astype(np.uint),itbuffer[:,0].astype(np.uint)]

   return itbuffer

# Get indices of n largest values in multidimensional array
# https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
def largest_indices(ary, n):
    # """Returns the n largest indices from a numpy array."""
    flat = ary.flatten()
    indices = np.argpartition(flat, -n)[-n:]
    indices = indices[np.argsort(-flat[indices])]
    return np.unravel_index(indices, ary.shape)


def swap_xy_coords(coords):
    for x, y in coords:
        yield (y, x)

# Given a downscaled image in which a QR code has been correctly found
# crop the original image to not include the QR code or a proportion of the
# area around it (QR_dead_zone).
# QR_dead_zone is a factor of the distance from the point to the centroid of the polygon
# 0 leaves this dead done unchanged, 1 expands it a lot. Suggest using 0.5
# QR_dead_zone_w refers to scaling in x direction, _h in y direction, take care with rotated images.
# Note this ignores the orientation of the code itself
def maw_crop_out_QR_code(im, image_scale, QR_polygon, QR_dead_zone_w, QR_dead_zone_h):
    # If the points do not form a quad, find convex hull
    if len(QR_polygon) > 4:
        hull = cv2.convexHull(np.array([point for point in QR_polygon], dtype=np.float32))
        hull = list(map(tuple, np.squeeze(hull)))
    else:
        hull = QR_polygon

    # Number of points in the convex hull
    n = len(hull)

    hull2 = []

    # Apply opposite of scaling factor to original image to allow
    # for the fact pyzbar library has a image size limit and
    # images have been scaled to be less than this, so the
    # polygon needs to be scaled up
    for point in hull:
        hull2.append((int(np.floor(point.x/image_scale)), int(np.floor(point.y/image_scale))))

    # New square for transformed image
    side_length_x = 1000
    side_length_y = 1000
    offset = 100
    square_300 = [(0, 0), (0, side_length_y), (side_length_x, side_length_y), (side_length_x, 0)]
    square_300_offset = [(0+offset, 0+offset), (0+offset, side_length_y+offset), (side_length_x+offset, side_length_y+offset), (side_length_x+offset, 0+offset)]

    # Draw the convex hull
    for j in range(0, n):
        cv2.line(im, hull2[j], hull2[(j + 1) % n], (255, 0, 0), 3)
        cv2.putText(im, str(j), hull2[j], cv2.FONT_HERSHEY_COMPLEX, 2, [0, 0, 255], 0)
        cv2.putText(im, str(j), square_300_offset[j], cv2.FONT_HERSHEY_COMPLEX, 3, [0, 255, 0], 0)

    # Remove perspective transformation (doesn't allow for curvature)
    # What about order of the points, will this cause unexpected rotations?
    pts1 = np.float32(hull2)

    #pts1 = array([[coord[1], coord[0]] for coord in pts1])
    pts2 = np.float32(square_300)
    #pts2 = array([[coord[1], coord[0]] for coord in pts2])

    M = cv2.getPerspectiveTransform(pts1, pts2)

    # Display results
    cv2.namedWindow("before transform", cv2.WINDOW_AUTOSIZE)
    before_pt = cv2.resize(im, dsize=(400, 400), dst=None, fx=0, fy=0, interpolation=cv2.INTER_AREA)
    cv2.imshow("before transform", before_pt)

    im_unwarped = cv2.warpPerspective(im, M, (side_length_x, side_length_y))

    cv2.namedWindow("after transform", cv2.WINDOW_AUTOSIZE)
    after_pt = cv2.resize(im_unwarped, dsize=(400, 400), dst=None, fx=0, fy=0, interpolation=cv2.INTER_AREA)
    cv2.imshow("after transform", after_pt)
    cv2.waitKey(0)


    # Determine the rotation angle of the QR code, looking at the original pixels


    (h, w) = im.shape[:2]

    # Find a tight rectangle around the convex hull to also get rotation angle
    (x_centroid, y_centroid), (box_w, box_h), angle = cv2.minAreaRect(np.int32(hull2))



    # For some reason the angle returned is negative (probably due to swapped y axis direction)
    # Possible due to rotating a box by the opposite angle is needed.
    angle = math.radians(angle)
    x_centroid = int(x_centroid)
    y_centroid = int(y_centroid)
    #cv2.drawMarker(im, (x_centroid, y_centroid), (0, 0, 255), cv2.MARKER_CROSS, 20, 5)

    box_h = int(box_h)
    box_w = int(box_w)

    # Make a new box fitting to polygon
    qr_box = np.zeros((4, 2), dtype=np.int64)
    qr_box[0, 0] = x_centroid - box_h/2  # Bottom left point
    qr_box[0, 1] = y_centroid - box_w/2
    qr_box[1, 0] = x_centroid - box_h/2  # Top left point
    qr_box[1, 1] = y_centroid + box_w/2
    qr_box[2, 0] = x_centroid + box_h/2  # Top right point
    qr_box[2, 1] = y_centroid + box_w/2
    qr_box[3, 0] = x_centroid + box_h/2  # Bottom right point
    qr_box[3, 1] = y_centroid - box_w/2

    qr_hull = []
    # Rotate points by tight rectangle box angle to match QR code
    for row in qr_box:
        x, y = row[0], row[1]
        tx, ty = x - x_centroid, y - y_centroid
        new_x = ( tx*np.cos(-angle) + ty*np.sin(-angle)) + x_centroid  # Could use clip here but might have strange effects at corners in some cases
        new_y = (-tx*np.sin(-angle) + ty*np.cos(-angle)) + y_centroid
        qr_hull.append((int(new_x), int(new_y)))


    # Mask image, check order of w and h
    mask_qr = np.ones([h, w], dtype=np.uint8)
    # Set mask to all values in this bounding box
    qrh = np.array([list(qr_hull)], dtype=np.int32)

    cv2.fillPoly(mask_qr, qrh, 255)
    qrim = cv2.copyTo(im, mask_qr)
    qrim[np.where(mask_qr != 255)] = [0, 0, 0]  # 0 needed as rotation will add zeros

    # Rotate image by QR angle
    M = cv2.getRotationMatrix2D((x_centroid, y_centroid), math.degrees(angle), 1.0)
    qr_rotated = cv2.warpAffine(qrim, M, (w, h))

    # Extract QR code from rotated image
    a = np.where(qr_rotated != 0)
    qr2 = qr_rotated[np.min(a[0]):np.max(a[0]), np.min(a[1]):np.max(a[1]), :]

    # Threshold QR code
    gray_qr_code = cv2.cvtColor(qr2, cv2.COLOR_BGR2GRAY)
    # cv2.imshow("grayscale", gray_qr_code)
    #blur = cv2.GaussianBlur(gray_qr_code, (3, 3), 0)  # Blur used to get better threshold result with Otsu (although Triangle method used)
    #t1 = cv2.equalizeHist(gray_qr_code)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(4, 4))
    t1 = clahe.apply(gray_qr_code)
    # cv2.imshow("adaptive equalised", t1)
    ret3, th3 = cv2.threshold(t1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_TRIANGLE)

    #th3 = cv2.adaptiveThreshold(t1, 250, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 511, 0)

    # Display results
    # cv2.namedWindow("Results 2", cv2.WINDOW_AUTOSIZE)
    #smaller_image_qr = cv2.resize(th3, dsize=(40, 40), dst=None, fx=0, fy=0, interpolation=cv2.INTER_NEAREST)
    #larger_image_qr = cv2.resize(smaller_image_qr, dsize=(400, 400), dst=None, fx=0, fy=0, interpolation=cv2.INTER_AREA)
    # cv2.imshow("Results 2", th3)
    #print("threshold = ", ret3)
    #cv2.waitKey(0)

    # ------------------------------------------------
    # Test out SIFT feature for finding corners, needs opencv_contrib installed
    #sift = cv2.xfeatures2d.SIFT_create()
    #kp = sift.detect(th3, None)
    #img_with_sift = cv2.drawKeypoints(th3, kp,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    #cv2.imshow("SIFT features", img_with_sift)
    #cv2.waitKey(0)

    #th3 = test_ORB(th3)
    #th3 = test_FAST(th3)

    # ------------------------------------------------
    # Do template matching to try and find which corner of the code doesn't match the template
    #square_11311_template = np.zeros((7, 7), dtype=np.uint8)

    square_11311_template = np.array([[1,1,1,1,1,1,1], [1,0,0,0,0,0,1], [1,0,1,1,1,0,1], \
        [1,0,1,1,1,0,1], [1,0,1,1,1,0,1], [1,0,0,0,0,0,1], [1,1,1,1,1,1,1]], dtype=np.uint8)
    square_11311_template = 1 - square_11311_template
    # square_template_edges = cv2.cvtColor(square_11311_template, cv2.COLOR_BGR2GRAY)
    square_11311_template = Image.fromarray(square_11311_template)
    square_11311_template = cv2.cvtColor(np.array(square_11311_template), cv2.COLOR_RGB2BGR)
    template = cv2.cvtColor(square_11311_template, cv2.COLOR_BGR2GRAY)

    gray = th3

    # Add some extra white border as sometimes the edge is slightly cropped off and doesn't match the template well enough
    border_added = 10
    gray_with_border = 255*np.ones((gray.shape[0]+border_added*2, gray.shape[1]+border_added*2), dtype=np.uint8)
    gray_with_border[border_added:(border_added+gray.shape[0]), border_added:(border_added+gray.shape[1])] = gray
    gray = gray_with_border
    th3 = gray
    #cv2.imshow("QR with border size "+str(border_added), gray_with_border)
    # Image conversions here, for playing with Canny
    # https://www.quora.com/How-do-I-plot-a-grayscale-image-with-a-2D-array-of-random-numbers-in-Python
    # https://blog.extramaster.net/2015/07/python-converting-from-pil-to-opencv-2.html
    #square_template_edges = cv2.Canny(square_11311_template, 0, 255, apertureSize=3)
    #template = cv2.Canny(template, 50, 200)
    (tH, tW) = template.shape[:2]
    #cv2.imshow("Template", square_template_edges)

    # Multiscale template matching:
    # https://www.pyimagesearch.com/2015/01/26/multi-scale-template-matching-using-python-opencv/
    found = None

    # loop over the scales of the image
    #for scale in np.linspace(7/1000, 7/40, 30)[::-1]:
    for scale in np.linspace(7/500, 7/40, 50)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        #resized = cv2.resize(gray, width=int(gray.shape[1] * scale))
        # Using interpolation cv2.INTER_NEAREST gives crisper results when downsizing regular patterns, helps
        # avoid the problem of QR2 codes 4th corner matching the pattern better than the actual corner
        resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        r = gray.shape[1] / float(resized.shape[1])

        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        #edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)

        # check to see if the iteration should be visualized
        if False:
            # draw a bounding box around the detected region
            #clone = np.dstack([edged, edged, edged])
            clone = np.copy(resized)
            clone = cv2.cvtColor(clone, cv2.COLOR_GRAY2BGR)
            cv2.rectangle(clone, (maxLoc[0], maxLoc[1]),
                          (maxLoc[0] + tW, maxLoc[1] + tH), (0, 0, 255), 1)
            cv2.imshow("Visualize", clone)
            cv2.waitKey(0)

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    if found is None:
        print("Unable to find a matching corner at any scale")
        return

    # unpack the bookkeeping variable and compute the (x, y) coordinates
    # of the bounding box based on the resized ratio
    (_, maxLoc, r) = found


    # Repeat the template matching to get all matching corners at this paticular scale
    resized = cv2.resize(gray, None, fx=1/r, fy=1/r, interpolation=cv2.INTER_AREA)
    #Use inter_area as interpolation once scale known, as initially low res images scale poorly with inter_nearest

    # Pull out corners and test each at the best scale found in the full code template match
    quarter_h = int(resized.shape[0]/2)  # Height is the first element of the size vector
    quarter_w = int(resized.shape[1]/2)
    top_left_QR = resized[0:quarter_h, 0:quarter_w]
    top_right_QR = resized[0:quarter_h, (resized.shape[1]-quarter_w):]
    bottom_left_QR = resized[(resized.shape[0]-quarter_h):, 0:quarter_w]
    bottom_right_QR = resized[(resized.shape[0]-quarter_h):, (resized.shape[1]-quarter_w):]



    # Match separate corners
    match_top_left = cv2.matchTemplate(top_left_QR, template, cv2.TM_CCOEFF_NORMED)
    match_top_right = cv2.matchTemplate(top_right_QR, template, cv2.TM_CCOEFF_NORMED)
    match_bottom_left = cv2.matchTemplate(bottom_left_QR, template, cv2.TM_CCOEFF_NORMED)
    match_bottom_right = cv2.matchTemplate(bottom_right_QR, template, cv2.TM_CCOEFF_NORMED)
    max_matches = [np.max(match_top_left), np.max(match_top_right), np.max(match_bottom_left), np.max(match_bottom_right)]
    min_match_corner = np.argmin(max_matches)
    print("min_match_corner = ", min_match_corner, " r = ", r)

    code_orientation = switch_corner_orientation(min_match_corner)
    # Rotate original image by QR angle
    #M = cv2.getRotationMatrix2D((quarter_h, quarter_w), code_orientation, 1.0)
    #qr_rotated_final = cv2.warpAffine(resized, M, (resized.shape[1], resized.shape[0]))
    #cv2.imshow("Fixed rotation", qr_rotated_final)

    # Rotate image by QR angle
    total_QR_angle_deg = math.degrees(angle) + code_orientation
    total_QR_angle_rad = math.radians(total_QR_angle_deg)
    # Show result on full image to confirm
    im_rotated = maw_rotate_image(im, total_QR_angle_deg)
    im_rotated_resized, scale__ = resize_max(im_rotated, 500)
    # cv2.imshow("Full image rotated", im_rotated_resized)

    #ktable, #rmarkdown, nice documentation.


    # cv2.imshow("top left QR", top_left_QR)
    # cv2.imshow("top right QR", top_right_QR)
    # cv2.imshow("bottom left QR", bottom_left_QR)
    # cv2.imshow("bottom right QR", bottom_right_QR)

    #disp("match_top_left = " + match_top_left)
    #print("match_top_right = " + match_top_right)
    #print("match_bottom_left = " + match_bottom_left)
    #print("match_bottom_right = " + match_bottom_right)


    # Match corners in complete QR code
    # result = cv2.matchTemplate(resized, template, cv2.TM_CCOEFF_NORMED)


    # Adjusting threshold to try and remove false positives
    #threshold = 0.6
    #result[result < threshold] = 0
    #loc = np.where(result >= threshold)

    ##top_3_matches = (-result).argsort(axis=-1)[:, :3]
    ##top_3_matches = result[result.argsort()[-3:]]
    #top_3_matches = np.transpose(largest_indices(result, 3))

    # th3 = cv2.cvtColor(th3, cv2.COLOR_GRAY2BGR)

    # # draw a bounding box around all the detected results and display the image
    # # for pt in zip(*loc[::-1]):
    # for pt in top_3_matches:
    #     (startX, startY) = (int(pt[0] * r), int(pt[1] * r))
    #     (endX, endY) = (int((pt[0] + tW) * r), int((pt[1] + tH) * r))
    #     # Note that Y and X have been swapped, following use of np.transpose on output of largest_indices
    #     cv2.rectangle(th3, (startY, startX), (endY, endX), (0, 0, 255), 2)
    # cv2.imshow("Results template matching", th3)

    #cv2.waitKey(0)

    #cv2.destroyAllWindows()

    # Interpolate to find centres of corner segments

    #rect = cv2.minAreaRect(np.int32(hull2))
    #box = cv2.boxPoints(rect)
    #box = np.int64(box)
    #print("box_width = " + str(box_w) + " box height = " + str(box_h) + " angle = " + str(angle))
    # Make an expanded box with length along x axis and width on y axis
    box = np.zeros((4, 2), dtype=np.int64)
    box[0, 0] = x_centroid - box_h * (1+QR_dead_zone_h) / 2  # Bottom left point
    box[0, 1] = y_centroid - box_w * (1+QR_dead_zone_w) / 2
    box[1, 0] = x_centroid - box_h * (1+QR_dead_zone_h) / 2  # Top left point
    box[1, 1] = y_centroid + box_w * (1+QR_dead_zone_w) / 2
    box[2, 0] = x_centroid + box_h * (1+QR_dead_zone_h) / 2  # Top right point
    box[2, 1] = y_centroid + box_w * (1+QR_dead_zone_w) / 2
    box[3, 0] = x_centroid + box_h * (1+QR_dead_zone_h) / 2  # Bottom right point
    box[3, 1] = y_centroid - box_w * (1+QR_dead_zone_w) / 2

    expanded_hull = []
    # Rotate points by tight rectangle box angle to match QR code
    for row in box:
        x, y = row[0], row[1]
        tx, ty = x - x_centroid, y - y_centroid
        new_x = ( tx*np.cos(-total_QR_angle_rad) + ty*np.sin(-total_QR_angle_rad)) + x_centroid  # Could use clip here but might have strange effects at corners in some cases
        new_y = (-tx*np.sin(-total_QR_angle_rad) + ty*np.cos(-total_QR_angle_rad)) + y_centroid
        expanded_hull.append((int(new_x), int(new_y)))

    # Draw the expanded polygon in green
    #for j in range(0, n):
    #    cv2.line(im, expanded_hull[j], expanded_hull[(j + 1) % n], (0, 255, 0), 3)

    # blank mask:
    expanded_mask = np.zeros([h, w], dtype=np.uint8)
    # Set all values in this bounding box to white
    eh = np.array([list(expanded_hull)], dtype=np.int32)
    cv2.fillPoly(expanded_mask, eh, 255)
    im[np.where(expanded_mask == 255)] = [255, 255, 255]

    # Display results
    #cv2.namedWindow("Results", cv2.WINDOW_AUTOSIZE)
    #smaller_imager = cv2.resize(im, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
    #cv2.imshow("Results", smaller_imager)
    #cv2.waitKey(0)
    return im

# Main
if __name__ == '__main__':

    # Read destination path
    if len(sys.argv) < 2:
        file_input_path = "."
    else:
        file_input_path = sys.argv[1]

    #file_input_path = "C:/Users/z3099851/PycharmProjects/QR_image_reader/20190328 Rowley SBL images"
    file_output_path = os.path.join(file_input_path, os.path.basename(os.path.abspath(file_input_path)) + "_decoded_blanked/")
    max_image_dimension = 1600  # Maximum image dimension, if greater than this will be resized before processing (output image is also resized)

    # Check if input directory exists and contains files
    if not os.path.exists(file_input_path):
        print("Input directory ", file_input_path, " does not exist")
        exit(1)

    if not does_file_exist_in_dir(file_input_path):
        print("Warning: No files in input directory: ", file_input_path,)
        exit(1)

    # Read all files in directory that are in image format as allowed by OpenCV
    # https: // docs.opencv.org / 3.0 - beta / modules / imgcodecs / doc / reading_and_writing_images.html
    # input_files = glob.glob(os.path.join(file_input_path, "*.jpg"))
    print(os.path.abspath(file_input_path))
    input_files = [f for f in os.listdir(file_input_path) if re.search(r'.*\.(jpg|png|bmp|dib|jpe|jpeg|jp2|tif|tiff)$', f)]
    input_files = list(map(lambda x: os.path.join(file_input_path, x), input_files))
    num_input_files = len(input_files)

    if num_input_files < 1:
        print("Warning: No image files in input directory: ", file_input_path)
        exit(0)

    # Create target directory & all intermediate directories if don't exists
    if not os.path.exists(file_output_path):
        os.makedirs(file_output_path)
        print("Output directory ", file_output_path, " created ")
    else:
        print("Warning: directory ", file_output_path, " already exists, existing files may be overwritten")

    input_files_it = 0
    num_correctly_parsed_files = 0

    log_file = open(os.path.join(file_output_path, "decoding_log.txt"), "w+")
    log_file.write("Image number\tInput filename\tOutput filename\tStatus (0 = correctly read and parsed QR code, " +
                   "1 = Insufficient contrast, 2 = Variable lighting across code, 3 = Occlusion of code, " +
                   "4 = Code bent, 5 = Code angle too acute, 6 = No code in image, 7 = Out of focus" +
                   "8 = Unknown)" +
                   "\tData\tType\tRect\tPolygon\n")



    print(str(num_input_files), " image files in ", file_input_path, " directory")
    for infile in input_files:
        # Read image
        im = cv2.imread(infile)
        if im is None:
            print("Warning: image " + infile + " could not be read")
            continue
        input_files_it = input_files_it + 1

        #if(input_files_it < 763 | input_files_it > 783):
        #    continue
        if(input_files_it > 10000):
            print("Warning: Halting execution as more than 10,000 files input")
            break
        # Rotate the image if required (only works for 180 degrees at present)
        im = maw_rotate_image(im, 270)

        # Check size and if too large, resize it
        # Check size and if too large, resize it
        resized_im, scaling_factor = resize_max(im, max_image_dimension)

        decodedObject = decode(resized_im)
        if decodedObject is None:
            # If image is not correctly processed, ignore user input
            # If image not correctly processed, display it and wait for user input
            #cv2.imshow("Resized image " + str(os.path.basename(infile)), resized_im)
            #key = cv2.waitKey()
            #cv2.destroyAllWindows()

            # Copy file as is and append '_' to filename. This will overwrite any existing files with this name (which are presumably useless)
            new_filename = os.path.join(file_output_path, ('_' + os.path.basename(infile)))
            shutil.copy2(infile, new_filename)
            log_file.write(str(input_files_it) + "\t" + os.path.basename(infile) + "\t" + "_" + os.path.basename(infile) + "\t" + "8" + "\t" + str(switch_error("8")) + "\n")

            continue
        num_correctly_parsed_files = num_correctly_parsed_files + 1
        #display(resized_im, decodedObject)

        # Crop out QR code related section of the image or use cropped_image = im
        # For common leaf images from Hawkes bay, (1.3, 2) are suitable arguments for a landscape aligned e-reader
        # (1.8, 1.25) are ok for revised rotated blanking of e-reader
        cropped_image = maw_crop_out_QR_code(im, scaling_factor, decodedObject.polygon, 1.8, 1.25)


        # Copy resized image and change its filename to match QR code
        # New filename, noting that QR code data includes 'b' and two apostraphes, presumably to indicate data type, so these are stripped out
        qr_data = str(decodedObject.data)[2:-1]

        # Remove strange characters from filename and replace with underscores
        # Derived frm list of special characters on Windows here: https://en.wikipedia.org/wiki/Filename#Reserved_characters_and_words
        qr_data = qr_data.replace('/', '_').replace('\\', '_').replace('?', '_').replace('%', '_').replace('*', '_').replace(':', '_').replace('|', '_').replace('\"', '_').replace('<', '_').replace('>', '_').replace('.', '_').replace(' ', '_')

        new_filename = file_output_path + qr_data + os.path.splitext(infile)[1]
        file_iterator = 0

        # Beware of filenames with strange characters
        if os.path.exists(new_filename):
            # File already exists
            while os.path.exists(new_filename):
                new_filename = file_output_path + qr_data + "_" + str(file_iterator).zfill(3) + os.path.splitext(infile)[1]
                file_iterator = file_iterator+1
        cv2.imwrite(new_filename, cropped_image)
        #shutil.copy2(infile, new_filename)
        log_file.write(str(input_files_it) + "\t" + os.path.basename(infile) + "\t" +
                       os.path.basename(new_filename) + "\t0\t" + str(decodedObject.data) + "\t" +
                       str(decodedObject.type) + "\t" + str(decodedObject.rect) + "\t" +
                       str(decodedObject.polygon) + "\t\n")
        print("Completed image ", input_files_it, " of ", num_input_files, " [", int(float(input_files_it)/float(num_input_files)*100), "%]")
    # Write out statistics to a text file
    stats_file = open(os.path.join(file_output_path, "decoding_stats.txt"), "w+")
    stats_file.write(os.path.basename(__file__) + " executed at " + str(datetime.now().isoformat(' ')) + '\n')
    stats_file.write("Read " + str(input_files_it) + " images from " + file_input_path + '\n')
    stats_file.write("Correctly parsed QR codes in " + str(num_correctly_parsed_files) + " image(s) and unable to parse " + str(input_files_it-num_correctly_parsed_files) + " image(s)\n")
    stats_file.close()

    log_file.close()
