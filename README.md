# QR_image_renamer
Code to rename a folder of images according to the content of a QR code in each image. 
This works together with a bulk QR code generator program: https://github.com/markwhittyunsw/GenerateQRCode
- Author: Mark Whitty UNSW
- m.whitty@unsw.edu.au
- Source code: https://github.com/markwhittyunsw/QR_image_renamer
- All rights reserved

## To run: 
1. Download the barcode-QRcodeScannerPy_v14.zip file from the 'dist' folder above and unzip the .exe file
2. Run the .exe as "barcode-QRcodeScannerPy_v14.exe [\<folder of images\> [\<destination folder for renamed images\>]]"
 - This can be done in any one of at least four ways:
  - Double-click the .exe and select the folder containing images to be renamed from the window that pops up. Then select the folder where you want the renamed images to be saved (original images are not altered) OR
  - In Windows, drag the folder containing images to be renamed onto the .exe and the renamed images will be placed in a subfolder OR
  - Open a command prompt (In Windows, press the Windows key, type 'cmd', press enter, then dir to the location of the exe), type "barcode-QRcodeScannerPy_v14.exe \<folder of images\>" and press enter OR
  - Open a command prompt (In Windows, press the Windows key, type 'cmd', press enter, then dir to the location of the exe), type "barcode-QRcodeScannerPy_v14.exe \<folder of images\> \<destination folder for renamed images\>" to specify by the input folder and destination folder
3. The destination folder will contain the renamed images, as well as a decoding log (decoding_log.txt) which can be used to see which images were successfully renamed and the corresponding filenames and QR code values. This log is tab separated for easy import into Excel. decoding_stats.txt will give brief statistics on the success rate. 
4. Images which were not successfully decoded are copied to the destination folder and prepended by a underscore so they are readily identifiable. Images with non-unique QR code contents have their filenames extended by an incrementing counter, so double-ups can easily be found.

## To compile from sourceInstallation instructions
HINT: Use Terminal in PyCharm to run commands for installing packages using pip (not Python Console)
 - install python 3.7.2 64 bit edition
 - update pip: "python -m pip install --upgrade pip" from location of Python installation using Administrator privileged CMD shell.
 - pip install pyzbar
 - pip install numpy
 - pip install opencv-contrib-python
 - run barcode-QRcodeScannerPy.py to install OpenCV (cv2)
 - pip install pyinstaller

## Using pyinstaller to generate an exe
https://medium.com/dreamcatcher-its-blog/making-an-stand-alone-executable-from-a-python-script-using-pyinstaller-d1df9170e263
pip install pyinstaller

20190326 Fixing bug in pyzbar when trying to use pyinstaller. 
https://github.com/NaturalHistoryMuseum/pyzbar/issues/27
Specifically 
  1) Generate the <project_name>.spec file by trying to run pyinstaller (it may fail or work, or the exe may give an error relating to not being able to find a module names libiconv.dll or libzbar-64.dll)
  2) Edit the .spec file generated by pyinstaller to include the following lines:
    from pathlib import Path
    from pyzbar import pyzbar
    ...
    # dylibs not detected because they are loaded by ctypes
    a.binaries += TOC([
        (Path(dep._name).name, dep._name, 'BINARY')
        for dep in pyzbar.EXTERNAL_DEPENDENCIES
    ])
3) Rerun pyinstaller with the .spec file as an argument (before the .py file)
4) The .exe works! Otherwise put libiconv.dll in the same folder as the dll.

Notice if jpg is created, this might need additional lines in .spec as the github link above shows.

## Progress
 - [x] Make report on QR code error rates and usage
 - [x] What if no QR code? Filename renamed (prefix underscore) 
 - [x] Duplicate QR codes?  Check if name already exists, add incrementing suffix until name is unique
 - [x] Remove strange characters from QR code which could make an invalid filename and replaces them with underscores
 - [x] What if 2 QR codes in one image - only handle the first
 - [x] Generate statistics on images processed 
 - [x] Generate file showing input image list and output file names 
 - [x] Make code distributable (pyinstaller)
 - [ ] Given the list of generated QR codes and processed set of codes should have same filenames, can have a tool to compare them and identify duplicates and missing items.
 - [ ] Encode scale into the QR code, so its distance from the reader is known (assume printed correctly)
 - [ ] Add website detailing the project and a fixed QR code which points to it, automatically generated on the cover of each document.
 - [ ] Plan paper around the use of these codes - submitted
 - [ ] Q: How to make into a Python module?
 - [ ] Q: How to make into a Jupyter notebook? And import the module.
 - [ ] Add salt and pepper noise just on code area. At code resolution.

## Useful resources for QR codes and Python / Jupyter programming
 - https://www.learnopencv.com/barcode-and-qr-code-scanner-using-zbar-and-opencv/
 - https://sourceforge.net/projects/zbar/
 - Generate QR code (https://cran.r-project.org/web/packages/qrcode/index.html)
 - https://www.riverbankcomputing.com/software/pyqt/intro PyQT for integrating C+ Qt framework with Python.
 - http://www.web2py.com/
 - NBViewer.jupyter.org
 - RPy2 library to run R code from within Python. (comparison infographic) https://www.datacamp.com/community/tutorials/r-or-python-for-data-analysis?utm_source=adwords_ppc&utm_campaignid=898687156&utm_adgroupid=48947256715&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=1t1&utm_creative=261400735633&utm_targetid=dsa-473406585795&utm_loc_interest_ms=&utm_loc_physical_ms=1011036&gclid=CjwKCAiAqaTjBRAdEiwAOdx9xp5-tFTtDESFPgomG_VaTJWdtOgl-3eXARrQTyRRkDcWCw4t9DeOBxoCr2oQAvD_BwE

