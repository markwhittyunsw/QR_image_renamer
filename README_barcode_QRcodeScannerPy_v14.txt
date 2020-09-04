At the link below is an executable "barcode-QRcodeScannerPy_v14.exe" compiled
for Windows systems which will enable you to process a folder of image files and
have them renamed to match the QR code data. Copies of the decoded and renamed
images are placed in a destination. The input folder and destination folder for 
renamed images are selected by graphical user input windows when no command line
arguments are specified, or if the exe is double-clicked.

If one command line argument is specified, this is assumed to be the location of
the input folder and the destination folder will be a subfolder of that. 

If two command line arguments are specified, the first will be the location of the
input folder and the second will be the destination folder.

Download zip here:
https://unsw-my.sharepoint.com/:u:/g/personal/z3099851_ad_unsw_edu_au/EdGmKrhThglKrpVUFRQMCJsBDHV4SHoCundC4OZhpMny2Q?e=nHWPH1

Usage:
Unzip the zip anywhere

Run the executable on the command line, specifying the location of the folder
containing the jpg files as the first input argument.

The quick way to do this on Windows is to drag the folder onto the .exe and let it go

Alternatively, Windows key, type 'cmd', press enter, then dir to the location of the
executable and run "barcode-QRcodeScannerPy_v14.exe <folder of images>"

It will rename and copy about 2 images per second, and should show progress in a console window.

Feel free to use it anywhere and please send lots of feedback, I'd like to know how it goes.

Mark Whitty
UNSW / Plant and Food Research
Date: 20200904
m.whitty@unsw.edu.au
Source code: https://github.com/markwhittyunsw/QR_image_renamer



