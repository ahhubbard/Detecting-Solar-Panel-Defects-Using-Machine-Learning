# Program to rename images in a folder for AI naming schemes

import os
import re
import tkinter
from tkinter import filedialog
from tkinter.simpledialog import askstring

root = tkinter.Tk()


def resetnames():
    file_name = 'an2351dsa3wf_no_one_would_name_a_file_like_this_3r20mng9a_'
    i = 0
    for file in os.listdir(files):
        if not file.startswith('.'):
            # Check if file already has the target filename format
            if not file.startswith(file_name) or not file.endswith('.jpg'):
                src = os.path.join(files, file)
                # Find the highest index used so far
                m = re.search(rf"{file_name}(\d+)\.jpg", file)
                if m:
                    i = max(i, int(m.group(1)))
                dst = os.path.join(files, file_name + str(i) + ".jpg")
                try:
                    os.rename(src, dst)
                    i += 1
                except FileExistsError:
                    print("File already exists")
                    while True:
                        i += 1
                        dst = os.path.join(files, file_name + str(i) + ".jpg")
                        if not os.path.exists(dst):
                            os.rename(src, dst)
                            break


def multi_filename_change():
    # After the directory is opened, entered the name for all the images (ex: "Good_" or "Bad_")
    file_name = askstring('Add file name', 'Please enter a  name for your files.')

    # Iterate through each file
    i = 0
    for file in os.listdir(files):
        if not file.startswith('.'):
            # Check if file already has the target filename format
            if not file.startswith(file_name) or not file.endswith('.jpg'):
                src = os.path.join(files, file)
                # Find the highest index used so far
                m = re.search(rf"{file_name}(\d+)\.jpg", file)
                if m:
                    i = max(i, int(m.group(1)))
                dst = os.path.join(files, file_name + str(i) + ".jpg")
                try:
                    os.rename(src, dst)
                    i += 1
                except FileExistsError:
                    print("File already exists")
                    while True:
                        i += 1
                        dst = os.path.join(files, file_name + str(i) + ".jpg")
                        if not os.path.exists(dst):
                            os.rename(src, dst)
                            break


# This will ask for the directory folder of the images, click through until you reach it
files = filedialog.askdirectory()
resetnames()
multi_filename_change()
