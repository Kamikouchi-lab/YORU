import os
import random
import shutil
import tkinter as tk
from tkinter import filedialog


def select_folder(title):
    # Open a dialog to select a folder
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_selected = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_selected


def copy_random_sets(num_sets=200):
    # Ask the user to select the source and destination folders
    src_folder = select_folder("Select Source Folder")
    if not src_folder:
        print("Source folder selection cancelled.")
        return

    dest_folder = select_folder("Select Destination Folder")
    if not dest_folder:
        print("Destination folder selection cancelled.")
        return

    # Get all files in the source folder
    all_files = os.listdir(src_folder)

    # Filter out files to get a list of unique base filenames without extensions
    base_filenames = set(os.path.splitext(file)[0] for file in all_files)

    # Select a random subset of these base filenames
    selected_sets = random.sample(base_filenames, min(num_sets, len(base_filenames)))

    # Copy the selected sets to the destination folder
    for base_name in selected_sets:
        for extension in [".png", ".txt"]:
            src_file = os.path.join(src_folder, base_name + extension)
            dest_file = os.path.join(dest_folder, base_name + extension)
            shutil.copy(src_file, dest_file)

    print(f"Copied {num_sets} sets from '{src_folder}' to '{dest_folder}'.")


# Example usage
copy_random_sets()

# This updated script uses tkinter to allow the user to select the source and destination folders through a GUI.
# The script will then copy 100 random sets (each set consisting of a .png and .txt file with the same name)
# from the selected source folder to the selected destination folder.
