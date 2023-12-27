import os
import shutil
import tkinter as tk
from tkinter import filedialog


def select_folder(title):
    root = tk.Tk()
    root.withdraw()  # Hide the main window
    folder_path = filedialog.askdirectory(title=title)
    root.destroy()
    return folder_path


# Select folders A, B, and C using file dialogs
folder_a = select_folder("Select Folder A")
folder_b = select_folder("Select Folder B")
folder_c = select_folder("Select Folder C")

# Create folder C if it doesn't exist
os.makedirs(folder_c, exist_ok=True)

# Compare files in folder A and B, and copy unique files from A to C
for filename in os.listdir(folder_a):
    if filename not in os.listdir(folder_b):
        shutil.copy(os.path.join(folder_a, filename), folder_c)

print("Files copied successfully.")
