import os
import tkinter as tk
from tkinter import filedialog
from tqdm import tqdm

def select_directory():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory()
    return directory

def calculate_accuracy(directory):
    total_labels = 0
    correct_predictions = 0

    for file in tqdm(os.listdir(directory)):
        if file.endswith('.png'):
            base_filename = file.split('.')[0]
            label_file = os.path.join(directory, base_filename + '.txt')
            yolo_file = os.path.join(directory, base_filename + '_yolo.txt')

            if os.path.exists(label_file) and os.path.exists(yolo_file):
                with open(label_file, 'r') as lf, open(yolo_file, 'r') as yf:
                    label_lines = lf.readlines()
                    labels =[int(line.split()[0]) for line in label_lines]
                    yolo_label_lines = yf.readlines()
                    yolo_labels =[int(line.split()[0]) for line in yolo_label_lines]

                    total_labels += len(labels)
                    for i, label in enumerate(labels):
                        if label in yolo_labels and len(yolo_labels) > 0:
                            correct_predictions += 1
                            yolo_labels.remove(label)


    print(total_labels)
    print(correct_predictions)
    if total_labels > 0:
        accuracy = correct_predictions / total_labels
        return accuracy, total_labels, correct_predictions
    else:
        return 0

directory = select_directory()
accuracy, total_labels, correct_predictions = calculate_accuracy(directory)
accuracy_message = (f"YOLO Prediction Accuracy: {accuracy:.2%}\n"
                    f"Total ground-truth labels: {total_labels}\n"
                    f"Total correct predictions: {correct_predictions}\n")

# 保存するファイル名を指定
output_filename = os.path.join("C:/Users/nokai/Desktop/zebra_orientation/CalAP/datas/accuracy_datas", "yolo_accuracy_zebra_2000x.txt")

# テキストファイルに書き込む
with open(output_filename, "w") as file:
    file.write(accuracy_message)

print(f"YOLO Prediction Accuracy: {accuracy:.2%}")
print(f"Accuracy saved to {output_filename}")
