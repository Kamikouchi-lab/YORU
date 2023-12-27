import os
import shutil
from tkinter import filedialog


def process_files(input_folder, output_folder):
    # 結果を保存するフォルダを作成
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 入力フォルダ内のファイルを処理
    for filename in os.listdir(input_folder):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        # classes.txtの場合は編集せずにコピー
        if filename == "classes.txt":
            shutil.copy(input_path, output_path)
        elif filename.endswith(".txt"):
            # テキストファイルの場合
            with open(input_path, "r", encoding="utf-8") as infile:
                # 各行に対して変換を行う
                lines = infile.readlines()
                lines = [
                    "0" + line[1:] if line.startswith("1") else line for line in lines
                ]

            # 結果を新しいファイルに保存
            with open(output_path, "w", encoding="utf-8") as outfile:
                outfile.writelines(lines)
        elif filename.endswith(".png"):
            # 画像ファイルの場合はそのままコピー
            shutil.copy(input_path, output_path)


if __name__ == "__main__":
    # 入力フォルダと出力フォルダを指定
    input_folder = filedialog.askdirectory(title="input_folder")
    output_folder = filedialog.askdirectory(title="output_folder")

    # ファイルを処理
    process_files(input_folder, output_folder)
