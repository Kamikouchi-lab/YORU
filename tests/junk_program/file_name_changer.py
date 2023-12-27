import os

# フォルダのパスを指定します。
directory = r"C:\Users\nokai\Desktop\mouse_tr\train\labels"

# フォルダ内の各ファイルをループで処理します。
for filename in os.listdir(directory):
    if filename.endswith(".txt"):  # .txtファイルのみを対象にします。
        # 新しいファイル名は "xml"を削除したものです。
        new_filename = filename.replace(".xml", "")
        old_file_path = os.path.join(directory, filename)
        new_file_path = os.path.join(directory, new_filename)

        # ファイル名を変更します。
        os.rename(old_file_path, new_file_path)
