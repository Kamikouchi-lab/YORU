import glob
import os

from PIL import Image
from tqdm import tqdm

# リサイズしたいフォルダのパス
folder_path = "E:/wing_extension_projects_231029/wing_extension"

# リサイズ後の画像を保存するフォルダのパス (存在しない場合は作成されます)
output_folder = "E:/wing_extension_projects_231029/wing_extension_resize"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# フォルダ内の全てのPNGファイルを取得
for file_path in tqdm(glob.glob(os.path.join(folder_path, "*.png"))):
    # 画像を開く
    img = Image.open(file_path)

    # 画像をリサイズ (アスペクト比を保持しない)
    img_resized = img.resize((640, 480))

    # リサイズした画像を保存
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_folder, base_name)
    img_resized.save(output_path)

    # print(f"{file_path} -> {output_path}")

print("リサイズ完了!")
