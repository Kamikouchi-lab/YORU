import os
import xml.etree.ElementTree as ET
from tkinter import Tk, filedialog


def convert(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x * dw
    w = w * dw
    y = y * dh
    h = h * dh
    return (x, y, w, h)


def convert_xml2yolo(xml_file_path):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    size = root.find("size")
    w = int(size.find("width").text)
    h = int(size.find("height").text)

    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        # この部分でクラス名をIDに変換します。YOLOのlabels.txtまたは.namesファイルを参照することで
        # クラスIDを取得できます。ここでは簡単のため "mouse" は 0 とします。
        cls_id = 0 if cls_name == "mouse" else -1

        if cls_id == -1:
            print("Warning: class name not recognized!")
            continue

        xmlbox = obj.find("bndbox")
        b = (
            float(xmlbox.find("xmin").text),
            float(xmlbox.find("xmax").text),
            float(xmlbox.find("ymin").text),
            float(xmlbox.find("ymax").text),
        )
        converted = convert((w, h), b)

        with open(xml_file_path.replace(".xml", ".txt"), "a") as txt_file:
            txt_file.write(f"{cls_id} {' '.join([str(a) for a in converted])}\n")


def main():
    root = Tk()
    root.withdraw()  # GUIウィンドウを表示しないようにする

    folder_path = filedialog.askdirectory(title="Choose folder containing xml files")
    if not folder_path:
        return

    for filename in os.listdir(folder_path):
        if filename.endswith(".xml"):
            convert_xml2yolo(os.path.join(folder_path, filename))

    print("Conversion completed.")


if __name__ == "__main__":
    main()
