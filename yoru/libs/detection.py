import time

import cv2
import dearpygui.dearpygui as dpg
import matplotlib.pyplot as plt
import numpy as np
import torch


class yolo_detection:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict
        self.yolo_model_path = self.m_dict["yolo_model"]

    def get_colormap(self, label_names, colormap_name):
        colormap = {}
        cmap = plt.get_cmap(colormap_name)
        label_ids = list(range(len(label_names)))
        for i in range(len(label_ids)):
            rgb = [int(d) for d in np.array(cmap(float(i) / len(label_ids))) * 255][:3]
            colormap[label_ids[i]] = tuple(rgb)

        return colormap

    def drawing(self, img, resutls):
        for *box, conf, cls in self.m_dict["yolo_results"]:
            label = f"{self.names[int(cls)]} {conf:.2f}"

            cv2.rectangle(
                img,
                pt1=(int(box[0]), int(box[1])),
                pt2=(int(box[2]), int(box[3])),
                color=self.colormap[int(cls)],
                thickness=4,
                lineType=cv2.LINE_4,
                shift=0,
            )
            cv2.putText(
                img,
                text=label,
                org=(int(box[0]), int(box[1]) - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1.5,
                color=self.colormap[int(cls)],
                thickness=5,
                lineType=cv2.LINE_4,
            )
            # print(label)
        # self.m_dict["yolo_detection_farme"] = img
        # cv2.imshow('prj_view2', img)
        return img

    def detect(self, m_dict):
        print("YOLO detection start...")

        while True:
            if self.m_dict["yolo_process_state"]:
                self.m_dict = m_dict
                self.yolo_model_path = self.m_dict["yolo_model"]
                print(self.m_dict["yolo_model"])

                # try:
                # dpg.disable_item("yolocheckbox")
                self.yolo_model = torch.hub.load(
                    "./libs/yolov5", "custom", path=self.yolo_model_path, source="local"
                )
                self.m_dict["class_list"] = (
                    self.yolo_model.module.names
                    if hasattr(self.yolo_model, "module")
                    else self.yolo_model.names
                )
                self.m_dict["class_name_list"] = list(
                    self.m_dict["class_list"].values()
                )
                print(self.m_dict["class_name_list"])
                # self.names = self.m_dict["class_name_list"]

                # self.colormap = self.get_colormap(self.names, "gist_rainbow")
                # # print(self.colormap)
                # dpg.enable_item("yolocheckbox")
                # except:
                #     pass

                with torch.no_grad():
                    while True:
                        image = self.m_dict["current_camera_frame"]
                        if image.any() & self.m_dict["yolo_detection"]:
                            self.detection_result = self.yolo_model(image)

                            result = (
                                self.detection_result.xyxy[0].detach().cpu().numpy()
                            )
                            class_ids = result[:, 5].astype(int)
                            # class_names = [model.names[class_id] for class_id in class_ids]
                            yoru_names_list = [self.m_dict["class_name_list"][i] for i in class_ids]

                            
                            # object型のNumPy配列を作成
                            yolo_results = np.empty((result.shape[0], result.shape[1] + 2), dtype=object)

                            # 座標部分をコピー
                            yolo_results[:, :-2] = result

                            # クラス名を追加
                            yolo_results[:, -2] = yoru_names_list
                            self.m_dict["yolo_class_names"] = yoru_names_list

                            yolo_results[:, -1] = self.m_dict["total_time"]

                            # 結果を保存
                            self.m_dict["yolo_results"] = yolo_results

                            # self.m_dict["yolo_results"] = [sublist + [self.m_dict["yolo_class_names"][i]] for i, sublist in enumerate(result)]

                            # print(self.m_dict["yolo_results"])
                            # print(self.m_dict["yolo_class_names"])
                            # image_result = self.drawing(image, self.m_dict["yolo_results"])
                            # self.m_dict["yolo_detection_frame"] = image_result
                            self.m_dict["now"] = time.perf_counter()

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                        elif self.m_dict["quit"]:
                            break
                        elif not self.m_dict["yolo_process_state"]:
                            print("YOLO break")
                            break
            if self.m_dict["quit"]:
                break


if __name__ == "__main__":
    imgWin = yolo_detection(m_dict=d)
    imgWin.detect()
