import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch


class yolo_drawing:
    def __init__(self, m_dict={}):
        self.m_dict = m_dict
        # self.yolo_model_path = self.m_dict["yolo_model"]

    def get_colormap(self, label_names, colormap_name):
        colormap = {}
        cmap = plt.get_cmap(colormap_name)
        label_ids = list(range(len(label_names)))
        for i in range(len(label_ids)):
            rgb = [int(d) for d in np.array(cmap(float(i) / len(label_ids))) * 255][:3]
            colormap[label_ids[i]] = tuple(rgb)

        return colormap

    def drawing(self, img, results):
        # print(results)
        for *box, conf, cls, name, time in results:
            label = f"{self.names[int(cls)]} {conf:.2f}"
            # label = f"{name} {conf:.2f}

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

    def YOLOdraw(self, m_dict):
        print("YOLO detection start...")

        while True:
            if self.m_dict["yolo_process_state"]:
                self.m_dict = m_dict

                # with torch.no_grad():
                while True:
                    self.names = self.m_dict["class_name_list"]
                    self.colormap = self.get_colormap(self.names, "gist_rainbow")

                    image = self.m_dict["current_camera_frame"]
                    results = self.m_dict["yolo_results"]
                    # print(self.m_dict["yolo_results"])

                    if image.any() & self.m_dict["yolo_detection"]:
                        # self.m_dict["yolo_results"] = self.detection_result.xyxy[0].detach().cpu().numpy()
                        # cv2.imshow("image", image)
                        # print(self.m_dict["yolo_results"])
                        image_result = self.drawing(image, results)
                        self.m_dict["yolo_detection_frame"] = image_result
                        self.m_dict["now"] = time.perf_counter()

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    elif self.m_dict["quit"]:
                        break
                    elif not self.m_dict["yolo_process_state"]:
                        print("YOLO drawing break")
                        break
            if self.m_dict["quit"]:
                break


if __name__ == "__main__":
    imgWin = yolo_detection(m_dict=d)
    imgWin.detect()
