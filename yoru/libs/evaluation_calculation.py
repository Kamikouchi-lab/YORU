import glob
import itertools
import json
import os
import time
import tkinter as tk
from collections import Counter
from tkinter import filedialog

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from tqdm import tqdm


class yolo_analysis_image:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        self.yolo_model_path = self.m_dict["model_path"]
        self.data_path = self.m_dict["data_dir"]
        print("init")

    def analyze_image(self):
        yolo_model = torch.hub.load(
            "./libs/yolov5", "custom", path=self.yolo_model_path, source="local"
        )

        # クラス名の取得
        class_names = (
            yolo_model.module.names
            if hasattr(yolo_model, "module")
            else yolo_model.names
        )

        search_path = os.path.join(self.data_path, "*.png")
        img_path_list = glob.glob(search_path)
        image_count = len(img_path_list)

        for img_path in tqdm(img_path_list, desc="Processing images"):
            base_name = os.path.basename(img_path)
            file_name_without_ext = os.path.splitext(base_name)[0]

            frame = cv2.imread(img_path)
            height, width, channels = frame.shape

            yolo_result = yolo_model(frame)

            # 出力パスの作成
            result_txt_path = os.path.join(
                self.data_path, file_name_without_ext + "_yolo.txt"
            )
            result = []
            for *box, conf, cls in yolo_result.xywhn[0]:
                # xyxy形式（中心x, 中心y, 幅, 高さ）のリスト
                class_name = class_names[int(cls.item())]
                # 結果をリストに保存
                result.append(
                    [
                        int(cls.item()),
                        conf.item(),
                        box[0].item(),
                        box[1].item(),
                        box[2].item(),
                        box[3].item(),
                    ]
                )

            # print(result)
            with open(result_txt_path, "w") as file:
                for sublist in result:
                    file.write(" ".join(map(str, sublist)) + "\n")

        print("Complete!")


class ModelValidation:
    def __init__(self, m_dict):
        self.m_dict = m_dict

    def calculate_iou(self, boxA, boxB):
        """Calculate Intersection over Union (IoU) for two bounding boxes."""
        x1_A, y1_A, x2_A, y2_A = boxA
        x1_B, y1_B, x2_B, y2_B = boxB

        x1_int = max(x1_A, x1_B)
        y1_int = max(y1_A, y1_B)
        x2_int = min(x2_A, x2_B)
        y2_int = min(y2_A, y2_B)

        intersection = max(0, x2_int - x1_int) * max(0, y2_int - y1_int)
        area_A = (x2_A - x1_A) * (y2_A - y1_A)
        area_B = (x2_B - x1_B) * (y2_B - y1_B)

        iou = intersection / (area_A + area_B - intersection)

        return iou

    def convert_to_corners(self, box):
        x_center, y_center, width, height = box
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]

    def calculate_tp_fp(self, gt_boxes, pred_boxes, iou_threshold):
        """
        gt_boxes: 真のバウンディングボックスのリスト
        pred_boxes: 予測されたバウンディングボックスのリスト。各ボックスは(score, x1, y1, x2, y2)の形式。
        iou_threshold: IoUのしきい値
        iou_list: Iouのリスト
        tp: True positive
        fp: False positive
        """
        if len(pred_boxes) < 1:
            tp = []
            fp = []
            iou_list = []
            return tp, fp, iou_list

        # IOUの結果のリスト
        iou_list = []

        # 予測ボックスをスコアでソート
        pred_boxes = sorted(pred_boxes, key=lambda x: x[0], reverse=True)

        tp = np.zeros(len(pred_boxes))
        # print(len(tp))
        fp = np.zeros(len(pred_boxes))
        matched = []
        # a = 0

        for i, pred_box in enumerate(pred_boxes):
            max_iou = -np.inf
            max_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                float_gt_box = [float(item) for item in gt_box[1:]]
                float_pred_box = [float(item) for item in pred_box[2:]]
                gt_box_corner = self.convert_to_corners(float_gt_box)
                pred_box_corner = self.convert_to_corners(float_pred_box)
                current_iou = self.calculate_iou(pred_box_corner, gt_box_corner)
                # print(gt_box_corner, pred_box_corner)

                if current_iou > max_iou:
                    max_iou = current_iou
                    max_gt_idx = j

            # IOUリストに追加
            if max_iou >= 0:
                iou_list.append(max_iou)

            # print(max_gt_idx)

            if max_iou >= iou_threshold:
                if max_gt_idx not in matched:
                    tp[i] = 1
                    matched.append(max_gt_idx)
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        return tp, fp, iou_list

        # print(tp)

    def calculate_precision_recall(self, tp, fp, box_num):
        tp_sum = np.sum(tp)
        fp_sum = np.sum(fp)

        if box_num < 1 or (tp_sum + fp_sum) < 1:
            recall = []
            precision = []
            return recall, precision
        recall = [tp_sum / float(box_num)]
        precision = [tp_sum / (tp_sum + fp_sum)]
        # print(recall, precision)

        return recall, precision


    def calculate_ap(self, recalls, precisions):
        """Interpolated AP - VOC 2010 way"""
        # 並び替える
        recalls = np.sort(recalls)
        precisions = np.sort(precisions)

        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        # Precisionの値を後ろから見ていき、現在の値より大きな値が見られた場合には現在の値をその大きな値に置き換えます
        for i in range(precisions.size - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])

        # 今回のrecallの値と前回のrecallの値の差分を取得
        indices = np.where(recalls[1:] != recalls[:-1])[0]

        # 各変化点でのprecisionの平均を取り、それをrecallの変化量で重み付けして合計する
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

        return ap



class Evaluator(ModelValidation):
    def __init__(self, m_dict):
        super().__init__(m_dict)
        print("Starting....")

    def evaluate(self, gt_boxes, pred_boxes, classes, model_base_name):
        iou_thresholds = np.arange(0.50, 1.00, 0.05)
        ap_50_95 = {}
        mAP_50_95 = {}
        ap_50 = {}
        ap_75 = {}
        iou_results = {}
        recalls_dict = {}
        precisions_dict = {}

        # 各クラスに対して評価を行う
        for class_id in tqdm(classes, desc="Processing images"):
            class_aps = []
            iou_per_class = []
            counter = 0
            for iou_thresh in iou_thresholds:
                all_tps = []
                all_fps = []
                total_gts = 0
                iou_thresh_hold_disp = int(iou_thresh * 100)

                for list_indx, pred_box_image in enumerate(pred_boxes):
                    gt_box_image = gt_boxes[list_indx]
                    class_gt_boxes = [
                        box
                        for box in gt_box_image
                        if int(float(box[0])) == int(class_id)
                    ]
                    class_pred_boxes = [
                        box
                        for box in pred_box_image
                        if int(float(box[0])) == int(class_id)
                    ]

                    tp, fp, iou_list = self.calculate_tp_fp(
                        class_gt_boxes, class_pred_boxes, iou_thresh
                    )

                    all_tps.extend(tp)
                    all_fps.extend(fp)
                    if counter == 0:
                        iou_list = [n for n in iou_list if n > 0]
                        iou_per_class.extend(iou_list)
                    total_gts += len(class_gt_boxes)

                # PrecisionとRecallの計算
                counter += 1
                tp_cumsum = np.cumsum(all_tps)
                fp_cumsum = np.cumsum(all_fps)
                recalls = tp_cumsum / total_gts
                precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

                # Precisionの補正
                precisions = np.concatenate(
                    ([0.0], np.maximum.accumulate(precisions[::-1]), [0.0])
                )
                recalls = np.concatenate(([0.0], recalls, [1.0]))

                self.plot_precision_recall_curve(
                    precisions,
                    recalls,
                    classes[class_id],
                    self.m_dict["pr_curve_dir"],
                    model_base_name,
                    iou_thresh_hold_disp,
                )

                # APの計算
                ap = np.sum((recalls[1:] - recalls[:-1]) * precisions[1:])
                class_aps.append(ap)

            ap_50_95[class_id] = class_aps
            ap_50[class_id] = class_aps[0]
            ap_75[class_id] = class_aps[5]
            mAP_50_95[class_id] = np.mean(class_aps)
            # iou_results[classes[class_id]] = iou_per_class
            iou_results[class_id] = iou_per_class

        mAP_50 = np.mean(list(ap_50.values()))
        mAP_75 = np.mean(list(ap_75.values()))

        return (
            {
                "AP@[.50:.05:.95]_per_class": ap_50_95,
                "mAP@[.50:.05:.95]_per_class": mAP_50_95,
                "AP@.50_per_class": ap_50,
                "mAP@.50": mAP_50,
                "AP@.75_per_class": ap_75,
                "mAP@.75": mAP_75,
            },
            iou_results,
            recalls_dict,
            precisions_dict,
        )


    def read_boxes_txt(self, box_path):
        boxes_list = []

        with open(box_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                boxes_list.append(line.strip().split())

        return boxes_list

    def dict_to_dataframe(self, data_dict, classes_dict):
        col_name = ["class", "value"]
        df = pd.DataFrame(
            [(key, value) for key, values in data_dict.items() for value in values],
            columns=col_name,
        )
        # クラス名を変換
        df["class_name"] = df["class"].map(classes_dict)
        return df

    def read_yolo_det_box_txt(self):
        yolo_model = torch.hub.load(
            "./libs/yolov5", "custom", path=self.m_dict["model_path"], source="local"
        )

        # クラス名の取得
        class_names = (
            yolo_model.module.names
            if hasattr(yolo_model, "module")
            else yolo_model.names
        )
        return class_names

    def save_dict_to_txt(self, dic, filename):
        with open(filename, "w") as file:
            json.dump(dic, file)

    def list_counter(self, list_of_lists):
        # Extract the first element of each sublist
        # print(list_of_lists)
        flattened_list = list(itertools.chain.from_iterable(list_of_lists))

        first_elements = [sublist[0] for sublist in flattened_list]

        element_counts = Counter(first_elements)

        return element_counts

    def keySort(self, dicts, reverse=False):
        dicts = sorted(dicts.items(), reverse=reverse)
        dicts = fruits = dict((x, y) for x, y in dicts)
        return dicts

    def create_info_text(self, class_names, gt_boxs_no, pred_boxs_no):
        gt_boxs_no = self.keySort(gt_boxs_no)
        pred_boxs_no = self.keySort(pred_boxs_no)

        print("Ground truth", gt_boxs_no)
        print("Predicted", pred_boxs_no)
        # Prepare the file content
        file_content = "Ground truth bounding box counts\n"
        for class_num, count in gt_boxs_no.items():
            class_name = class_names.get(int(class_num), "Unknown")
            file_content += f"{class_name}({class_num}): {count}\n"

        file_content += "\nPredicted bounding box counts\n"
        for class_num2, count2 in pred_boxs_no.items():
            class_name2 = class_names.get(int(class_num2), "Unknown")
            file_content += f"{class_name2}({class_num2}): {count2}\n"

        return file_content
    
    def count_correct_predictions(self, labels, predictions):
        """
        ラベルと予測ラベルの一致をカウントします。重複するラベルにも対応しています。
        """
        label_counts = Counter(labels)
        prediction_counts = Counter(predictions)
        correct = 0
        for label in label_counts:
            correct += min(label_counts[label], prediction_counts.get(label, 0))
        return correct


    def calculate_precision_recall(self, directory):
        total_labels = 0
        total_predictions = 0
        correct_predictions = 0

        for file in tqdm(os.listdir(directory)):
            if file.endswith(".png") or file.endswith(".jpg"):
                base_filename = file.split(".")[0]
                label_file = os.path.join(directory, base_filename + ".txt")
                yolo_file = os.path.join(directory, base_filename + "_yolo.txt")

                if os.path.exists(label_file) and os.path.exists(yolo_file):
                    with open(label_file, "r") as lf, open(yolo_file, "r") as yf:
                        labels = [int(line.split()[0]) for line in lf.readlines()]
                        predictions = [int(line.split()[0]) for line in yf.readlines()]

                        total_labels += len(labels)
                        total_predictions += len(predictions)

                        correct_predictions += self.count_correct_predictions(labels, predictions)

        if total_labels > 0 and total_predictions > 0:
            recall = correct_predictions / total_labels
            precision = correct_predictions / total_predictions
            return precision, recall, total_labels, total_predictions, correct_predictions
        else:
            return 0, 0, 0, 0, 0

    def run_evaluation(self, image_directory):
        # ディレクトリ内の全ての画像を取得

        precision, recall, total_labels, total_predictions, correct_predictions = self.calculate_precision_recall(image_directory)

        image_files = [
            f
            for f in os.listdir(image_directory)
            if f.endswith(".jpg") or f.endswith(".png")
        ]

        gt_boxes = []
        pred_boxes = []

        for img_file in image_files:
            # 対応するground truthとYOLOの結果のtxtファイル名を構築
            base_name = os.path.splitext(img_file)[0]
            gt_txt = os.path.join(image_directory, base_name + ".txt")
            yolo_txt = os.path.join(image_directory, base_name + "_yolo.txt")

            # これらのtxtファイルからデータを読み込む
            gt_boxes.append(self.read_boxes_txt(gt_txt))
            pred_boxes.append(self.read_boxes_txt(yolo_txt))

        # クラス名のリストを取得
        class_names = self.read_yolo_det_box_txt()
        print(class_names)

        # count bounding boxes
        gt_boxes_no = self.list_counter(gt_boxes)
        pred_boxes_no = self.list_counter(pred_boxes)

        # creating info txt file
        file_contents = self.create_info_text(class_names, gt_boxes_no, pred_boxes_no)
        info_directory = os.path.join(self.m_dict["result_dir"], "infomation.txt")
        with open(info_directory, "w") as file:
            file.write(file_contents)



        filename = os.path.basename(self.m_dict["model_path"])
        model_base_name = os.path.splitext(filename)[0]

        results, iou_res, recalls_dict, precisions_dict = self.evaluate(
            gt_boxes, pred_boxes, class_names, model_base_name
        )

        # add precision and recall data
        results["Precision of all"] = precision
        results["Recall of all"] = recall
        results["Total ground-truth labels"] = total_labels
        results["Total YOLO predictions"] = total_predictions
        results["Total correct predictions"] = correct_predictions

        print(results)

        result_directory = os.path.join(
            self.m_dict["result_dir"], model_base_name + ".txt"
        )
        self.save_dict_to_txt(results, result_directory)
        print(result_directory)

        # iou listの出力
        iou_dataframe = self.dict_to_dataframe(iou_res, class_names)
        result_directory_iou = os.path.join(
            self.m_dict["result_dir"], model_base_name + "_iou_results" + ".csv"
        )
        iou_dataframe.to_csv(result_directory_iou)



        # 結果の描写
        self.drawing_graph(
            results, class_names, self.m_dict["result_dir"], model_base_name
        )
        self.drawing_iou_boxplot_graph(
            iou_dataframe, class_names, self.m_dict["result_dir"], model_base_name
        )
        print("Complete!")

    def drawing_graph(self, results, classes_dict, result_dir, model_base_name):
        result = results["AP@[.50:.05:.95]_per_class"]
        x = [0.5 + i * 0.05 for i in range(10)]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
        # データのプロット
        for key, color in zip(result, colors):
            plt.plot(x, result[key], color=color, marker="o", label=classes_dict[key])

        # x軸の目盛りを設定
        x_ticks = np.arange(0.5, 0.95 + 0.05, 0.10)  # 0.5から0.95まで0.05刻み
        plt.xticks(x_ticks)
        plt.xlim(0.45, 1.00)
        plt.ylim(0, 1)

        # 軸のラベル
        plt.xlabel("IOU")
        plt.ylabel("AP")

        # タイトルと凡例
        plt.title("AP@[.50:.05:.95]")
        plt.legend()

        # グラフを保存
        result_directory = os.path.join(result_dir, model_base_name + "_ap50-95.png")
        plt.savefig(result_directory)

    def drawing_iou_boxplot_graph(
        self, dataframe, classes_dict, result_dir, model_base_name
    ):
        # # クラス名を変換
        # dataframe["class"] = dataframe["class"].map(classes_dict)

        # データのプロット
        sns.set()
        sns.set_style("whitegrid")
        sns.set_palette("Set3")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # y軸の目盛りを設定
        plt.ylim(0, 1)

        sns.boxplot(x="class_name", y="value", data=dataframe, showfliers=False, ax=ax)
        sns.stripplot(
            x="class_name", y="value", data=dataframe, jitter=True, color="black", ax=ax
        )

        # サンプル数の計算
        sample_counts = dataframe["class_name"].value_counts()

        # 各カテゴリの位置を取得
        categories = dataframe["class_name"].unique()

        # 各ボックスプロットのx=0のラインの少し上にサンプル数を表示
        for i, category in enumerate(categories):
            # カテゴリに対応するサンプル数
            count = sample_counts[category]

            # カテゴリの位置を取得
            category_pos = i

            # テキストの位置を設定（x軸の位置とy軸の少し下）
            ax.text(
                category_pos,
                0.05,  # y軸の位置（0の少し下）
                f"n={count}",
                horizontalalignment="center",
                size="medium",  # テキストのサイズを大きく
                color="black",
                weight="semibold",
            )

        # 軸のラベル
        plt.xlabel("Class")
        plt.ylabel("IOU")

        # タイトルと凡例
        plt.title("IOU distributions")

        # グラフを保存
        result_directory = os.path.join(result_dir, model_base_name + "_iou_graph.png")
        fig.savefig(result_directory)
        plt.close()

    def plot_precision_recall_curve(
        self,
        precisions,
        recalls,
        class_name,
        result_dir,
        model_base_name,
        iou_threshold,
    ):
        """
        Precision-Recall 曲線を描画し、保存する関数。

        Args:
            precisions: Precision のリスト
            recalls: Recall のリスト
            class_name: クラス名
            result_dir: 結果を保存するディレクトリのパス
            model_base_name: モデルのベース名
        """
        # 図の準備
        plt.figure()
        plt.plot(recalls, precisions, marker=".", label=class_name)

        # 軸のラベル
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        # タイトルと凡例
        plt.title(f"Precision-Recall Curve - {class_name}")
        plt.legend()

        # グラフを保存
        file_path = os.path.join(
            result_dir,
            f"{model_base_name}_precision_recall_{class_name}_IOU{str(iou_threshold)}.png",
        )
        plt.savefig(file_path)
        plt.close()
