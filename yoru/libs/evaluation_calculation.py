# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

import itertools
import json
import logging
import os
from collections import Counter

import cv2
import matplotlib

# Evaluation writes figures to files, including from a GUI worker thread.
# An interactive Matplotlib backend would create another GUI on that thread.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from yoru.libs.plugins import get_detector
from yoru.libs.detector_base import obb_of
from yoru.libs.obb import obb_corners

logger = logging.getLogger(__name__)


class EvaluationImageAnalyzer:
    def __init__(self, m_dict):
        self.m_dict = m_dict
        self.yolo_model_path = self.m_dict["model_path"]
        self.data_path = self.m_dict["data_dir"]
        logger.debug("EvaluationImageAnalyzer initialized")

    def analyze_image(self):
        detector = get_detector("auto", self.yolo_model_path)

        # Get class names
        class_names = detector.names
        logger.debug("Class names: %s", class_names)

        # run_evaluation() below also accepts .jpg, so gather the same set here;
        # a JPEG dataset used to produce no _yolo.txt files at all.
        img_path_list = sorted(
            os.path.join(self.data_path, name) for name in os.listdir(self.data_path)
            if os.path.splitext(name)[1].lower() in (".png", ".jpg", ".jpeg")
        )
        logger.debug("Found %d images under %s", len(img_path_list), self.data_path)
        image_count = len(img_path_list)

        for img_path in tqdm(img_path_list, desc="Processing images"):
            if self.m_dict.get("quit", False):
                return
            base_name = os.path.basename(img_path)
            file_name_without_ext = os.path.splitext(base_name)[0]

            frame = cv2.imread(img_path)
            if frame is None:
                raise ValueError(f"Could not read evaluation image: {img_path}")
            height, width, channels = frame.shape

            detections = detector.detect(frame)

            # Create the output path
            result_txt_path = os.path.join(
                self.data_path, file_name_without_ext + "_yolo.txt"
            )
            result = []
            for d in detections:
                if d.get("angle") is not None:
                    corners = obb_corners(obb_of(d))
                    result.append([d["class_id"], d["conf"], *[
                        value for x, y in corners for value in (x / width, y / height)
                    ]])
                    continue
                # Convert to xywhn format (center x, center y, width, height), normalised
                x_center = (d["x1"] + d["x2"]) / 2 / width
                y_center = (d["y1"] + d["y2"]) / 2 / height
                w = (d["x2"] - d["x1"]) / width
                h = (d["y2"] - d["y1"]) / height
                # Save the result to the list
                result.append(
                    [
                        d["class_id"],
                        d["conf"],
                        x_center,
                        y_center,
                        w,
                        h,
                    ]
                )

            with open(result_txt_path, "w") as file:
                for sublist in result:
                    file.write(" ".join(map(str, sublist)) + "\n")

        logger.info("Evaluation image analysis complete!")


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

        denominator = area_A + area_B - intersection
        iou = intersection / denominator if denominator > 0 else 0.0

        return iou

    def convert_to_corners(self, box):
        """Upright envelope; evaluation uses label_iou to preserve rotation."""
        values = [float(v) for v in box]
        if len(values) == 8:
            xs = values[0::2]
            ys = values[1::2]
            return [min(xs), min(ys), max(xs), max(ys)]
        x_center, y_center, width, height = values
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        return [x1, y1, x2, y2]

    def label_iou(self, box_a, box_b):
        """Geometric IoU of YOLO xywh or YOLO-OBB four-corner labels.

        Normalising x and y independently preserves intersection/union ratios,
        including for non-square images. Never refit a rectangle after that
        transformation: the normalised polygon can be a parallelogram.
        """
        def polygon(box):
            values = np.asarray(box, dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("Box coordinates must be finite")
            if len(values) == 4:
                x, y, w, h = values
                if w < 0 or h < 0:
                    raise ValueError("Box dimensions must be non-negative")
                values = np.array([[x-w/2, y-h/2], [x+w/2, y-h/2],
                                   [x+w/2, y+h/2], [x-w/2, y+h/2]], dtype=np.float32)
            elif len(values) == 8:
                values = values.reshape(4, 2)
            else:
                raise ValueError("Expected xywh or four corner coordinates")
            # Work above OpenCV's small-coordinate intersection tolerances.
            return cv2.convexHull(values.reshape(4, 2) * 1024.0)

        a, b = polygon(box_a), polygon(box_b)
        area_a, area_b = cv2.contourArea(a), cv2.contourArea(b)
        if area_a <= 0 or area_b <= 0:
            return 0.0
        intersection, _ = cv2.intersectConvexConvex(a, b)
        intersection = min(max(float(intersection), 0.0), area_a, area_b)
        return intersection / (area_a + area_b - intersection)

    def calculate_tp_fp(self, gt_boxes, pred_boxes, iou_threshold):
        """
        gt_boxes: List of ground-truth bounding boxes
        Predictions are (class, score, *coordinates), GT is (class, *coordinates).
        Outputs follow descending confidence order.
        iou_threshold: IoU threshold
        iou_list: List of IoU values
        tp: True positive
        fp: False positive
        """
        if len(pred_boxes) < 1:
            tp = []
            fp = []
            iou_list = []
            return tp, fp, iou_list

        # List of IoU results
        iou_list = []

        # Sort predicted boxes by score
        pred_boxes = sorted(pred_boxes, key=lambda x: float(x[1]), reverse=True)

        tp = np.zeros(len(pred_boxes))
        fp = np.zeros(len(pred_boxes))
        matched = []

        for i, pred_box in enumerate(pred_boxes):
            max_iou = -np.inf
            max_gt_idx = -1

            for j, gt_box in enumerate(gt_boxes):
                if j in matched or int(float(gt_box[0])) != int(float(pred_box[0])):
                    continue
                current_iou = self.label_iou(pred_box[2:], gt_box[1:])

                if current_iou > max_iou:
                    max_iou = current_iou
                    max_gt_idx = j

            # Append to the IoU list
            if max_iou >= 0:
                iou_list.append(max_iou)

            if max_iou >= iou_threshold:
                if max_gt_idx not in matched:
                    tp[i] = 1
                    matched.append(max_gt_idx)
                else:
                    fp[i] = 1
            else:
                fp[i] = 1
        return tp, fp, iou_list

    def calculate_precision_recall(self, tp, fp, box_num):
        tp_sum = np.sum(tp)
        fp_sum = np.sum(fp)

        if box_num < 1 or (tp_sum + fp_sum) < 1:
            recall = []
            precision = []
            return recall, precision
        recall = [tp_sum / float(box_num)]
        precision = [tp_sum / (tp_sum + fp_sum)]
        return recall, precision

    def calculate_ap(self, recalls, precisions):
        """Interpolated AP - VOC 2010 way"""
        recalls = np.concatenate(([0.0], recalls, [1.0]))
        precisions = np.concatenate(([0.0], precisions, [0.0]))

        # Scan the precision values from the end; if a larger value is found, replace the current value with that larger value
        for i in range(precisions.size - 2, -1, -1):
            precisions[i] = np.maximum(precisions[i], precisions[i + 1])

        # Get the difference between the current recall value and the previous recall value
        indices = np.where(recalls[1:] != recalls[:-1])[0]

        # Take the precision at each change point, weight it by the change in recall, and sum the result
        ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])

        return ap


class Evaluator(ModelValidation):
    def __init__(self, m_dict):
        super().__init__(m_dict)
        logger.info("Starting evaluation...")

    def evaluate(self, gt_boxes, pred_boxes, classes, model_base_name):
        if len(gt_boxes) != len(pred_boxes):
            raise ValueError("Ground truth and predictions must cover the same images")
        thresholds = np.linspace(0.5, 0.95, 10)
        aps, iou_results, recalls_dict, precisions_dict = {}, {}, {}, {}
        for class_id in tqdm(classes, desc="Processing classes"):
            gt = [[b for b in image if int(float(b[0])) == int(class_id)]
                  for image in gt_boxes]
            pred = [sorted((b for b in image if int(float(b[0])) == int(class_id)),
                           key=lambda b: float(b[1]), reverse=True)
                    for image in pred_boxes]
            total_gt = sum(map(len, gt))
            scores = [float(b[1]) for image in pred for b in image]
            order = np.argsort(-np.asarray(scores), kind="stable")
            aps[class_id] = []
            iou_results[class_id] = []
            for index, threshold in enumerate(thresholds):
                if self.m_dict.get("quit", False):
                    raise InterruptedError("Evaluation cancelled")
                tps, fps = [], []
                for truth, predictions in zip(gt, pred):
                    if self.m_dict.get("quit", False):
                        raise InterruptedError("Evaluation cancelled")
                    tp, fp, ious = self.calculate_tp_fp(truth, predictions, threshold)
                    tps.extend(tp)
                    fps.extend(fp)
                    if index == 0:
                        iou_results[class_id].extend(ious)
                tp = np.cumsum(np.asarray(tps)[order])
                fp = np.cumsum(np.asarray(fps)[order])
                recall = tp / total_gt if total_gt else np.zeros_like(tp)
                precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
                # Classes without annotations have undefined AP and do not enter mAP.
                aps[class_id].append(float(self.calculate_ap(recall, precision))
                                     if total_gt else None)
                if index == 0:
                    recalls_dict[class_id] = recall.tolist()
                    precisions_dict[class_id] = precision.tolist()
                plot_recall = np.r_[0.0, recall, 1.0]
                plot_precision = np.r_[0.0, precision, 0.0]
                plot_precision = np.maximum.accumulate(plot_precision[::-1])[::-1]
                self.plot_precision_recall_curve(
                    plot_precision, plot_recall, classes[class_id],
                    self.m_dict["pr_curve_dir"], model_base_name, round(threshold * 100),
                )

        ap50 = {c: values[0] for c, values in aps.items()}
        ap75 = {c: values[5] for c, values in aps.items()}
        per_class = {c: float(np.mean(values)) if values[0] is not None else None
                     for c, values in aps.items()}

        def mean_defined(values):
            values = [v for v in values if v is not None]
            return float(np.mean(values)) if values else None

        return ({
            "AP@[.50:.05:.95]_per_class": aps,
            "mAP@[.50:.05:.95]_per_class": per_class,
            "AP@.50_per_class": ap50, "mAP@.50": mean_defined(ap50.values()),
            "AP@.75_per_class": ap75, "mAP@.75": mean_defined(ap75.values()),
            "mAP@[.50:.05:.95]": mean_defined(per_class.values()),
            "IoU method": "polygon (rotation preserved)",
            "AP method": "all-point interpolated precision envelope",
        }, iou_results, recalls_dict, precisions_dict)

    def read_boxes_txt(self, box_path):
        boxes_list = []

        with open(box_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                if line.strip():
                    boxes_list.append(line.strip().split())

        return boxes_list

    def dict_to_dataframe(self, data_dict, classes_dict):
        col_name = ["class", "value"]
        df = pd.DataFrame(
            [(key, value) for key, values in data_dict.items() for value in values],
            columns=col_name,
        )
        # Convert class names
        df["class_name"] = df["class"].map(classes_dict)
        return df

    def read_yolo_det_box_txt(self):
        detector = get_detector("auto", self.m_dict["model_path"])

        # Get class names
        class_names = detector.names
        return class_names

    def save_dict_to_txt(self, dic, filename):
        with open(filename, "w") as file:
            json.dump(dic, file)

    def list_counter(self, list_of_lists):
        # Extract the first element of each sublist
        flattened_list = list(itertools.chain.from_iterable(list_of_lists))

        first_elements = [sublist[0] for sublist in flattened_list]

        element_counts = Counter(first_elements)

        return element_counts

    def keySort(self, dicts, reverse=False):
        dicts = sorted(dicts.items(), reverse=reverse)
        dicts = dict((x, y) for x, y in dicts)
        return dicts

    def create_info_text(self, class_names, gt_boxs_no, pred_boxs_no):
        gt_boxs_no = self.keySort(gt_boxs_no)
        pred_boxs_no = self.keySort(pred_boxs_no)

        logger.info("Ground truth: %s", gt_boxs_no)
        logger.info("Predicted: %s", pred_boxs_no)
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
        Count matches between labels and predicted labels. Handles duplicate labels as well.
        """
        label_counts = Counter(labels)
        prediction_counts = Counter(predictions)
        correct = 0
        for label in label_counts:
            correct += min(label_counts[label], prediction_counts.get(label, 0))
        return correct

    def calculate_precision_recall(self, directory):
        """Micro precision/recall with one-to-one, same-class matches at IoU .5."""
        total_labels = total_predictions = correct_predictions = 0
        for filename in sorted(os.listdir(directory)):
            if self.m_dict.get("quit", False):
                raise InterruptedError("Evaluation cancelled")
            if os.path.splitext(filename)[1].lower() not in (".png", ".jpg", ".jpeg"):
                continue
            base = os.path.join(directory, os.path.splitext(filename)[0])
            truth = self.read_boxes_txt(base + ".txt")
            predictions = self.read_boxes_txt(base + "_yolo.txt")
            tp, _, _ = self.calculate_tp_fp(truth, predictions, 0.5)
            total_labels += len(truth)
            total_predictions += len(predictions)
            correct_predictions += int(np.sum(tp))
        precision = correct_predictions / total_predictions if total_predictions else 0.0
        recall = correct_predictions / total_labels if total_labels else 0.0
        return precision, recall, total_labels, total_predictions, correct_predictions

    def run_evaluation(self, image_directory):
        # Get all images in the directory

        precision, recall, total_labels, total_predictions, correct_predictions = (
            self.calculate_precision_recall(image_directory)
        )

        image_files = [
            f
            for f in sorted(os.listdir(image_directory))
            if os.path.splitext(f)[1].lower() in (".jpg", ".png", ".jpeg")
        ]

        gt_boxes = []
        pred_boxes = []

        for img_file in image_files:
            # Build the txt file names for the corresponding ground truth and YOLO results
            base_name = os.path.splitext(img_file)[0]
            gt_txt = os.path.join(image_directory, base_name + ".txt")
            yolo_txt = os.path.join(image_directory, base_name + "_yolo.txt")

            # Read the data from these txt files
            gt_boxes.append(self.read_boxes_txt(gt_txt))
            pred_boxes.append(self.read_boxes_txt(yolo_txt))

        # Get the list of class names
        class_names = self.read_yolo_det_box_txt()
        logger.debug("Class names: %s", class_names)

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

        logger.info("Results: %s", results)

        result_directory = os.path.join(
            self.m_dict["result_dir"], model_base_name + ".txt"
        )
        self.save_dict_to_txt(results, result_directory)
        logger.info("Saved results to: %s", result_directory)

        # Output the iou list
        iou_dataframe = self.dict_to_dataframe(iou_res, class_names)
        result_directory_iou = os.path.join(
            self.m_dict["result_dir"], model_base_name + "_iou_results" + ".csv"
        )
        iou_dataframe.to_csv(result_directory_iou)

        # Plot the results
        self.drawing_graph(
            results, class_names, self.m_dict["result_dir"], model_base_name
        )
        self.drawing_iou_boxplot_graph(
            iou_dataframe, class_names, self.m_dict["result_dir"], model_base_name
        )
        logger.info("Evaluation complete!")

    def drawing_graph(self, results, classes_dict, result_dir, model_base_name):
        result = results["AP@[.50:.05:.95]_per_class"]
        x = [0.5 + i * 0.05 for i in range(10)]
        colors = plt.cm.rainbow(np.linspace(0, 1, len(results)))
        # Draw on a fresh figure: without this the plot was added to whatever
        # figure was current, so a second evaluation overlaid the first.
        plt.figure()
        # Plot the data
        for key, color in zip(result, colors):
            plt.plot(x, result[key], color=color, marker="o", label=classes_dict[key])

        # Set the x-axis ticks
        x_ticks = np.arange(0.5, 0.95 + 0.05, 0.10)  # From 0.5 to 0.95 in steps of 0.05
        plt.xticks(x_ticks)
        plt.xlim(0.45, 1.00)
        plt.ylim(0, 1)

        # Axis labels
        plt.xlabel("IOU")
        plt.ylabel("AP")

        # Title and legend
        plt.title("AP@[.50:.05:.95]")
        plt.legend()

        # Save the graph
        result_directory = os.path.join(result_dir, model_base_name + "_ap50-95.png")
        plt.savefig(result_directory)
        plt.close()

    def drawing_iou_boxplot_graph(
        self, dataframe, classes_dict, result_dir, model_base_name
    ):
        # Plot the data
        sns.set()
        sns.set_style("whitegrid")
        sns.set_palette("Set3")
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        # Set the y-axis ticks
        plt.ylim(0, 1)

        if not dataframe.empty:
            sns.boxplot(x="class_name", y="value", data=dataframe, showfliers=False, ax=ax)
            sns.stripplot(
                x="class_name", y="value", data=dataframe, jitter=True, color="black", ax=ax
            )

        # Calculate the sample counts
        sample_counts = dataframe["class_name"].value_counts()

        # Get the position of each category
        categories = dataframe["class_name"].unique()

        # Display the sample count slightly above the x=0 line of each box plot
        for i, category in enumerate(categories):
            # Sample count corresponding to the category
            count = sample_counts[category]

            # Get the position of the category
            category_pos = i

            # Set the text position (x-axis position and slightly below the y-axis)
            ax.text(
                category_pos,
                0.05,  # y-axis position (slightly below 0)
                f"n={count}",
                horizontalalignment="center",
                size="medium",  # Enlarge the text size
                color="black",
                weight="semibold",
            )

        # Axis labels
        plt.xlabel("Class")
        plt.ylabel("IOU")

        # Title and legend
        plt.title("IOU distributions")

        # Save the graph
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
        Function that draws and saves the Precision-Recall curve.

        Args:
            precisions: List of precision values
            recalls: List of recall values
            class_name: Class name
            result_dir: Path to the directory where results are saved
            model_base_name: Base name of the model
        """
        # Prepare the figure
        plt.figure()
        plt.plot(recalls, precisions, marker=".", label=class_name)

        # Axis labels
        plt.xlabel("Recall")
        plt.ylabel("Precision")

        # Title and legend
        plt.title(f"Precision-Recall Curve - {class_name}")
        plt.legend()

        # Save the graph
        file_path = os.path.join(
            result_dir,
            f"{model_base_name}_precision_recall_{class_name}_IOU{str(iou_threshold)}.png",
        )
        plt.savefig(file_path)
        plt.close()
