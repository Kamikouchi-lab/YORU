# Copyright (c) 2016 Tzutalin
# Create by TzuTaLin <tzu.ta.lin@gmail.com>

try:
    from PyQt5.QtGui import QImage
except ImportError:
    from PyQt4.QtGui import QImage

from base64 import b64encode, b64decode
from .pascal_voc_io import PascalVocWriter
from .yolo_io import YOLOWriter
from .yolo_obb_io import YoloOBBWriter
from .pascal_voc_io import XML_EXT
from .create_ml_io import CreateMLWriter
from .create_ml_io import JSON_EXT
from enum import Enum
import os.path
import sys


class LabelFileFormat(Enum):
    PASCAL_VOC = 1
    YOLO = 2
    CREATE_ML = 3
    #: YOLO-OBB (DOTA-style): one class index plus four corners per line.
    #: Shares the .txt extension with YOLO — see libs/yolo_obb_io.py.
    YOLO_OBB = 4


def coerce_label_file_format(value, default=None):
    """A :class:`LabelFileFormat` for whatever the settings file held.

    The remembered format is *pickled*, and a settings file written by a
    different labelImg build (the pip-installed ``labelImg``, say, whose enum
    lives in ``libs.labelFile``) unpickles as a member of **that** class, which
    compares unequal to every member of this one.  Every lookup then falls
    through and the window crashes before it opens.  Matching by name, then by
    value, recovers the user's actual choice; anything still unrecognised falls
    back to YOLO rather than to nothing.
    """
    if default is None:
        default = LabelFileFormat.YOLO
    if isinstance(value, LabelFileFormat):
        return value
    name = getattr(value, "name", None)
    if isinstance(name, str) and name in LabelFileFormat.__members__:
        return LabelFileFormat[name]
    try:
        return LabelFileFormat(getattr(value, "value", value))
    except (ValueError, TypeError):
        return default


class LabelFileError(Exception):
    pass


class LabelFile(object):
    # It might be changed as window creates. By default, using XML ext
    # suffix = '.lif'
    suffix = XML_EXT

    def __init__(self, filename=None):
        self.shapes = ()
        self.image_path = None
        self.image_data = None
        self.verified = False

    def save_create_ml_format(self, filename, shapes, image_path, image_data, class_list, line_color=None, fill_color=None, database_src=None):
        img_folder_name = os.path.basename(os.path.dirname(image_path))
        img_file_name = os.path.basename(image_path)

        image = QImage()
        image.load(image_path)
        image_shape = [image.height(), image.width(),
                       1 if image.isGrayscale() else 3]
        writer = CreateMLWriter(img_folder_name, img_file_name,
                                image_shape, shapes, filename, local_img_path=image_path)
        writer.verified = self.verified
        writer.write()


    def save_pascal_voc_format(self, filename, shapes, image_path, image_data,
                               line_color=None, fill_color=None, database_src=None):
        img_folder_path = os.path.dirname(image_path)
        img_folder_name = os.path.split(img_folder_path)[-1]
        img_file_name = os.path.basename(image_path)
        if isinstance(image_data, QImage):
            image = image_data
        else:
            image = QImage()
            image.load(image_path)
        image_shape = [image.height(), image.width(),
                       1 if image.isGrayscale() else 3]
        writer = PascalVocWriter(img_folder_name, img_file_name,
                                 image_shape, local_img_path=image_path)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            # Add Chris
            difficult = int(shape['difficult'])
            bnd_box = LabelFile.convert_points_to_bnd_box(points)
            writer.add_bnd_box(bnd_box[0], bnd_box[1], bnd_box[2], bnd_box[3], label, difficult)

        writer.save(target_file=filename)
        return

    def save_yolo_format(self, filename, shapes, image_path, image_data, class_list,
                         line_color=None, fill_color=None, database_src=None):
        img_folder_path = os.path.dirname(image_path)
        img_folder_name = os.path.split(img_folder_path)[-1]
        img_file_name = os.path.basename(image_path)
        if isinstance(image_data, QImage):
            image = image_data
        else:
            image = QImage()
            image.load(image_path)
        image_shape = [image.height(), image.width(),
                       1 if image.isGrayscale() else 3]
        writer = YOLOWriter(img_folder_name, img_file_name,
                            image_shape, local_img_path=image_path)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            label = shape['label']
            # Add Chris
            difficult = int(shape['difficult'])
            bnd_box = LabelFile.convert_points_to_bnd_box(points)
            writer.add_bnd_box(bnd_box[0], bnd_box[1], bnd_box[2], bnd_box[3], label, difficult)

        writer.save(target_file=filename, class_list=class_list)
        return

    def save_yolo_obb_format(self, filename, shapes, image_path, image_data, class_list,
                             line_color=None, fill_color=None, database_src=None):
        """Write the shapes as rotated boxes, corner for corner.

        The corners go to the writer untouched.  Reducing them to a bounding
        box first — the way every other format here does — is precisely what
        would throw the rotation away, so this method exists instead of a
        branch inside :meth:`save_yolo_format`.
        """
        img_folder_path = os.path.dirname(image_path)
        img_folder_name = os.path.split(img_folder_path)[-1]
        img_file_name = os.path.basename(image_path)
        if isinstance(image_data, QImage):
            image = image_data
        else:
            image = QImage()
            image.load(image_path)
        image_shape = [image.height(), image.width(),
                       1 if image.isGrayscale() else 3]
        writer = YoloOBBWriter(img_folder_name, img_file_name,
                               image_shape, local_img_path=image_path)
        writer.verified = self.verified

        for shape in shapes:
            points = shape['points']
            if len(points) != 4:
                # An OBB file has no way to express anything but a rectangle.
                continue
            writer.add_obb(points, shape['label'], int(shape['difficult']))

        writer.save(target_file=filename, class_list=class_list)
        return

    def toggle_verify(self):
        self.verified = not self.verified

    @staticmethod
    def is_label_file(filename):
        file_suffix = os.path.splitext(filename)[1].lower()
        return file_suffix == LabelFile.suffix

    @staticmethod
    def convert_points_to_bnd_box(points):
        x_min = float('inf')
        y_min = float('inf')
        x_max = float('-inf')
        y_max = float('-inf')
        for p in points:
            x = p[0]
            y = p[1]
            x_min = min(x, x_min)
            y_min = min(y, y_min)
            x_max = max(x, x_max)
            y_max = max(y, y_max)

        # Martin Kersner, 2015/11/12
        # 0-valued coordinates of BB caused an error while
        # training faster-rcnn object detector.
        if x_min < 1:
            x_min = 1

        if y_min < 1:
            y_min = 1

        return int(x_min), int(y_min), int(x_max), int(y_max)
