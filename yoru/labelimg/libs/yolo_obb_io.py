# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) YORU contributors — see LICENSE for details.

"""Read and write YOLO-OBB (DOTA-style) label files.

One line per object, eight coordinates rather than four::

    class_index x1 y1 x2 y2 x3 y3 x4 y4

Every coordinate is normalised to ``[0, 1]`` by the image width (x) or height
(y), and the four corners are listed **in order around the rectangle** — the
same order as ``Shape.points`` in the canvas, which is what lets a rotated box
survive a save/load round trip without being re-derived from anything.  This is
exactly the format ultralytics reads for ``task="obb"``, so the files this
writes train a ``*-obb.pt`` model with no conversion step.

The extension is ``.txt``, the same as plain YOLO.  A file therefore cannot be
identified by its name, only by its contents, which :func:`is_obb_file` does by
counting fields: nine means OBB, five means axis-aligned.  ``labelimg.py`` uses
that to open either kind of project correctly even when the format button says
the other one.
"""

import codecs
import os

from .constants import DEFAULT_ENCODING

TXT_EXT = '.txt'
ENCODE_METHOD = DEFAULT_ENCODING

#: Fields in one line of each format: a class index plus its coordinates.
OBB_FIELDS = 9
AABB_FIELDS = 5


def sniff_obb(file_path):
    """``True`` for an OBB file, ``False`` for an axis-aligned one, ``None``
    when the file says nothing either way.

    The three-valued answer is the point.  An image with no objects on it has
    an empty label file, and an empty file in an OBB project must **not** be
    read as "this project is axis-aligned" — that would flip the format on the
    fly and start writing upright boxes over a rotated dataset.  ``None`` tells
    the caller to keep the format it already has.
    """
    try:
        with open(file_path, 'r', encoding=ENCODE_METHOD) as f:
            for line in f:
                fields = line.split()
                if not fields:
                    continue
                return len(fields) == OBB_FIELDS
    except (OSError, UnicodeDecodeError):
        return None
    return None


def is_obb_file(file_path):
    """``True`` only when the file positively holds OBB lines."""
    return sniff_obb(file_path) is True


class YoloOBBWriter:
    """Collects rotated boxes in pixels and writes them normalised."""

    def __init__(self, folder_name, filename, img_size, database_src='Unknown',
                 local_img_path=None):
        self.folder_name = folder_name
        self.filename = filename
        self.database_src = database_src
        #: ``(height, width, channels)`` — the order labelImg uses everywhere.
        self.img_size = img_size
        self.box_list = []
        self.local_img_path = local_img_path
        self.verified = False

    def add_obb(self, points, name, difficult):
        """Add one box from its four corners, in pixels."""
        self.box_list.append({
            'points': [(float(p[0]), float(p[1])) for p in points],
            'name': name,
            'difficult': difficult,
        })

    def obb_to_yolo_line(self, box, class_list=None):
        """``(class_index, [x1, y1, ... x4, y4])``, normalised and clamped.

        Every coordinate is clamped into ``[0, 1]``, because ultralytics
        rejects a label file with anything outside the unit square
        (``verify_image_label`` refuses "non-normalized or out of bounds
        coordinates") and one such line would cost the whole image.

        **For a rotated box this clamp is not free**, and the trade-off is
        deliberate.  Clamping an axis-aligned box just trims it at the frame
        edge; clamping one corner of a *rotated* box moves that corner while
        its neighbours stay put, which leaves a quadrilateral very slightly out
        of true.  The alternatives are worse: sliding the whole box back inside
        would move it off the animal, and dropping the box would lose the
        annotation outright.  Clamping keeps each corner at the nearest legal
        point, so the box stays where the user put it.  Ultralytics reduces the
        four corners to ``xywhr`` on load, which absorbs a sub-pixel shear.

        In practice this only bites for an animal drawn hard against the frame
        edge: labelImg keeps a dragged corner inside the image, so a box away
        from the edge is already within bounds and the clamp does nothing.
        """
        if class_list is None:
            class_list = []
        box_name = box['name']
        if box_name not in class_list:
            class_list.append(box_name)
        class_index = class_list.index(box_name)

        height, width = self.img_size[0], self.img_size[1]
        coords = []
        for x, y in box['points']:
            coords.append(min(max(x / width, 0.0), 1.0))
            coords.append(min(max(y / height, 0.0), 1.0))
        return class_index, coords

    def save(self, class_list=None, target_file=None):
        if class_list is None:
            class_list = []
        if target_file is None:
            target_file = self.filename + TXT_EXT

        classes_file = os.path.join(
            os.path.dirname(os.path.abspath(target_file)), "classes.txt")

        with codecs.open(target_file, 'w', encoding=ENCODE_METHOD) as out_file:
            for box in self.box_list:
                class_index, coords = self.obb_to_yolo_line(box, class_list)
                out_file.write(
                    "%d %s\n" % (class_index, " ".join("%.6f" % c for c in coords))
                )

        with codecs.open(classes_file, 'w', encoding=ENCODE_METHOD) as out_class_file:
            for c in class_list:
                out_class_file.write(c + '\n')


class YoloOBBReader:
    """Parses a YOLO-OBB file into labelImg's ``(label, points, ...)`` shapes."""

    def __init__(self, file_path, image, class_list_path=None):
        self.shapes = []
        self.file_path = file_path

        if class_list_path is None:
            dir_path = os.path.dirname(os.path.realpath(self.file_path))
            self.class_list_path = os.path.join(dir_path, "classes.txt")
        else:
            self.class_list_path = class_list_path

        with open(self.class_list_path, 'r', encoding=ENCODE_METHOD) as classes_file:
            self.classes = classes_file.read().strip('\n').split('\n')

        self.img_size = [image.height(), image.width(),
                         1 if image.isGrayscale() else 3]
        self.verified = False
        self.parse_yolo_obb_format()

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, points, difficult):
        self.shapes.append((label, points, None, None, difficult))

    def yolo_obb_line_to_shape(self, class_index, coords):
        """Denormalise one line back to pixel corners.

        Unlike the axis-aligned reader this keeps floats: rounding each corner
        to an integer would shear a rotated rectangle by up to half a pixel per
        corner, so a box that was merely loaded and saved again would drift.
        """
        index = int(class_index)
        label = self.classes[index] if 0 <= index < len(self.classes) else str(index)
        height, width = self.img_size[0], self.img_size[1]
        points = [
            (float(coords[i]) * width, float(coords[i + 1]) * height)
            for i in range(0, 8, 2)
        ]
        return label, points

    def parse_yolo_obb_format(self):
        with open(self.file_path, 'r', encoding=ENCODE_METHOD) as bnd_box_file:
            for line in bnd_box_file:
                fields = line.split()
                if len(fields) != OBB_FIELDS:
                    # Skip blanks and any stray axis-aligned line rather than
                    # raising: one bad line must not cost the user the whole
                    # image's annotations.
                    continue
                label, points = self.yolo_obb_line_to_shape(fields[0], fields[1:])
                # The difficult flag has no column in this format, as in plain
                # YOLO.
                self.add_shape(label, points, False)
