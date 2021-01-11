import os

import numpy as np
import cv2 as cv
import qimage2ndarray

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *


class BpzcvEggLayingWoolyMilkSow:

    def __init__(self, main_window):
        self._main_window = main_window
        self._shapes = None
        self._dirty = False

        self.widget = self._create_widget()
        self._main_window.addDockWidget(Qt.RightDockWidgetArea, self.widget)

        self._overridden_load_labels = self._main_window.loadLabels
        self._overridden_load_file = self._main_window.loadFile

        self._main_window.loadLabels = self._load_labels
        self._main_window.loadFile = self._load_file

    def _load_file(self, filePath=None):
        ok = self._overridden_load_file(filePath)
        if ok:
            self.frame = qimage2ndarray.byte_view(self._main_window.image)
        self._main_window.dirty = self._dirty
        self._dirty = False

    def _load_labels(self, shapes):
        if shapes is not None:
            self._overridden_load_labels(shapes)
        elif self.tracking_check_box.isChecked() and self._shapes is not None:
            self._shapes = self._track_shapes()
            self._overridden_load_labels(self._shapes)
            return
        self._shapes = shapes

    def _points_to_bbox(self, points):
        if len(points) != 4:
            raise ValueError('illegal amount of points')

        top_left = points[0]
        top_right = points[1]
        bot_left = points[3]

        return (top_left[0], top_left[1], top_right[0] - top_left[0], bot_left[1] - top_left[1])

    def _bbox_to_points(self, bbox):
        return (
            (bbox[0], bbox[1]),
            (bbox[0] + bbox[2], bbox[1]),
            (bbox[0] + bbox[2], bbox[1] + bbox[3]),
            (bbox[0], bbox[1] + bbox[3])
        )

    def _track_shapes(self):
        shapes = []

        for label, points, line_color, fill_color, difficult in self._shapes:
            if len(points) != 4:
                continue
            tracker = cv.TrackerMIL_create()
            tracker.init(self.frame, self._points_to_bbox(points))
            ok, bbox = tracker.update(qimage2ndarray.byte_view(self._main_window.image))

            if not ok:
                print('tracking failed')
                continue

            shapes.append((
                label, self._bbox_to_points(bbox), line_color, fill_color, difficult
            ))

            # force save dialog
            self._dirty = True

        return shapes

    def _create_widget(self):
        self.tracking_check_box = QCheckBox("Tracking")
        self.tracking_check_box.setChecked(False)

        tracking_config_list_layout = QVBoxLayout()
        tracking_config_list_layout.setContentsMargins(10, 10, 10, 10)
        tracking_config_list_layout.addWidget(self.tracking_check_box)

        tracking_config_container = QWidget()
        tracking_config_container.setLayout(tracking_config_list_layout)

        tracking_config_widget = QDockWidget("BPZCV", self._main_window)
        tracking_config_widget.setObjectName("BPZCV")
        tracking_config_widget.setWidget(tracking_config_container)
        return tracking_config_widget
